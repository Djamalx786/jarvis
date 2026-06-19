"""Telegram bot frontend for the Life OS Agent. Replies are always in German."""
import asyncio
import json
import logging
import os
import subprocess
from datetime import date, datetime, timedelta
from functools import partial

from dotenv import load_dotenv
from telegram import Update
from telegram.error import NetworkError, TimedOut
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    TypeHandler,
    filters,
)
from langgraph.types import Command

from src import calendar_service, calendar_tools, rag, reminders, scheduler
from src.agent import LLMError, chat as groq_chat, chat_with_tools as agent_chat_with_tools, transcribe_audio
from src.graph import build_graph
from src.startup import prepare_runtime

load_dotenv()
prepare_runtime()

STATE_FILE = "data/state.json"

agent_graph = build_graph()

# Per-chat ephemeral state (lost on restart, the LangGraph checkpointer keeps the real plan state)
pending_plans = set()      # chat_ids with a plan awaiting approval/feedback
pending_oauth = set()      # chat_ids that ran /connect and are waiting for the OAuth code
chat_histories = {}        # chat_id -> list of {"role", "content"}

HELP_TEXT = (
    "*Life OS Agent – Befehle*\n\n"
    "/plan – Erstellt deinen Wochenplan (mit Rückfrage zur Bestätigung)\n"
    "/today – Zeigt deine heutigen Kalendertermine\n"
    "/add <text> – Erstellt einen Termin per Texteingabe, z. B. `/add Zahnarzt Dienstag 15 Uhr`\n"
    "/connect – Verbindet dein Google Calendar Konto\n"
    "/status – Zeigt Verbindungsstatus, letzten Plan und Wissensbasis\n"
    "/merke <Text> – Merkt sich eine Notiz für künftige Pläne, z. B. `/merke Ich brauche mehr Pufferzeit am Montag`\n"
    "/checkin – Startet sofort einen Abend-Check-in (sonst automatisch um 21:45)\n"
    "/scheduler_status – Zeigt, wann die nächsten proaktiven Nachrichten kommen\n"
    "/help – Diese Übersicht\n\n"
    "Ich melde mich auch von selbst: morgens (Mo–Fr 07:00) mit deinen 3 Prioritäten, "
    "abends (21:45) zum Abschalten mit einer kurzen Tagesfrage und sonntags (19:00) zum "
    "Wochenrückblick. Was du abends/sonntags antwortest, fließt in deine nächsten Pläne ein.\n\n"
    "Du kannst mir auch einfach schreiben oder eine Sprachnachricht senden – z. B. "
    "_„Hab ich Mittwoch was vor?“_, _„Verschieb mein Training am Dienstag auf 18 Uhr“_ "
    "oder _„Lösch den Termin am Freitagvormittag“_. Ich verwalte dabei nur Termine, die "
    "ich selbst angelegt habe, und prüfe immer auf Überschneidungen."
)


# ── Helpers ──

async def _run_sync(func, *args, **kwargs):
    """Run a blocking function in a thread so the bot's event loop stays responsive."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))


async def _send_markdown_safe(message, text: str) -> None:
    """Send a message with Markdown formatting, falling back to plain text if parsing fails."""
    try:
        await message.reply_text(text, parse_mode="Markdown")
    except Exception:
        await message.reply_text(text)


def _load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(data: dict) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(data, f)


def _register_chat(chat_id: int) -> None:
    """Remember a chat_id so proactive reminders know where to send. Persisted across restarts."""
    state = _load_state()
    ids = set(state.get("chat_ids", []))
    if chat_id not in ids:
        ids.add(chat_id)
        state["chat_ids"] = sorted(ids)
        _save_state(state)


async def track_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """First-pass handler (group -1): record the chat for reminders, then let normal handlers run."""
    if update.effective_chat:
        _register_chat(update.effective_chat.id)


# ── Commands ──

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Remember where to send the proactive morning/evening/weekly messages.
    scheduler.save_chat_id(update.message.chat_id)
    await update.message.reply_text(
        "Hallo Djamal! 👋\n\nIch bin dein Life OS Agent.\nSchreib /help für eine Übersicht aller Befehle."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _send_markdown_safe(update.message, HELP_TEXT)


async def plan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.message.chat_id)
    config = {"configurable": {"thread_id": chat_id}}

    await update.message.reply_text("🧠 Ich erstelle deinen Wochenplan, einen Moment...")

    initial_state = {"user_input": "Erstelle einen Wochenplan für die kommende Woche", "iteration": 0, "feedback": ""}

    try:
        result = await _run_sync(agent_graph.invoke, initial_state, config=config)
    except Exception as e:
        await update.message.reply_text(f"❌ Fehler beim Erstellen des Plans: {e}")
        return

    await _send_plan_or_finish(update, result, chat_id)


async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not calendar_service.is_connected():
        await update.message.reply_text("📅 Google Calendar ist nicht verbunden. Nutze /connect, um es zu verbinden.")
        return

    try:
        events = await _run_sync(calendar_service.get_day_events, datetime.now())
    except calendar_service.CalendarError as e:
        await update.message.reply_text(f"❌ Termine konnten nicht geladen werden: {e}")
        return

    summary = calendar_service.format_events_summary(events)
    await _send_markdown_safe(update.message, f"*Heute, {datetime.now().strftime('%d.%m.%Y')}*\n\n{summary}")


async def add_event(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Nutzung: /add <Beschreibung>, z. B. `/add Zahnarzt Dienstag 15 Uhr`")
        return

    if not calendar_service.is_connected():
        await update.message.reply_text("📅 Google Calendar ist nicht verbunden. Nutze /connect, um es zu verbinden.")
        return

    text = " ".join(context.args)
    now = datetime.now()

    weekday_names = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
    date_table = "\n".join(
        f"- {(now + timedelta(days=i)).strftime('%Y-%m-%d')}: "
        f"{weekday_names[(now + timedelta(days=i)).weekday()]}"
        + (" (heute)" if i == 0 else " (morgen)" if i == 1 else "")
        for i in range(14)
    )

    system_prompt = (
        "Du extrahierst Termininformationen aus einer Texteingabe und antwortest AUSSCHLIESSLICH mit einem "
        "validen JSON-Objekt ohne Markdown-Codeblöcke, mit den Feldern:\n"
        '"title" (kurzer Titel auf Deutsch), "date" (Format YYYY-MM-DD), '
        '"start_time" (Format HH:MM), "end_time" (Format HH:MM; falls keine Dauer genannt wird, '
        '1 Stunde nach start_time), "description" (kurze Zusatzinfo oder leerer String).\n\n'
        "Hier ist eine Tabelle der nächsten 14 Tage mit Wochentag - nutze sie, um das passende Datum "
        f"für relative Angaben (z. B. 'Dienstag', 'morgen', 'nächste Woche Montag') nachzuschlagen:\n"
        f"{date_table}\n\n"
        "Wähle bei einem genannten Wochentag (z. B. 'Dienstag') ohne weiteren Zusatz das NÄCHSTE "
        "Vorkommen dieses Wochentags aus der Tabelle (auch wenn es heute schon dieser Wochentag ist, "
        "dann das kommende, nicht das heutige)."
    )

    try:
        raw = await _run_sync(
            groq_chat,
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
            json_mode=True,
        )
        event_data = json.loads(raw)
        event_date = date.fromisoformat(event_data["date"])
        await _run_sync(
            calendar_service.create_event,
            event_data["title"],
            event_date,
            event_data["start_time"],
            event_data["end_time"],
            event_data.get("description", ""),
        )
    except LLMError as e:
        await update.message.reply_text(f"❌ Groq ist aktuell nicht erreichbar: {e}")
        return
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        await update.message.reply_text(f"❌ Konnte den Termin nicht verstehen: {e}")
        return
    except calendar_service.CalendarError as e:
        await update.message.reply_text(f"❌ Termin konnte nicht erstellt werden: {e}")
        return

    await _send_markdown_safe(
        update.message,
        f"✅ Termin erstellt: *{event_data['title']}*\n"
        f"{event_date.strftime('%A, %d.%m.%Y')}, {event_data['start_time']}–{event_data['end_time']}",
    )


async def connect(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.message.chat_id)

    try:
        auth_url = await _run_sync(calendar_service.get_auth_url)
    except calendar_service.CalendarNotConfiguredError as e:
        await update.message.reply_text(f"❌ {e}")
        return

    pending_oauth.add(chat_id)
    await update.message.reply_text(
        "🔗 Öffne diesen Link, melde dich mit deinem Google-Konto an und erlaube den Kalenderzugriff:\n\n"
        f"{auth_url}\n\n"
        "Die Seite nach der Bestätigung lädt evtl. nicht – das ist normal. "
        "Kopiere den `code`-Parameter aus der Adresszeile (oder die ganze URL) und schick ihn mir hier."
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    connected = calendar_service.is_connected()
    state = _load_state()
    last_plan = state.get("last_plan_date", "noch nie")
    doc_count = await _run_sync(rag.get_doc_count)

    text = (
        "*Status*\n\n"
        f"📅 Google Calendar: {'verbunden ✅' if connected else 'nicht verbunden ❌'}\n"
        f"🗂 Letzter Wochenplan: {last_plan}\n"
        f"📚 Wissensbasis: {doc_count} Einträge"
    )
    await _send_markdown_safe(update.message, text)


async def merke(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Nutzung: /merke <Text>, z. B. /merke Ich mag morgens lieber Tee als Kaffee")
        return
    text = " ".join(context.args)
    await _run_sync(rag.add_to_memory, f"Notiz von Djamal: {text}")
    await update.message.reply_text(f"📝 Gemerkt: {text}")


async def checkin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manually trigger an evening check-in (handy for testing without waiting for 21:45)."""
    await scheduler.trigger_evening_checkin(context.bot, update.message.chat_id)


async def scheduler_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(scheduler.status_text())


# ── Plan approval / feedback flow ──

async def _send_plan_or_finish(update: Update, result: dict, chat_id: str) -> None:
    if "__interrupt__" in result:
        plan_text = result["__interrupt__"][0].value.get("plan_text", "")
        pending_plans.add(chat_id)
        await _send_markdown_safe(
            update.message,
            f"{plan_text}\n\n_Antworte mit 'ok', wenn der Plan passt, oder schreib mir, was ich ändern soll._",
        )
        return

    pending_plans.discard(chat_id)
    await _finalize_plan(update, result)


async def _finalize_plan(update: Update, result: dict) -> None:
    pdf_path = result.get("pdf_path")
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            await update.message.reply_document(f, filename=os.path.basename(pdf_path))
    else:
        await update.message.reply_text("⚠️ PDF konnte nicht erstellt werden.")

    summary = result.get("calendar_sync_summary", "")
    if summary:
        await update.message.reply_text(summary)

    state = _load_state()
    state["last_plan_date"] = datetime.now().strftime("%Y-%m-%d")
    _save_state(state)

    await update.message.reply_text("✅ Plan abgeschlossen!")


async def _handle_plan_feedback(update: Update, chat_id: str, text: str) -> None:
    config = {"configurable": {"thread_id": chat_id}}
    await update.message.reply_text("⏳ Einen Moment...")

    try:
        result = await _run_sync(agent_graph.invoke, Command(resume=text), config=config)
    except Exception as e:
        pending_plans.discard(chat_id)
        await update.message.reply_text(f"❌ Fehler: {e}")
        return

    await _send_plan_or_finish(update, result, chat_id)


# ── Google Calendar OAuth code handoff ──

async def _handle_oauth_code(update: Update, chat_id: str, text: str) -> None:
    try:
        await _run_sync(calendar_service.exchange_code, text)
    except calendar_service.CalendarError as e:
        await update.message.reply_text(f"❌ Verbindung fehlgeschlagen: {e}\n\nVersuche es erneut oder starte /connect neu.")
        return

    pending_oauth.discard(chat_id)
    await update.message.reply_text("✅ Google Calendar erfolgreich verbunden!")


# ── General chat fallback ──

MAX_TOOL_ROUNDS = 8


async def _handle_chat(update: Update, chat_id: str, text: str) -> None:
    history = chat_histories.setdefault(chat_id, [])
    connected = calendar_service.is_connected()

    context_text = await _run_sync(partial(rag.search, text, k=2))
    system_content = (
        "Du bist Djamals persönlicher Life OS Agent. Antworte auf Deutsch, freundlich und prägnant. "
        f"Heute ist {datetime.now().strftime('%A, %d.%m.%Y')}."
    )
    if context_text:
        system_content += f"\n\nRelevanter Kontext über Djamal:\n{context_text}"

    if connected:
        try:
            events = await _run_sync(calendar_service.get_upcoming_events, 7)
            calendar_text = calendar_service.format_events_summary(events)
        except calendar_service.CalendarError:
            calendar_text = "Kalender konnte nicht gelesen werden."
        system_content += (
            "\n\nTermine der nächsten 7 Tage:\n"
            f"{calendar_text}\n\n"
            "Verwalte Termine mit deinen Tools (list_events zuerst für Referenzen). "
            "Sicherheitsregel: Nur vom Agent erstellte Termine ändern/löschen; eigene Termine "
            "von Djamal sind tabu. Bei Konflikten keine Alternative erzwingen, sondern vorschlagen."
        )

    messages = [{"role": "system", "content": system_content}]
    messages += history[-6:]
    messages.append({"role": "user", "content": text})

    ref_map: dict = {}
    reply = ""
    try:
        if not connected:
            # No calendar tools available; plain chat.
            reply = await _run_sync(groq_chat, messages)
        else:
            for _ in range(MAX_TOOL_ROUNDS):
                msg = await _run_sync(agent_chat_with_tools, messages, calendar_tools.TOOLS)
                if not msg.tool_calls:
                    reply = msg.content or ""
                    break

                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in msg.tool_calls
                    ],
                })

                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    # The model sometimes emits arguments as the literal `null`
                    # (or a non-object), which json.loads turns into None/str.
                    if not isinstance(args, dict):
                        args = {}
                    result = await _run_sync(
                        calendar_tools.execute_tool, tc.function.name, args, ref_map
                    )
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            else:
                reply = "Das hat leider zu viele Schritte gebraucht. Bitte formuliere es etwas einfacher."
    except LLMError as e:
        await update.message.reply_text(f"❌ Groq ist aktuell nicht erreichbar: {e}")
        return

    if not reply:
        reply = "Erledigt."

    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": reply})
    chat_histories[chat_id] = history[-12:]

    await update.message.reply_text(reply)


# ── Message + voice handlers ──

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str | None = None) -> None:
    chat_id = str(update.message.chat_id)
    user_text = text if text is not None else update.message.text

    if chat_id in pending_oauth:
        await _handle_oauth_code(update, chat_id, user_text)
    elif chat_id in scheduler.pending_checkins:
        # Reply to an evening/weekly check-in → feed it into the feedback loop.
        await scheduler.handle_checkin_reply(context.bot, chat_id, user_text)
    elif chat_id in pending_plans:
        await _handle_plan_feedback(update, chat_id, user_text)
    else:
        await _handle_chat(update, chat_id, user_text)


async def voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🎤 Sprachnachricht empfangen, transkribiere...")

    ogg_path = f"data/voice_{update.message.message_id}.ogg"
    wav_path = f"data/voice_{update.message.message_id}.wav"

    try:
        voice_file = await update.message.voice.get_file()
        await voice_file.download_to_drive(ogg_path)

        proc = await _run_sync(subprocess.run, ["ffmpeg", "-y", "-i", ogg_path, wav_path], capture_output=True)
        if proc.returncode != 0:
            await update.message.reply_text("❌ Audio konnte nicht konvertiert werden.")
            return

        text = await _run_sync(transcribe_audio, wav_path)
    except LLMError as e:
        await update.message.reply_text(f"❌ Transkription fehlgeschlagen: {e}")
        return
    finally:
        for path in (ogg_path, wav_path):
            if os.path.exists(path):
                os.remove(path)

    await _send_markdown_safe(update.message, f"📝 Verstanden: _{text}_")
    await handle_text(update, context, text=text)


# ── App setup ──

log = logging.getLogger(__name__)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Catch-all so handler exceptions and transient network blips don't crash the bot."""
    err = context.error
    # Bad Gateway / timeouts on long-polling are transient; PTB retries automatically.
    if isinstance(err, (NetworkError, TimedOut)):
        log.warning("Transient Telegram network error: %s", err)
        return

    log.exception("Unhandled exception while processing update", exc_info=err)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "❌ Da ist etwas schiefgelaufen. Versuch es bitte nochmal."
            )
        except Exception:
            pass


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
    )
    app = (
        ApplicationBuilder()
        .token(os.environ["TELEGRAM_TOKEN"])
        # Start/stop the proactive scheduler inside run_polling()'s event loop so it
        # reinitializes cleanly on every (re)start and shuts down without lingering jobs.
        .post_init(scheduler.start)
        .post_shutdown(scheduler.shutdown)
        .build()
    )

    # Runs first for every update so reminders always have an up-to-date chat list.
    app.add_handler(TypeHandler(Update, track_chat), group=-1)

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("plan", plan))
    app.add_handler(CommandHandler("today", today))
    app.add_handler(CommandHandler("add", add_event))
    app.add_handler(CommandHandler("connect", connect))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("merke", merke))
    app.add_handler(CommandHandler("checkin", checkin))
    app.add_handler(CommandHandler("scheduler_status", scheduler_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, voice))

    app.add_error_handler(error_handler)

    reminders.register(app.job_queue)

    print("🤖 Bot läuft...")
    app.run_polling()


if __name__ == "__main__":
    main()
