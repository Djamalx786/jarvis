"""Proactive daily scheduler with a feedback loop — the core of the Life OS Agent.

Three recurring touchpoints run on an AsyncIOScheduler (Europe/Berlin), wired into the
python-telegram-bot lifecycle via post_init / post_shutdown hooks so they live inside
the same event loop as ``app.run_polling()``:

  - Morning nudge      weekdays 07:00  LLM-generated 3 priorities + 1 health reminder
  - Evening wind-down  daily   21:45   phone-away nudge + one rotating reflection question
  - Weekly reflection  Sunday  19:00   review the week, then hand off to /plan

Replies to the evening/weekly questions are captured (see ``handle_checkin_reply``),
distilled into a one-line insight by Groq, appended to data/notes/feedback_history.txt
and added to the live RAG index, so the next /plan run actually learns from them.

The scheduler uses an in-memory job store with stable job ids and ``replace_existing``:
on a restart (e.g. systemd ``Restart=always``) it re-registers cleanly without ever
duplicating jobs.
"""
import asyncio
import json
import logging
import os
import random
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src import agent, calendar_service, rag

log = logging.getLogger(__name__)

TZ = ZoneInfo("Europe/Berlin")
USER_CONFIG_PATH = "data/user_config.json"
STATE_FILE = "data/state.json"
FEEDBACK_PATH = "data/notes/feedback_history.txt"

GERMAN_WEEKDAYS = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]

# Rotated so the evening check-in never feels robotic.
EVENING_QUESTIONS = [
    "Wie ist der Tag gelaufen? Was hat gut funktioniert, was nicht?",
    "Hast du heute Zeit für dich gefunden? Was würdest du morgen anders machen?",
    "Plattfuß-Übungen heute geschafft? Was lief gut, was nicht?",
]

# Module-level scheduler handle plus a record of which chats are mid-check-in.
# ``pending_checkins`` maps chat_id (str) -> "evening" | "weekly"; telegram_bot consults
# it to route a free-text reply into the feedback loop instead of normal chat. It is
# in-memory by design: a restart simply drops a half-finished check-in, which is harmless.
_scheduler: AsyncIOScheduler | None = None
pending_checkins: dict[str, str] = {}


# ── User config / target chats ──

def _load_user_config() -> dict:
    try:
        with open(USER_CONFIG_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_chat_id(chat_id: int) -> None:
    """Persist the user's chat_id so scheduled jobs know where to send. Idempotent.

    Single user for now, but the file is a dict so more fields/users can be added later.
    """
    config = _load_user_config()
    if config.get("chat_id") != chat_id:
        config["chat_id"] = chat_id
        os.makedirs(os.path.dirname(USER_CONFIG_PATH), exist_ok=True)
        with open(USER_CONFIG_PATH, "w") as f:
            json.dump(config, f)
        log.info("Saved chat_id %s to %s", chat_id, USER_CONFIG_PATH)


def _target_chat_ids() -> list[int]:
    """Chat ids to notify. Primary source is user_config.json (set on /start); falls back
    to state.json ``chat_ids`` (auto-recorded on every message) so jobs still work even
    before the first /start, e.g. when testing with /checkin."""
    ids: list[int] = []
    primary = _load_user_config().get("chat_id")
    if primary is not None:
        ids.append(int(primary))
    try:
        with open(STATE_FILE) as f:
            for c in json.load(f).get("chat_ids", []):
                if int(c) not in ids:
                    ids.append(int(c))
    except (OSError, json.JSONDecodeError):
        pass
    return ids


# ── Sending ──

async def _send(bot, chat_id: int, text: str) -> bool:
    """Send one message, swallowing network/API errors so a job never crashes the bot."""
    try:
        await bot.send_message(chat_id=chat_id, text=text)
        return True
    except Exception as e:
        log.warning("Scheduled send to %s failed: %s", chat_id, e)
        return False


# ── Morning message (weekdays 07:00) ──

def _today_calendar_text() -> str:
    """Today's events as a short summary, or '' if the calendar isn't connected/readable."""
    if not calendar_service.is_connected():
        return ""
    try:
        events = calendar_service.get_day_events(datetime.now(TZ))
        return calendar_service.format_events_summary(events)
    except calendar_service.CalendarError:
        return ""


def _build_morning_message() -> str:
    """Build the morning message via Groq. Blocking (RAG + LLM) — call in a thread."""
    now = datetime.now(TZ)
    weekday = GERMAN_WEEKDAYS[now.weekday()]
    context_text = rag.search(f"{weekday} Termine Prioritäten Gewohnheiten", k=3)
    calendar_text = _today_calendar_text()

    system = (
        "Du bist Djamals enger Freund und persönlicher Coach, der ihm morgens eine kurze "
        "Nachricht schreibt. Locker und motivierend, per Du, wie ein guter Kumpel – nicht "
        "corporate, nicht belehrend. Antworte auf Deutsch."
    )
    user = (
        f"Heute ist {weekday}, {now.strftime('%d.%m.%Y')}.\n\n"
        f"Kontext über Djamal (aus seinen Notizen):\n{context_text or '(kein Kontext)'}\n\n"
        f"Heutige Kalendertermine:\n{calendar_text or '(kein Kalender verbunden)'}\n\n"
        "Gib mir GENAU 3 Prioritäten für heute (je max. 2 Sätze), nummeriert mit 1️⃣ 2️⃣ 3️⃣, "
        "plus EINE kleine Gesundheits-Erinnerung (z. B. Plattfuß-Übungen) mit 🦶. "
        "Extrem knapp und scanbar – wird unterwegs in der Bahn in 10 Sekunden gelesen. "
        "Keine Einleitung, kein 'Guten Morgen' (kommt schon davor), direkt mit 'Heute:' starten."
    )
    try:
        body = agent.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.7,
        )
        body = (body or "").strip()
    except agent.LLMError as e:
        log.warning("Morning LLM call failed, sending fallback: %s", e)
        body = (
            "Heute:\n1️⃣ Die wichtigste Sache zuerst angehen\n"
            "2️⃣ Eine Sache für deine Ziele tun\n3️⃣ Abends bewusst abschalten\n\n"
            "🦶 Nicht vergessen: kurz Plattfuß-Übungen, auch wenn's nur 10 min sind."
        )
    return f"🌤️ Guten Morgen Djamal!\n\n{body}"


async def morning_job(application) -> None:
    text = await asyncio.to_thread(_build_morning_message)
    for chat_id in _target_chat_ids():
        await _send(application.bot, chat_id, text)


# ── Evening wind-down (daily 21:45) ──

def _evening_text() -> str:
    question = random.choice(EVENING_QUESTIONS)
    return (
        "🌙 Gleich ist Handy-weg-Zeit für heute. Bevor du abschaltest, kurz für mich:\n\n"
        f"{question}"
    )


async def evening_job(application) -> None:
    text = _evening_text()
    for chat_id in _target_chat_ids():
        if await _send(application.bot, chat_id, text):
            pending_checkins[str(chat_id)] = "evening"


async def trigger_evening_checkin(bot, chat_id: int) -> None:
    """Manually fire an evening check-in to a single chat (used by /checkin for testing)."""
    text = _evening_text()
    if await _send(bot, chat_id, text):
        pending_checkins[str(chat_id)] = "evening"


# ── Weekly reflection (Sunday 19:00) ──

def _recent_feedback(days: int = 7) -> str:
    """Return date-stamped feedback_history.txt entries from the last ``days`` days."""
    cutoff = date.today() - timedelta(days=days)
    lines: list[str] = []
    try:
        with open(FEEDBACK_PATH) as f:
            for line in f:
                m = re.match(r"\[(\d{4}-\d{2}-\d{2})\]", line.strip())
                if m and date.fromisoformat(m.group(1)) >= cutoff:
                    lines.append(line.strip())
    except OSError:
        pass
    return "\n".join(lines)


def _weekly_text() -> str:
    recent = _recent_feedback(7)
    recap = f"\n\nDas hast du diese Woche notiert:\n{recent}" if recent else ""
    return (
        "🗓 Wochenrückblick: Was hat diese Woche richtig gut geklappt, was würdest du "
        f"nächste Woche ändern?{recap}"
    )


async def weekly_job(application) -> None:
    text = await asyncio.to_thread(_weekly_text)
    for chat_id in _target_chat_ids():
        if await _send(application.bot, chat_id, text):
            pending_checkins[str(chat_id)] = "weekly"


# ── Feedback loop: capture reply → insight → log + RAG ──

def _extract_insight(reply: str) -> str:
    """Distil the user's free-text reply into a 1-2 sentence insight for the feedback log.

    Falls back to the raw reply if Groq is unavailable — feedback must NEVER be lost.
    """
    system = (
        "Du destillierst Djamals Tagesreflexion in eine sehr kurze, sachliche Notiz "
        "(1-2 Sätze, Deutsch) für sein persönliches Feedback-Log, aus dem später "
        "Wochenpläne lernen. Nenne konkret, was gut oder schlecht lief, und – falls "
        "sinnvoll – eine Konsequenz für die Planung. Keine Anrede, keine Emojis, nur die Notiz."
    )
    try:
        insight = agent.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": reply}],
            temperature=0.3,
        )
        insight = (insight or "").strip()
        return insight or reply.strip()
    except agent.LLMError as e:
        log.warning("Feedback extraction failed, storing raw reply: %s", e)
        return reply.strip()


def _append_feedback(insight: str, weekly: bool) -> None:
    """Append a dated insight to feedback_history.txt and add it to the live RAG index."""
    stamp = date.today().isoformat()
    tag = " [WOCHENRÜCKBLICK]" if weekly else ""
    entry = f"[{stamp}]{tag} {insight}"
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    with open(FEEDBACK_PATH, "a") as f:
        f.write(f"\n{entry}\n")
    # Make it searchable immediately, without rebuilding the whole index.
    rag.add_feedback_entry(entry)


async def handle_checkin_reply(bot, chat_id: str, reply: str) -> None:
    """Process a free-text reply to an evening/weekly check-in: extract an insight,
    append it to the feedback log + RAG, and confirm. Always clears the pending state."""
    mode = pending_checkins.pop(chat_id, "evening")
    insight = await asyncio.to_thread(_extract_insight, reply)
    await asyncio.to_thread(_append_feedback, insight, mode == "weekly")

    if mode == "weekly":
        await _send(
            bot, int(chat_id),
            "Danke! Schreib /plan wenn du bereit für den Plan für nächste Woche bist.",
        )
    else:
        await _send(bot, int(chat_id), "Danke, hab's mir notiert 📝 Gute Nacht!")


# ── Lifecycle (post_init / post_shutdown hooks) ──

def _build_scheduler(application) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=TZ)
    # Stable ids + replace_existing → a restart re-registers cleanly, never duplicating.
    scheduler.add_job(morning_job, "cron", day_of_week="mon-fri", hour=7, minute=0,
                      args=[application], id="morning", replace_existing=True)
    scheduler.add_job(evening_job, "cron", hour=21, minute=45,
                      args=[application], id="evening", replace_existing=True)
    scheduler.add_job(weekly_job, "cron", day_of_week="sun", hour=19, minute=0,
                      args=[application], id="weekly", replace_existing=True)
    return scheduler


async def start(application) -> None:
    """PTB post_init hook: build and start the scheduler inside the running event loop."""
    global _scheduler
    if _scheduler and _scheduler.running:
        return
    _scheduler = _build_scheduler(application)
    _scheduler.start()
    log.info("Proactive scheduler started.\n%s", status_text())


async def shutdown(application) -> None:
    """PTB post_shutdown hook: stop the scheduler cleanly so jobs don't linger."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        log.info("Proactive scheduler stopped.")
    _scheduler = None


# ── Status (/scheduler_status) ──

def _next_run(job_id: str) -> str:
    if not _scheduler:
        return "—"
    job = _scheduler.get_job(job_id)
    if not job or not job.next_run_time:
        return "—"
    return job.next_run_time.astimezone(TZ).strftime("%a %d.%m. %H:%M")


def status_text() -> str:
    running = bool(_scheduler and _scheduler.running)
    return (
        f"Scheduler läuft: {'ja ✅' if running else 'nein ❌'}\n"
        f"🌤️ Nächste Morgen-Nachricht: {_next_run('morning')}\n"
        f"🌙 Nächste Abend-Nachricht: {_next_run('evening')}\n"
        f"🗓 Nächster Wochenrückblick: {_next_run('weekly')}"
    )
