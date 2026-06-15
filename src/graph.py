"""LangGraph agent for Djamal's weekly planning workflow.

Flow: load_context -> plan_week -> human_approval -> (export_pdf -> sync_calendar) | plan_week
The graph pauses at human_approval via interrupt() and resumes via Command(resume=...).
"""
import json
import os
import re
import sqlite3
from datetime import datetime, date, timedelta
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt

from src import calendar_service, rag
from src.agent import chat, LLMError
from src.config import BMW_START_DATE, STRENGTH_TRAINING_ALLOWED_FROM
from src.pdf_export import CATEGORY_COLORS, DAYS, generate_pdf
from src.rag import search

MAX_ITERATIONS = 3
APPROVAL_WORDS = {"ok", "okay", "ja", "yes", "gut", "passt", "genau", "perfekt", "super", "approved", "go"}

CATEGORIES = list(CATEGORY_COLORS.keys())

CATEGORY_EMOJI = {
    "Arbeit (BMW)": "🟦",
    "KI & Projekte": "🟪",
    "Islam & Reflexion": "🟩",
    "Familie & Ehe": "🟥",
    "Sport & Freizeit": "🟨",
}


class AgentState(TypedDict):
    user_input: str
    rag_context: str
    calendar_context: str
    plan_data: dict
    plan_text: str
    pdf_path: str
    feedback: str
    iteration: int
    calendar_sync_summary: str
    week_start: str
    error: str


def _next_monday() -> date:
    today = datetime.now().date()
    return today + timedelta(days=(7 - today.weekday()))


# ── NODE 1: Load context from RAG + Google Calendar ──
def load_context(state: AgentState) -> AgentState:
    print("Loading context from RAG and Google Calendar...")
    query = state.get("user_input") or "Wochenplan, Prioritäten, BMW, Sport, Familie"

    rag_context = search(query)
    if not rag_context:
        rag_context = "Keine zusätzlichen Informationen in der Wissensbasis gefunden."

    week_start = _next_monday()

    calendar_context = "Google Calendar ist nicht verbunden."
    if calendar_service.is_connected():
        try:
            events = calendar_service.get_week_events(week_start)
            calendar_context = calendar_service.format_events_summary(events)
        except calendar_service.CalendarError as e:
            calendar_context = f"Kalender konnte nicht gelesen werden: {e}"

    return {
        "rag_context": rag_context,
        "calendar_context": calendar_context,
        "week_start": week_start.isoformat(),
        "error": "",
    }


# ── NODE 2: Plan the week ──
def _reserved_summary(week_start: date) -> str:
    """Per-day, human-readable list of the blocks that are auto-added, so the LLM plans around them."""
    lines = []
    for i, day in enumerate(DAYS):
        day_date = week_start + timedelta(days=i)
        fixed = _fixed_blocks_for_date(day_date)
        desc = "; ".join(f"{b['time']} {b['title']}" for b in fixed)
        lines.append(f"- {day} ({day_date.strftime('%d.%m.')}): {desc}")
    return "\n".join(lines)


def _work_context(week_start: date) -> str:
    """Describe whether the planned week falls before or after the BMW start date."""
    week_end = week_start + timedelta(days=6)
    if week_end < BMW_START_DATE:
        return (
            f"Djamal startet erst am {BMW_START_DATE.strftime('%d.%m.%Y')} bei BMW. Diese ganze Woche "
            "liegt DAVOR: an den Werktagen gibt es KEINE Arbeit und KEINEN Arbeitsweg. Behandle Mo-Fr "
            "tagsüber als freie, produktive Tage (KI Skills, Projekte, Bewegung, Erholung)."
        )
    if week_start >= BMW_START_DATE:
        return (
            f"Djamal arbeitet diese Woche Vollzeit bei BMW (Start war am {BMW_START_DATE.strftime('%d.%m.%Y')}). "
            "Arbeitsweg + BMW Arbeit (Mo-Fr) werden automatisch ergänzt – plane nur um sie herum."
        )
    return (
        f"Djamal startet am {BMW_START_DATE.strftime('%d.%m.%Y')} bei BMW – mitten in dieser Woche. "
        "Vor diesem Datum sind die Werktage tagsüber frei, ab diesem Datum gilt die feste Arbeitszeit "
        "(beides wird automatisch korrekt ergänzt – siehe Liste der reservierten Blöcke)."
    )


def _sport_context(week_start: date) -> str:
    """Describe whether strength training is allowed for the planned week."""
    week_end = week_start + timedelta(days=6)
    if week_end < STRENGTH_TRAINING_ALLOWED_FROM:
        return (
            "KEIN Kraftsport diese Woche (ärztliche Pause). Erlaubt sind ausschließlich: entspanntes "
            "Fahrrad, Spaziergänge, Dehnen und Plattfuß-Übungen."
        )
    if week_start >= STRENGTH_TRAINING_ALLOWED_FROM:
        return (
            f"Kraftsport ist seit {STRENGTH_TRAINING_ALLOWED_FROM.strftime('%d.%m.%Y')} wieder erlaubt und "
            "darf wieder behutsam eingeplant werden – weiterhin auch leichte Bewegung (Spaziergang, Fahrrad, Dehnen)."
        )
    return (
        f"Kraftsport ist erst ab {STRENGTH_TRAINING_ALLOWED_FROM.strftime('%d.%m.%Y')} wieder erlaubt – vor "
        "diesem Datum NUR leichte Bewegung (Fahrrad, Spaziergang, Dehnen, Plattfuß-Übungen)."
    )


def _build_plan_prompt(state: AgentState) -> list:
    feedback = state.get("feedback", "")
    iteration = state.get("iteration", 0)
    week_start = date.fromisoformat(state["week_start"])

    feedback_prompt = ""
    if feedback and iteration > 0:
        feedback_prompt = f"\n\nFeedback des Nutzers zum letzten Plan: \"{feedback}\"\nBitte passe den Plan entsprechend an."

    category_list = "\n".join(f"- {c}" for c in CATEGORIES)
    day_keys = ", ".join(DAYS)

    system_prompt = (
        "Du bist Djamals persönlicher Life OS Agent. Du erstellst durchdachte, realistische "
        "Wochenpläne und antwortest ausschließlich mit einem validen JSON-Objekt, ohne "
        "Markdown-Codeblöcke und ohne Erklärtext.\n\n"
        f"Das JSON hat als Schlüssel genau diese sieben Tage: {day_keys}.\n"
        "Jeder Tag ist eine Liste von Aufgaben-Objekten mit den Feldern:\n"
        '  - "time": Zeitspanne im Format "HH:MM - HH:MM" (24h, keine Überlappungen innerhalb eines Tages)\n'
        '  - "title": konkreter, handlungsorientierter Titel auf Deutsch (max. ca. 42 Zeichen)\n'
        '  - "category": EXAKT einer der folgenden Werte:\n'
        f"{category_list}\n\n"
        "Regeln für einen guten Plan:\n"
        "1. RESERVIERTE BLÖCKE – NICHT SELBST PLANEN, NICHT ÜBERLAPPEN: Arbeitsweg, BMW Arbeit, "
        "Freitagsgebet, gemeinsames Abendessen (18:30-20:30) und der Sonntags-Familienbesuch werden "
        "AUTOMATISCH ergänzt. Du bekommst unten pro Tag genau die bereits reservierten Zeitfenster. "
        "Plane NICHTS in diese Fenster und füge sie NICHT selbst hinzu.\n"
        "2. KONKRET statt generisch: Schreibe spezifische Titel wie 'KI: LangGraph-Tutorial' oder "
        "'Lesen: Reinforcement Learning', nicht bloß 'KI & Projekte' oder 'Lernen'. Nenne BMW-Vorbereitung "
        "NICHT 'BMW-Vorbereitung' (erzeugt Druck), sondern z.B. 'KI Skills' oder 'Lernen'.\n"
        "3. ABWECHSLUNG: Wiederhole nicht stumpf jeden Tag dieselben Blöcke. Variiere Themen und "
        "Bewegungsarten über die Woche.\n"
        "4. RHYTHMUS & MASS: Max. 2 Fokusthemen pro Tag, lieber weniger und dafür richtig. Abends "
        "höchstens 1 produktiver Block (ca. 1h, frühestens nach 20:30) und NICHT jeden Abend – "
        "mindestens 2 Abende pro Woche komplett frei. Lasse bewusst Freiräume.\n"
        "5. FREITAG ABEND ist IMMER frei: nach Feierabend bzw. nach 17:30 KEIN produktiver Block, "
        "der Abend soll gemütlich ausklingen.\n"
        "6. BEWEGUNG: " + _sport_context(week_start) + " Plattfuß-Übungen 3-4x pro Woche, je 15-20 min, "
        "am besten 17:30-18:30 vor dem Abendessen. Sport/Bewegung bevorzugt nachmittags oder morgens.\n"
        "7. KI & PROJEKTE hat hohe Priorität und gehört in die fokussierten Tagesstunden bzw. (an "
        "Arbeitstagen) in die freien Abende.\n"
        "8. KORAN LESEN: GENAU AN EINEM Tag am Wochenende (Samstag ODER Sonntag, nicht an beiden!) "
        "einen frühen Morgen-Block einplanen, ca. 05:00 - 05:45 Uhr, "
        '{"title": "Koran lesen", "category": "Islam & Reflexion"}. '
        "Am jeweils anderen Wochenendtag KEINEN Islam & Reflexion-Block einplanen.\n"
        "9. HANDY-DETOX: Plane nach 22:45 NICHTS mehr – der letzte Block endet spätestens 22:45 "
        "(danach Handy weg, schlafen).\n"
        "10. Plane NIE über bestehende Kalendertermine – arbeite um sie herum.\n"
        "11. HAUSHALT/AUFRÄUMEN nur am Wochenende oder Freitagnachmittag, nicht unter der Woche. "
        "Einkaufen NICHT fest einplanen (läuft spontan)."
    )

    user_prompt = (
        f"Informationen über mich (aus meiner Wissensbasis):\n{state.get('rag_context', '')}\n\n"
        f"Arbeitsstatus dieser Woche: {_work_context(week_start)}\n\n"
        f"Bereits reservierte (automatisch ergänzte) Blöcke – plane NICHT hinein:\n{_reserved_summary(week_start)}\n\n"
        f"Bestehende Kalendertermine für die kommende Woche:\n{state.get('calendar_context', '')}\n\n"
        f"Erstelle einen Wochenplan für die kommende Woche.{feedback_prompt}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _parse_plan_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    data = json.loads(raw)

    normalized = {}
    for day in DAYS:
        tasks = data.get(day, [])
        if not isinstance(tasks, list):
            tasks = []
        cleaned = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            cleaned.append({
                "time": str(task.get("time", "")).strip(),
                "title": str(task.get("title", "")).strip(),
                "category": str(task.get("category", "")).strip(),
            })
        normalized[day] = cleaned

    return normalized


def _time_range_minutes(time_str: str):
    """Parse 'HH:MM - HH:MM' into (start_minutes, end_minutes). Returns None if invalid."""
    if not time_str:
        return None
    cleaned = time_str.replace("–", "-").replace("—", "-")
    parts = [p.strip() for p in cleaned.split("-")]
    if len(parts) != 2:
        return None
    try:
        sh, sm = (int(x) for x in parts[0].split(":"))
        eh, em = (int(x) for x in parts[1].split(":"))
    except (ValueError, IndexError):
        return None
    start = sh * 60 + sm
    end = eh * 60 + em
    if end <= start:
        end = start + 60
    return start, end


# ── Fixed-block templates (deterministically injected, never left to the LLM) ──
# "Heilige" Blöcke aus Djamals Wissensbasis: Arbeitsweg + BMW Arbeit (nur ab BMW-Start),
# Freitagsgebet, gemeinsames Abendessen, Sonntags-Familienbesuch.
COMMUTE_IN   = {"time": "06:30 - 07:30", "title": "Arbeitsweg (Hinweg)",   "category": "Arbeit (BMW)"}
COMMUTE_OUT  = {"time": "16:30 - 17:30", "title": "Arbeitsweg (Rückweg)",  "category": "Arbeit (BMW)"}
WORK_FULL    = {"time": "07:30 - 16:30", "title": "BMW Arbeit",            "category": "Arbeit (BMW)"}
WORK_AM      = {"time": "07:30 - 13:30", "title": "BMW Arbeit",            "category": "Arbeit (BMW)"}
WORK_PM      = {"time": "14:30 - 16:30", "title": "BMW Arbeit",            "category": "Arbeit (BMW)"}
JUMMAH_BLOCK = {"time": "13:30 - 14:30", "title": "Freitagsgebet (Jummah)", "category": "Islam & Reflexion"}
DINNER_BLOCK = {"time": "18:30 - 20:30", "title": "Kochen & Abendessen mit Frau", "category": "Familie & Ehe"}
FAMILY_BLOCK = {"time": "14:00 - 20:30", "title": "Familie / Eltern besuchen",    "category": "Familie & Ehe"}


def _overlaps(a: tuple, b: tuple) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _fixed_blocks_for_date(day_date: date) -> list:
    """Return the fixed, non-negotiable blocks for a concrete calendar date.

    Work/commute blocks only appear on workdays on/after the BMW start date, so
    weeks before Djamal starts at BMW are correctly left free during the day.
    On Fridays the work day is split around the Jummah prayer; on Fridays before
    the BMW start a standalone Jummah block is still reserved.
    """
    wd = day_date.weekday()  # 0 = Montag ... 6 = Sonntag
    is_workday = wd < 5
    is_friday = wd == 4
    is_sunday = wd == 6
    bmw_active = day_date >= BMW_START_DATE

    blocks = []
    if is_workday and bmw_active:
        blocks.append(dict(COMMUTE_IN))
        if is_friday:
            blocks.append(dict(WORK_AM))
            blocks.append(dict(JUMMAH_BLOCK))
            blocks.append(dict(WORK_PM))
        else:
            blocks.append(dict(WORK_FULL))
        blocks.append(dict(COMMUTE_OUT))
    elif is_friday:
        blocks.append(dict(JUMMAH_BLOCK))

    blocks.append(dict(FAMILY_BLOCK) if is_sunday else dict(DINNER_BLOCK))
    return blocks


def _enforce_fixed_blocks(plan_data: dict, week_start: date) -> dict:
    """Deterministically guarantee the fixed blocks for each day of the week.

    For every day we compute the fixed blocks for that concrete date, drop any
    LLM-planned task that overlaps one of them, then inject the fixed blocks and
    sort by start time. Idempotent: re-applying drops the previously injected
    blocks (they overlap exactly) and re-adds them, so no duplicates appear.
    """
    for i, day in enumerate(DAYS):
        day_date = week_start + timedelta(days=i)
        fixed = _fixed_blocks_for_date(day_date)
        reserved = [_time_range_minutes(b["time"]) for b in fixed]

        kept = []
        for task in plan_data.get(day, []):
            span = _time_range_minutes(task.get("time", ""))
            if span is None:
                continue
            if any(_overlaps(span, window) for window in reserved):
                continue
            kept.append(task)

        kept.extend(fixed)
        kept.sort(key=lambda t: _time_range_minutes(t["time"])[0])
        plan_data[day] = kept

    return plan_data


def format_plan_text(plan_data: dict, week_start: date) -> str:
    """Render the structured plan as a readable German text for Telegram."""
    dates = {DAYS[i]: week_start + timedelta(days=i) for i in range(7)}
    week_range = f"{dates['Montag'].strftime('%d.%m.')} - {dates['Sonntag'].strftime('%d.%m.%Y')}"

    lines = [f"*Wochenplan ({week_range})*", ""]
    for day in DAYS:
        tasks = plan_data.get(day, [])
        lines.append(f"*{day}, {dates[day].strftime('%d.%m.')}*")
        if not tasks:
            lines.append("  Frei")
        else:
            for task in tasks:
                emoji = CATEGORY_EMOJI.get(task.get("category", ""), "⬜")
                time_range = task.get("time", "")
                title = task.get("title", "")
                lines.append(f"  {emoji} {time_range} {title}")
        lines.append("")

    return "\n".join(lines).strip()


def plan_week(state: AgentState) -> AgentState:
    print("Planning week with Groq...")
    iteration = state.get("iteration", 0)
    week_start = date.fromisoformat(state["week_start"])

    try:
        raw = chat(_build_plan_prompt(state), temperature=0.6, json_mode=True)
        plan_data = _enforce_fixed_blocks(_parse_plan_json(raw), week_start)
        plan_text = format_plan_text(plan_data, week_start)
        error = ""
    except LLMError as e:
        plan_data = _enforce_fixed_blocks(state.get("plan_data", {day: [] for day in DAYS}), week_start)
        plan_text = (
            f"⚠️ Der Plan konnte nicht erstellt werden (Groq-Fehler): {e}\n\n"
            "Antworte mit 'ok', um es trotzdem mit dem letzten Plan zu versuchen, "
            "oder versuche es später erneut."
        )
        error = str(e)
    except (json.JSONDecodeError, ValueError) as e:
        plan_data = _enforce_fixed_blocks(state.get("plan_data", {day: [] for day in DAYS}), week_start)
        plan_text = (
            f"⚠️ Der Plan konnte nicht verarbeitet werden (ungültiges Format): {e}\n\n"
            "Antworte mit 'ok', um es trotzdem mit dem letzten Plan zu versuchen, "
            "oder gib Feedback für einen neuen Versuch."
        )
        error = str(e)

    return {
        "plan_data": plan_data,
        "plan_text": plan_text,
        "iteration": iteration + 1,
        "feedback": "",
        "error": error,
    }


# ── NODE 3: Human approval (pauses the graph) ──
def human_approval(state: AgentState) -> AgentState:
    print("Waiting for human approval...")
    feedback = interrupt({
        "plan_text": state["plan_text"],
        "iteration": state.get("iteration", 0),
        "message": "Bitte prüfe den Plan. Antworte mit 'ok' zum Bestätigen oder gib Feedback.",
    })
    return {"feedback": feedback}


def route_after_approval(state: AgentState) -> str:
    feedback = (state.get("feedback") or "").strip().lower()

    if feedback in APPROVAL_WORDS:
        return "export_pdf"
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "export_pdf"
    return "plan_week"


# ── NODE 4: Export PDF ──
def export_pdf(state: AgentState) -> AgentState:
    print("Generating PDF...")
    week_start = date.fromisoformat(state["week_start"])
    try:
        pdf_path = generate_pdf(state["plan_data"], week_start=week_start)
    except Exception as e:
        print(f"PDF export failed: {e}")
        return {"pdf_path": "", "error": f"PDF-Export fehlgeschlagen: {e}"}

    return {"pdf_path": pdf_path}


# ── NODE 5: Sync to Google Calendar ──
def sync_calendar(state: AgentState) -> AgentState:
    print("Syncing with Google Calendar...")
    week_start = date.fromisoformat(state["week_start"])

    if not calendar_service.is_connected():
        return {"calendar_sync_summary": "Google Calendar ist nicht verbunden. Nutze /connect, um Termine zu synchronisieren."}

    try:
        calendar_service.delete_week_events(week_start)
        results = calendar_service.create_week_plan(state["plan_data"], week_start)
    except calendar_service.CalendarError as e:
        return {"calendar_sync_summary": f"Kalender-Synchronisation fehlgeschlagen: {e}"}

    created = [r for r in results if "event_id" in r]
    failed = [r for r in results if "error" in r]

    summary_lines = [f"{len(created)} Termine im Google Calendar angelegt."]
    if failed:
        summary_lines.append(f"{len(failed)} Termine konnten nicht angelegt werden.")

    return {"calendar_sync_summary": "\n".join(summary_lines)}


# ── NODE 6: Save the confirmed plan to the agent's memory ──
def save_to_memory(state: AgentState) -> AgentState:
    print("Saving confirmed plan to memory...")
    week_start = date.fromisoformat(state["week_start"])
    week_end = week_start + timedelta(days=6)
    lines = [f"Wochenplan {week_start.strftime('%d.%m.')} - {week_end.strftime('%d.%m.%Y')} (bestätigt):"]
    for day in DAYS:
        tasks = state["plan_data"].get(day, [])
        titles = ", ".join(t["title"] for t in tasks if t.get("category") != "Arbeit (BMW)")
        lines.append(f"- {day}: {titles or 'frei'}")

    try:
        rag.add_to_memory("\n".join(lines))
    except Exception as e:
        print(f"Memory save failed: {e}")

    return {}


# ── BUILD GRAPH ──
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("load_context", load_context)
    graph.add_node("plan_week", plan_week)
    graph.add_node("human_approval", human_approval)
    graph.add_node("export_pdf", export_pdf)
    graph.add_node("sync_calendar", sync_calendar)
    graph.add_node("save_to_memory", save_to_memory)

    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "plan_week")
    graph.add_edge("plan_week", "human_approval")

    graph.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "export_pdf": "export_pdf",
            "plan_week": "plan_week",
        },
    )

    graph.add_edge("export_pdf", "sync_calendar")
    graph.add_edge("sync_calendar", "save_to_memory")
    graph.add_edge("save_to_memory", END)

    # Persistent checkpointer: an in-progress plan (awaiting approval/feedback) survives a
    # bot restart instead of being lost like the old in-memory MemorySaver. check_same_thread
    # is off because the graph is invoked from worker threads via run_in_executor.
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/checkpoints.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    return graph.compile(checkpointer=checkpointer)
