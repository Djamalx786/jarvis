"""Proactive Telegram reminders — the part that makes the agent feel like a real
'Life OS' instead of a passive command bot.

Registered on the bot's JobQueue in ``telegram_bot.main()``. Every job broadcasts to
all chat_ids that have interacted with the bot (persisted in ``data/state.json`` under
``chat_ids``). All times are Europe/Berlin.

Jobs (the interactive morning/evening/weekly touchpoints live in src/scheduler.py):
- Weekend briefing  : Sat/Sun 08:30 — today's calendar + a gentle nudge.
- Movement reminder : Mon/Wed/Thu 17:45 — Plattfuß-Übungen & Dehnen before dinner.
- Friday wind-down  : Fri 17:30 — protect the free Friday evening.
"""
import asyncio
import json
from datetime import datetime, time
from functools import partial
from zoneinfo import ZoneInfo

from telegram.ext import ContextTypes

from src import calendar_service

STATE_FILE = "data/state.json"
TZ = ZoneInfo("Europe/Berlin")

# PTB JobQueue day convention: 0-6 == Sunday-Saturday.
SUN, MON, TUE, WED, THU, FRI, SAT = range(7)
WEEKDAYS = (MON, TUE, WED, THU, FRI)
WEEKEND = (SAT, SUN)

_GERMAN_WEEKDAYS = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]


def _chat_ids() -> list[int]:
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    return [int(c) for c in data.get("chat_ids", [])]


async def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))


async def _broadcast(context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    for chat_id in _chat_ids():
        try:
            await context.bot.send_message(chat_id=chat_id, text=text)
        except Exception:
            # A single unreachable chat must never take down the whole job.
            pass


def _german_date(dt: datetime) -> str:
    return f"{_GERMAN_WEEKDAYS[dt.weekday()]}, {dt.strftime('%d.%m.%Y')}"


# ── Jobs ──

async def morning_briefing(context: ContextTypes.DEFAULT_TYPE) -> None:
    now = datetime.now(TZ)
    header = f"☀️ Guten Morgen, Djamal! Heute ist {_german_date(now)}."

    if calendar_service.is_connected():
        try:
            events = await _run_sync(calendar_service.get_day_events, now)
            agenda = calendar_service.format_events_summary(events)
        except calendar_service.CalendarError:
            agenda = "Kalender konnte gerade nicht gelesen werden."
    else:
        agenda = "Google Calendar ist nicht verbunden (/connect)."

    await _broadcast(context, f"{header}\n\n📅 Dein Tag:\n{agenda}")


async def movement_reminder(context: ContextTypes.DEFAULT_TYPE) -> None:
    await _broadcast(
        context,
        "🦶 Kurze Bewegung: Plattfuß-Übungen & Dehnen (15–20 min) vor dem Abendessen. "
        "Regelmäßig dranbleiben zählt mehr als lang.",
    )


async def friday_winddown(context: ContextTypes.DEFAULT_TYPE) -> None:
    await _broadcast(
        context,
        "🎉 Wochenende! Der Freitagabend gehört dir – bewusst nichts vornehmen, "
        "einfach gemütlich ausklingen lassen.",
    )


def register(job_queue) -> None:
    """Register the simple one-way nudges on the given JobQueue.

    The interactive proactive touchpoints (weekday morning briefing, evening wind-down,
    Sunday weekly reflection) now live in ``src/scheduler.py`` with their feedback loop;
    only the non-overlapping nudges remain here so the user is never double-messaged.
    """
    job_queue.run_daily(morning_briefing, time=time(8, 30, tzinfo=TZ), days=WEEKEND, name="briefing_weekend")
    job_queue.run_daily(movement_reminder, time=time(17, 45, tzinfo=TZ), days=(MON, WED, THU), name="movement")
    job_queue.run_daily(friday_winddown, time=time(17, 30, tzinfo=TZ), days=(FRI,), name="friday_winddown")
