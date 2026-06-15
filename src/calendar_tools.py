"""Tool definitions + dispatcher that let the chat agent read, create, move and
delete Google Calendar events safely.

Safety model:
- The agent may only *modify or delete* events it created itself (tagged with
  ``calendar_service.SOURCE_TAG``). ``update_event`` / ``delete_event`` enforce this
  server-side; the tool layer surfaces a clear message if a user-owned event is hit.
- Before creating or moving an event the dispatcher checks for overlaps with any
  existing event and refuses if there is a conflict.
- Events are addressed via short references ("evt1", "evt2", ...) handed out by
  ``list_events`` and resolved through ``ref_map`` — the LLM never juggles raw IDs.
"""
from datetime import date, datetime

from src import calendar_service

# OpenAI/Groq-style tool schemas exposed to the chat model.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_events",
            "description": (
                "Listet Djamals Kalendertermine der nächsten Tage auf. Jeder Termin bekommt eine "
                "kurze Referenz (z. B. 'evt1'), die du für move_event und delete_event verwendest. "
                "Außerdem wird markiert, welche Termine vom Agent erstellt wurden – nur diese darfst "
                "du ändern oder löschen. Rufe dieses Tool IMMER zuerst auf, bevor du etwas verschiebst "
                "oder löschst, um die richtige Referenz zu erhalten."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days_ahead": {
                        "type": "integer",
                        "description": "Wie viele Tage ab heute durchsucht werden (Standard 14).",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_event",
            "description": (
                "Erstellt einen neuen Termin. Prüft vorher automatisch auf Überlappungen mit "
                "bestehenden Terminen und legt NICHTS an, wenn es einen Konflikt gibt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Kurzer Titel des Termins."},
                    "date": {"type": "string", "description": "Datum im Format YYYY-MM-DD."},
                    "start_time": {"type": "string", "description": "Startzeit HH:MM (24h)."},
                    "end_time": {"type": "string", "description": "Endzeit HH:MM (24h)."},
                    "description": {"type": "string", "description": "Optionale Zusatzinfo."},
                },
                "required": ["title", "date", "start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_event",
            "description": (
                "Verschiebt bzw. ändert die Zeit eines vom Agent erstellten Termins (per Referenz aus "
                "list_events). Prüft vorher auf Überlappungen und verschiebt NICHT, wenn es einen "
                "Konflikt gibt. Funktioniert nur für Agent-eigene Termine."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "description": "Referenz aus list_events, z. B. 'evt2'."},
                    "new_date": {"type": "string", "description": "Neues Datum YYYY-MM-DD (optional, falls der Tag wechselt)."},
                    "new_start_time": {"type": "string", "description": "Neue Startzeit HH:MM (optional)."},
                    "new_end_time": {"type": "string", "description": "Neue Endzeit HH:MM (optional)."},
                },
                "required": ["ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_event",
            "description": (
                "Löscht einen vom Agent erstellten Termin (per Referenz aus list_events). "
                "Funktioniert nur für Agent-eigene Termine."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "description": "Referenz aus list_events, z. B. 'evt2'."},
                },
                "required": ["ref"],
            },
        },
    },
]

WEEKDAYS = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]


def _fmt_event_line(ref: str, ev: dict) -> str:
    day = WEEKDAYS[ev["start"].weekday()]
    owner = "vom Agent erstellt – darf geändert/gelöscht werden" if ev["is_agent"] \
        else "eigener Termin – schreibgeschützt"
    if ev["all_day"]:
        when = f"{day}, {ev['start'].strftime('%d.%m.')} (ganztägig)"
    else:
        when = f"{day}, {ev['start'].strftime('%d.%m.')} {ev['start'].strftime('%H:%M')}–{ev['end'].strftime('%H:%M')}"
    return f"[{ref}] {when} – {ev['summary']} ({owner})"


def _fmt_conflicts(conflicts: list[dict]) -> str:
    parts = []
    for ev in conflicts:
        parts.append(f"{ev['start'].strftime('%H:%M')}–{ev['end'].strftime('%H:%M')} {ev['summary']}")
    return "; ".join(parts)


def execute_tool(name: str, args: dict, ref_map: dict) -> str:
    """Execute a single tool call and return a human-readable result string for the LLM.

    ref_map is a per-conversation dict mapping short references to real event ids;
    list_events populates it, move/delete_event resolve through it.
    """
    try:
        if name == "list_events":
            days = int(args.get("days_ahead") or 14)
            events = calendar_service.get_upcoming_events(days)
            if not events:
                return "Keine Termine in diesem Zeitraum."
            lines = []
            for i, ev in enumerate(events, start=1):
                ref = f"evt{i}"
                ref_map[ref] = ev["id"]
                lines.append(_fmt_event_line(ref, ev))
            return "\n".join(lines)

        if name == "create_event":
            event_date = date.fromisoformat(args["date"])
            conflicts = calendar_service.find_overlaps(event_date, args["start_time"], args["end_time"])
            if conflicts:
                return (
                    f"Konflikt: In diesem Zeitfenster liegt bereits {_fmt_conflicts(conflicts)}. "
                    "Termin wurde NICHT erstellt. Schlage Djamal eine freie Alternative vor."
                )
            calendar_service.create_event(
                args["title"], event_date, args["start_time"], args["end_time"], args.get("description", "")
            )
            return (
                f"Termin '{args['title']}' am {event_date.strftime('%d.%m.%Y')} "
                f"{args['start_time']}–{args['end_time']} wurde erstellt."
            )

        if name in ("move_event", "delete_event"):
            ref = args.get("ref", "")
            event_id = ref_map.get(ref)
            if not event_id:
                return (
                    f"Unbekannte Referenz '{ref}'. Rufe zuerst list_events auf, "
                    "um gültige Referenzen zu erhalten."
                )

            try:
                ev = calendar_service.get_event(event_id)
            except calendar_service.CalendarError as e:
                return f"Termin konnte nicht geladen werden: {e}"

            if not ev["is_agent"]:
                return (
                    f"'{ev['summary']}' wurde nicht vom Agent erstellt und darf daher nicht "
                    "geändert oder gelöscht werden. Bitte teile Djamal mit, dass er das selbst tun muss."
                )

            if name == "delete_event":
                calendar_service.delete_event(event_id)
                return f"Termin '{ev['summary']}' wurde gelöscht."

            # move_event
            new_date = date.fromisoformat(args["new_date"]) if args.get("new_date") else ev["start"].date()
            new_start = args.get("new_start_time") or ev["start"].strftime("%H:%M")
            new_end = args.get("new_end_time") or ev["end"].strftime("%H:%M")

            conflicts = calendar_service.find_overlaps(new_date, new_start, new_end, exclude_event_id=event_id)
            if conflicts:
                return (
                    f"Verschieben nicht möglich: Im Zielzeitfenster liegt bereits "
                    f"{_fmt_conflicts(conflicts)}. Der Termin bleibt unverändert. "
                    "Schlage Djamal eine freie Alternative vor."
                )

            updated = calendar_service.update_event(
                event_id, date_=new_date, start_time=new_start, end_time=new_end
            )
            return (
                f"Termin '{updated['summary']}' wurde auf {new_date.strftime('%d.%m.%Y')} "
                f"{new_start}–{new_end} verschoben."
            )

        return f"Unbekanntes Tool: {name}"

    except calendar_service.CalendarNotConnectedError:
        return "Google Calendar ist nicht verbunden. Djamal muss zuerst /connect ausführen."
    except calendar_service.CalendarError as e:
        return f"Kalender-Fehler: {e}"
    except (ValueError, KeyError) as e:
        return f"Ungültige Angaben für das Tool: {e}"
