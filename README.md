# Life OS Agent

Djamal's personal AI life planner. Runs as a Telegram bot, plans the upcoming week with a
LangGraph agent (Groq `llama-3.3-70b-versatile` + RAG over personal notes), exports a
polished weekly PDF calendar, and syncs the plan to Google Calendar.

## Features

- `/plan` — Generates a weekly plan, asks for approval/feedback in Telegram (human-in-the-loop,
  up to 3 revision rounds), then exports a PDF and syncs it to Google Calendar.
- `/today` — Shows today's Google Calendar events.
- `/add <text>` — Creates a calendar event from natural language (e.g. `/add Zahnarzt Dienstag 15 Uhr`).
- `/connect` — Connects your Google Calendar account (headless OAuth via Telegram).
- `/status` — Shows Google Calendar connection status, last plan date, and RAG knowledge base size.
- **Proactive reminders** (no command needed) via the bot's JobQueue — see `src/reminders.py`:
  morning briefing with the day's agenda (weekdays 06:05, weekend 08:30), Plattfuß/movement
  nudge before dinner (Mon/Wed/Thu 17:45), a free-Friday-evening reminder, a nightly phone-away
  reminder (22:45), and a Sunday nudge to build next week's plan.
- **Date-aware planning** — `src/config.py` pins the BMW start date (01.07.2026) and the
  strength-training clearance date (01.08.2026). The planner leaves weekday daytimes free before
  BMW starts, splits the Friday work day around Jummah, and only allows strength training once
  cleared. The fixed blocks (commute, work, Jummah, dinner, Sunday family) are injected
  deterministically in `src/graph.py`, so they no longer depend on the LLM remembering them.
- Voice messages are transcribed (Groq Whisper) and handled like text.
- Free-text chat is a RAG-grounded conversation with Groq that can also **manage your
  calendar via tool calls**: ask things like *"Hab ich Mittwoch was vor?"*, *"Verschieb
  mein Training am Dienstag auf 18 Uhr"* or *"Lösch den Termin am Freitagvormittag"*.
  - **Overlap-safe:** before creating or moving an event the agent checks for conflicts
    and refuses (suggesting a free slot) instead of double-booking.
  - **Safety guardrail:** the agent only ever modifies or deletes events it created
    itself. Agent events are tagged via `extendedProperties` (the hard gate) and carry a
    visible `🤖 Life OS Agent` marker in their description. Events you created yourself are
    read-only to the agent.

## Requirements

- Python 3.12
- `ffmpeg` (for voice message transcription): `sudo apt install ffmpeg`
- A [Groq API key](https://console.groq.com/keys)
- A Telegram bot token (create one via [@BotFather](https://t.me/BotFather))
- (Optional, for Google Calendar features) a Google Cloud OAuth client

## Setup

```bash
cd ~/agents/life-os-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

Copy `.env.example` to `.env` and fill in your keys, or export them in `~/.bashrc`:

```bash
export GROQ_API_KEY="..."
export TELEGRAM_TOKEN="..."
```

### Build the RAG knowledge base

The agent uses `data/notes/mein_leben.txt` as its personal knowledge base. Build (or rebuild
after editing the file) the ChromaDB index:

```bash
python -c "from src.rag import build_db; build_db()"
```

### Google Calendar setup (optional but recommended)

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (or reuse an existing one).
3. Enable the **Google Calendar API** for the project (APIs & Services → Enable APIs and Services).
4. Go to **APIs & Services → Credentials → Create Credentials → OAuth client ID**.
5. Choose application type **Desktop app**.
6. Download the resulting JSON file, rename it to `credentials.json`, and place it in the
   project root (`~/agents/life-os-agent/credentials.json`).
7. Start the bot (see below) and send `/connect` in Telegram.
8. Open the link the bot sends you in any browser, sign in with your Google account, and
   grant calendar access.
9. After confirming, the browser will redirect to a `http://localhost/...` URL that fails
   to load — that's expected. Copy the `code` parameter from the address bar (or the whole
   URL) and send it back to the bot in Telegram.
10. The bot saves `token.json` and confirms the connection. Tokens are refreshed automatically.

If `credentials.json` is missing or `/connect` hasn't been completed, calendar-related
features degrade gracefully (the agent still plans your week, just without calendar sync).

## Running the bot

```bash
source .venv/bin/activate
python -m src.telegram_bot
```

The bot runs as a long-running process (`app.run_polling()`). Use a tool like `systemd`,
`tmux`, or `screen` to keep it running on your Ubuntu machine.

## Project structure

```
life-os-agent/
├── src/
│   ├── __init__.py
│   ├── graph.py             # LangGraph agent (plan / approve / export / sync)
│   ├── rag.py                # ChromaDB RAG over personal notes
│   ├── agent.py               # Groq LLM client + chat helpers
│   ├── pdf_export.py          # WeasyPrint weekly PDF calendar
│   ├── calendar_service.py    # Google Calendar OAuth + event read/create/move/delete
│   ├── calendar_tools.py      # LLM tool schemas + dispatcher (overlap-safe, own-events-only)
│   └── telegram_bot.py        # Telegram bot entry point
├── data/
│   ├── notes/mein_leben.txt   # Personal knowledge base for RAG
│   ├── todos/
│   ├── exports/                # Generated weekly PDFs
│   └── state.json              # Small bot state (e.g. last plan date)
├── memory/chroma/              # ChromaDB persistence
├── credentials.json             # Google OAuth client (you provide this)
├── token.json                   # Google OAuth token (auto-generated)
├── requirements.txt
├── .env.example
└── README.md
```

## Notes

- Categories used in the weekly plan/PDF: **BMW Vorbereitung**, **Sport & Fitness**,
  **Familie & Ehe**, **Life OS Projekt**, **KI & Lernen**, **Freizeit & Erholung**.
- The agent always replies to Djamal in German.
- `credentials.json` and `token.json` contain secrets — keep them out of version control.
