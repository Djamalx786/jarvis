# Deploying the Life OS Agent to Railway

This runs the Telegram bot 24/7 so you can message it from your phone anytime.
The app is containerized (`Dockerfile`) and reads all secrets from environment
variables — **no secrets are ever committed to GitHub**.

## 1. Push the code (already done)

The repo lives at `git@github.com:Djamalx786/jarvis.git` on branch `main`.

## 2. Create the Railway service

1. Go to <https://railway.app> → **New Project** → **Deploy from GitHub repo**.
2. Pick the `jarvis` repo. Railway detects the `Dockerfile` automatically.

## 3. Set the environment variables

In the Railway service → **Variables**, add:

| Variable | Value |
|---|---|
| `GROQ_API_KEY` | your Groq API key |
| `TELEGRAM_TOKEN` | your Telegram bot token |
| `GOOGLE_CREDENTIALS_JSON` | the **entire contents** of your local `credentials.json` |
| `GOOGLE_TOKEN_JSON` | the **entire contents** of your local `token.json` |

To copy the JSON file contents to your clipboard (Linux):

```bash
cat credentials.json | xclip -selection clipboard   # then paste into Railway
cat token.json       | xclip -selection clipboard
```

(Or just open the files and copy the text.) On boot, `src/startup.py` writes these
back to `credentials.json` / `token.json` inside the container. Because `token.json`
contains a refresh token, Google Calendar reconnects automatically — no `/connect`
needed on the server.

The Google calendar variables are optional: without them the bot still runs (plans,
chat, reminders), just without calendar sync.

## 4. Deploy

Railway builds and starts `python main.py`. Watch the **Deploy Logs** for
`🤖 Bot läuft...`. Then message your bot from your phone — done.

## 5. (Optional but recommended) Add a Volume for persistence

The container filesystem resets on every redeploy. To keep the RAG index, the
plan checkpoints, and known chat IDs across redeploys, add a Railway **Volume**
mounted at `/app/data` and one at `/app/memory`. Without a volume the bot still
works — it just rebuilds the RAG index on each cold start and re-learns your chat
ID the next time you message it.

## Notes / resource needs

- The image includes `sentence-transformers` + `torch` (CPU) for RAG embeddings,
  which needs roughly 1 GB RAM. Pick a Railway plan accordingly.
- System libraries for the PDF (`Pango`/`Cairo`) and `ffmpeg` for voice messages
  are installed by the `Dockerfile`.
- Run only **one** instance — two bot instances polling the same token will conflict.
