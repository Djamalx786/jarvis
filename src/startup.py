"""Runtime bootstrap for hosted environments (e.g. Railway).

Locally everything lives as files in the project root (.env, credentials.json,
token.json) and the RAG index is prebuilt. In a hosted container those files are
not present (they are git-ignored), so this module recreates them from environment
variables at startup and builds the RAG index on first boot.

Set these in your host's environment (Railway > Variables):
  GROQ_API_KEY            - Groq API key
  TELEGRAM_TOKEN          - Telegram bot token
  GOOGLE_CREDENTIALS_JSON - full contents of credentials.json (optional, for calendar)
  GOOGLE_TOKEN_JSON       - full contents of token.json (optional, for calendar)
"""
import os

from src import calendar_service, rag


def _materialize(env_var: str, path: str) -> None:
    """Write a JSON file from an env var if the var is set and the file is missing."""
    content = os.environ.get(env_var)
    if content and not os.path.exists(path):
        with open(path, "w") as f:
            f.write(content)
        print(f"📄 {path} aus {env_var} wiederhergestellt.")


def prepare_runtime() -> None:
    """Recreate credential files from env vars and ensure the RAG index exists."""
    _materialize("GOOGLE_CREDENTIALS_JSON", calendar_service.CREDENTIALS_PATH)
    _materialize("GOOGLE_TOKEN_JSON", calendar_service.TOKEN_PATH)

    try:
        if rag.get_doc_count() == 0:
            print("📚 RAG-Index ist leer – baue ihn aus data/notes auf...")
            rag.build_db()
    except Exception as e:
        # The bot is still usable without RAG; just log and continue.
        print(f"⚠️ RAG-Index konnte nicht aufgebaut werden: {e}")
