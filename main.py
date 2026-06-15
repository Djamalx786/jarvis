"""Entry point for the Life OS Agent. Loads env vars and starts the Telegram bot.

Run with:  python main.py
"""
from dotenv import load_dotenv

from src.telegram_bot import main

if __name__ == "__main__":
    load_dotenv()
    main()
