
import os
import logging

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ALLOWED_USER_ID = os.getenv("TELEGRAM_ALLOWED_USER_ID", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")

client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def is_allowed(update: Update) -> bool:
    user = update.effective_user
    return bool(
        user
        and TELEGRAM_ALLOWED_USER_ID
        and str(user.id) == TELEGRAM_ALLOWED_USER_ID
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return

    await update.effective_message.reply_text(
        "AI Coding Agent તૈયાર છે. મને મેસેજ મોકલો."
    )


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return

    if not client or not OPENAI_MODEL:
        await update.effective_message.reply_text(
            "AI setup અધૂરું છે. API key અને model configuration તપાસો."
        )
        return

    message = update.effective_message.text or ""

    try:
        response = await client.responses.create(
            model=OPENAI_MODEL,
            instructions=(
                "You are a careful coding assistant. "
                "Explain changes before suggesting them. "
                "Never claim to have edited files or run tests unless "
                "that actually happened. Never request API secrets in chat."
            ),
            input=message,
        )
        await update.effective_message.reply_text(
            response.output_text[:4000] or "જવાબ મળ્યો નથી."
        )
    except Exception:
        logging.exception("AI request failed")
        await update.effective_message.reply_text(
            "AI request નિષ્ફળ ગઈ. Server logs તપાસો; secrets શેર ન કરો."
        )


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing")

    if not TELEGRAM_ALLOWED_USER_ID:
        raise RuntimeError("TELEGRAM_ALLOWED_USER_ID missing")

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, chat)
    )

    application.run_polling()


if __name__ == "__main__":
    main()
  
