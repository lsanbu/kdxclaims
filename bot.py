import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

API_URL = "http://127.0.0.1:8000/upload-claim"  # your FastAPI
BOT_TOKEN = "7657777300:AAHf78wpToC4U4fctWzBx0LT-Umshqflh5A"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Welcome to KDxClaims! Send me a bill photo or details.")

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # For now, assume text = amount
    payload = {
        "telegram_user_id": update.effective_user.id,
        "telegram_chat_id": update.message.chat_id,
        "claim_type": "fuel",
        "claim_date": "2025-08-24",
        "total_rs": 123.45,
        "station": "Demo Station",
        "reference_no": "TEST123"
    }
    res = requests.post(API_URL, json=payload)
    ack = res.json().get("ack", "❌ Failed")
    await update.message.reply_text(ack)

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.run_polling()

if __name__ == "__main__":
    main()
