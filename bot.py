# bot.py — KDxClaims Telegram Bot
# python-telegram-bot v20+
import os
import logging
import asyncio
import requests
from datetime import datetime
from telegram import Update, InputFile
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)

API_BASE   = os.getenv("API_BASE", "https://kdxclaims.onrender.com")
BOT_TOKEN  = os.getenv("BOT_TOKEN")  # required
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # e.g. https://<your-render-app>/tg/webhook

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kdxclaims-tg")

# -------- Helpers --------
def call_extract(image_bytes: bytes, filename: str = "bill.jpg"):
    """POST /extract with a file, return JSON or raise."""
    url = f"{API_BASE}/extract"
    files = {"file": (filename, image_bytes, "image/jpeg")}
    r = requests.post(url, files=files, timeout=120)
    r.raise_for_status()
    return r.json()

def call_upload_claim(uid: int, chat_id: int, fields: dict):
    """POST /upload-claim using extracted fields."""
    url = f"{API_BASE}/upload-claim"
    payload = {
        "telegram_user_id": uid,
        "telegram_chat_id": chat_id,
        "claim_type": "fuel",  # default; change via command if you support more
        "claim_date": fields.get("txn_date"),
        "claim_time": fields.get("time"),
        "total_rs": fields.get("total_amount"),
        "station": fields.get("source"),
        "reference_no": fields.get("transaction_id"),
        "rate_rs_per_l": fields.get("rate_rs_per_l"),
        "volume_l": fields.get("volume_l"),
        "notes": "auto-uploaded from TG",
    }
    r = requests.post(url, json=payload, timeout=60)
    if r.status_code == 404:
        # User not registered in user_telegram
        return {"_error": "not_registered"}
    r.raise_for_status()
    return r.json()

def fmt_ack(fields: dict) -> str:
    parts = [
        "🧾 *Bill Parsed*",
        f"*Ref:* `{fields.get('transaction_id') or '—'}`",
        f"*Date:* `{fields.get('txn_date') or '—'}`  *Time:* `{fields.get('time') or '—'}`",
        f"*Amount:* ₹{(fields.get('total_amount') or 0):,.2f}",
    ]
    if fields.get("source"):
        parts.append(f"*Station:* {fields['source']}")
    if fields.get("rate_rs_per_l") and fields.get("volume_l"):
        parts.append(f"*Rate:* {fields['rate_rs_per_l']}  |  *Vol:* {fields['volume_l']} L")
    return "\n".join(parts)

# -------- Handlers --------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Hi! Send me a *photo or image file* of your bill.\n"
        "I’ll extract details and log your claim automatically.\n\n"
        "• /help — usage\n"
        "• /health — API health\n"
        "• /claimtype fuel|service|insurance|accessories|driver_salary — set default type (per chat)\n"
        "• /echo — debug"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Just send a bill *photo* or *image file*.\n"
        "I’ll parse and save it. Use /claimtype to switch from fuel to others.",
        parse_mode=ParseMode.MARKDOWN,
    )

async def health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        r = requests.get(f"{API_BASE}/health", timeout=10)
        await update.message.reply_text(r.text)
    except Exception as e:
        await update.message.reply_text(f"Health check failed: {e}")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("OK")

async def set_claim_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text("Usage: /claimtype fuel|service|insurance|accessories|driver_salary")
    ct = context.args[0].strip().lower()
    valid = {"fuel", "service", "insurance", "accessories", "driver_salary"}
    if ct not in valid:
        return await update.message.reply_text(f"Invalid. Choose one of: {', '.join(sorted(valid))}")
    # chat-scoped default
    context.chat_data["claim_type"] = ct
    await update.message.reply_text(f"Default claim_type set to *{ct}*", parse_mode=ParseMode.MARKDOWN)

async def handle_photo_or_doc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user = update.effective_user
    chat = update.effective_chat

    # Pick the highest-resolution photo if it's a photo; else handle image document
    file_id = None
    filename = "bill.jpg"

    if msg.photo:
        file_id = msg.photo[-1].file_id
        filename = "bill.jpg"
    elif msg.document and (msg.document.mime_type or "").startswith("image/"):
        file_id = msg.document.file_id
        filename = msg.document.file_name or "bill.jpg"
    else:
        return await msg.reply_text("Please send a *photo* or *image file* (jpg/png/webp).", parse_mode=ParseMode.MARKDOWN)

    try:
        await msg.chat.send_action(ChatAction.UPLOAD_PHOTO)
        tg_file = await context.bot.get_file(file_id)
        # Download bytes into memory
        image_bytes = await tg_file.download_as_bytearray()

        # 1) Extract
        fields = call_extract(bytes(image_bytes), filename)

        # Allow per-chat claim_type override
        claim_type = context.chat_data.get("claim_type", "fuel")
        fields["_claim_type"] = claim_type  # optional note

        # 2) Upload claim
        res = call_upload_claim(user.id, chat.id, fields)
        if isinstance(res, dict) and res.get("_error") == "not_registered":
            return await msg.reply_text(
                "You are not registered yet. Please link your Telegram in the app (user_telegram table).",
            )

        # 3) ACK to user
        ack_text = fmt_ack(fields)
        if isinstance(res, dict) and res.get("ack"):
            ack_text += "\n\n—\n" + res["ack"]
        await msg.reply_text(ack_text, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)

    except requests.HTTPError as he:
        try:
            j = he.response.json()
            detail = j.get("detail") if isinstance(j, dict) else he.response.text
        except Exception:
            detail = he.response.text if he.response is not None else str(he)
        logger.exception("HTTPError in handle: %s", detail)
        await msg.reply_text(f"❌ API error: {detail}")
    except Exception as e:
        logger.exception("Unexpected error")
        await msg.reply_text(f"❌ Failed: {e}")

# -------- Main entry --------
def build_app():
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN not set")
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("health", health))
    app.add_handler(CommandHandler("echo", echo))
    app.add_handler(CommandHandler("claimtype", set_claim_type))
    app.add_handler(MessageHandler(filters.PHOTO | (filters.Document.IMAGE), handle_photo_or_doc))

    return app

async def run_polling():
    app = build_app()
    logger.info("Starting polling...")
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    await app.updater.idle()

async def run_webhook():
    app = build_app()
    logger.info("Starting webhook...")
    await app.initialize()
    await app.start()
    # Set webhook
    if not WEBHOOK_URL:
        raise RuntimeError("WEBHOOK_URL not set for webhook mode")
    await app.bot.set_webhook(url=WEBHOOK_URL, drop_pending_updates=True)
    # PTB v20 uses built-in webhook server via .run_webhook (if needed),
    # but on Render we usually use an external ASGI server (see below).
    await asyncio.Event().wait()

if __name__ == "__main__":
    mode = os.getenv("TG_MODE", "polling").lower()
    if mode == "webhook":
        asyncio.run(run_webhook())
    else:
        asyncio.run(run_polling())
