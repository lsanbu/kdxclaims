# main.py — KDxClaims API + Telegram Webhook (single service)

from __future__ import annotations

import os
import re
import logging
import tempfile
import shutil
from datetime import datetime, date, timedelta
import zoneinfo
from typing import Optional, Dict, Any, Tuple

import requests
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supabase import create_client, Client
from dataclasses import asdict
from receipt_ocr import parse_receipt_image
import cv2, pytesseract

# --- Telegram imports ---
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)

# ---------------- App setup ----------------
load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"ok": True, "service": "kdxclaims-api+tg"}

@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}

# ---------------- Logging & middleware ----------------
logger = logging.getLogger("uvicorn.error")
logging.basicConfig(level=logging.INFO)

@app.middleware("http")
async def log_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"ok": False, "error": "Internal Server Error"})

# ---------------- Config ----------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BOT_TOKEN    = os.getenv("BOT_TOKEN")  # used by both API TG send & bot
API_BASE     = os.getenv("API_BASE", "https://kdxclaims.onrender.com")
WEBHOOK_URL  = os.getenv("WEBHOOK_URL", f"{API_BASE.rstrip('/')}/tg/")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY.")

# ---------------- Globals ----------------
supabase: Optional[Client] = None
_httpx: Optional[httpx.AsyncClient] = None
_ptb_app: Optional[Application] = None

# ---------------- FastAPI lifecycle ----------------
@app.on_event("startup")
async def _startup():
    # Supabase
    global supabase
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Async HTTP client
    global _httpx
    _httpx = httpx.AsyncClient(timeout=60)

    # Telegram bot (webhook mode)
    if not BOT_TOKEN:
        logger.warning("BOT_TOKEN not set — Telegram webhook disabled.")
        return

    global _ptb_app
    _ptb_app = build_tg_app()
    await _ptb_app.initialize()
    await _ptb_app.start()

    # Set webhook (idempotent)
    try:
        await _ptb_app.bot.set_webhook(url=WEBHOOK_URL, drop_pending_updates=True)
        logger.info("Telegram webhook set to: %s", WEBHOOK_URL)
    except Exception:
        logger.exception("Failed to set Telegram webhook")

@app.on_event("shutdown")
async def _shutdown():
    try:
        if _ptb_app:
            await _ptb_app.stop()
    except Exception:
        logger.exception("Error stopping PTB app")
    try:
        if _httpx:
            await _httpx.aclose()
    except Exception:
        logger.exception("Error closing httpx client")

# ---------------- Supabase helpers ----------------
def _sb() -> Client:
    if supabase is None:
        raise RuntimeError("Supabase client not initialized")
    return supabase

# ---------------- Models ----------------
class ClaimIn(BaseModel):
    telegram_user_id: int
    telegram_chat_id: int
    claim_type: str   # "fuel", "driver_salary", "insurance", "service", "accessories"
    claim_date: str   # YYYY-MM-DD
    claim_time: Optional[str] = None
    total_rs: float
    station: Optional[str] = None
    reference_no: Optional[str] = None
    rate_rs_per_l: Optional[float] = None
    volume_l: Optional[float] = None
    notes: Optional[str] = None

# ---------------- Business helpers ----------------
def get_user_id(telegram_user_id: int) -> str:
    result = (
        _sb().table("user_telegram")
        .select("user_id")
        .eq("telegram_user_id", telegram_user_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="User not registered")
    return result.data[0]["user_id"]

def get_user_prefs(user_id: str) -> Dict[str, Any]:
    res = _sb().table("user_prefs").select("*").eq("user_id", user_id).execute()
    return res.data[0] if res.data else {
        "cutoff_day": 15, "tz": "Asia/Kolkata",
        "weekly_reminder": True, "daily_pre_cutoff": True
    }

def current_period(cutoff_day: int, tzname: str) -> Tuple[date, date]:
    tz = zoneinfo.ZoneInfo(tzname)
    now = datetime.now(tz).date()

    def month_add(d: date, months: int) -> date:
        y = d.year + (d.month - 1 + months) // 12
        m = (d.month - 1 + months) % 12 + 1
        day = min(
            d.day,
            [31, 29 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 28,
             31,30,31,30,31,31,30,31,30,31][m-1]
        )
        return date(y, m, day)

    if now.day <= cutoff_day:
        start_m = month_add(now.replace(day=1), -1)
        last_day_lookup = [31, 29 if start_m.year % 4 == 0 and (start_m.year % 100 != 0 or start_m.year % 400 == 0) else 28,
                           31,30,31,30,31,31,30,31,30,31]
        start_day = min(cutoff_day + 1, last_day_lookup[start_m.month - 1])
        start = start_m.replace(day=start_day)
        end = now.replace(day=cutoff_day)
    else:
        start = now.replace(day=cutoff_day + 1)
        nm = month_add(now.replace(day=1), 1)
        end = nm.replace(day=cutoff_day)
    return start, end

def period_totals(user_id: str, start: date, end: date):
    q = (_sb().table("claims")
         .select("claim_type,total_rs,claim_date")
         .eq("user_id", user_id)
         .gte("claim_date", str(start))
         .lte("claim_date", str(end))
         .execute())
    rows = q.data or []
    total = sum(r.get("total_rs", 0) for r in rows)
    by_type: Dict[str, float] = {}
    for r in rows:
        by_type[r["claim_type"]] = by_type.get(r["claim_type"], 0) + r.get("total_rs", 0)
    last_txn = max(rows, key=lambda r: r["claim_date"]) if rows else None
    return total, by_type, last_txn

def send_tg(chat_id: int, text: str):
    if not BOT_TOKEN:
        logger.info("TG send skipped: BOT_TOKEN missing")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=10)
        if r.status_code != 200:
            logger.warning("TG send error: %s %s", r.status_code, r.text)
    except Exception as e:
        logger.warning("TG send exception: %r", e)

def list_users():
    res = _sb().table("user_telegram").select("user_id, telegram_chat_id").execute()
    return res.data or []

# ---------------- OCR.space config & helpers ----------------
OCRSPACE_KEY = os.getenv("OCRSPACE_KEY")
OCRSPACE_URL = "https://api.ocr.space/parse/image"

def _ocrspace_from_url(image_url: str) -> str:
    if not OCRSPACE_KEY:
        raise HTTPException(503, "OCRSPACE_KEY not configured")
    try:
        resp = requests.post(
            OCRSPACE_URL,
            data={"apikey": OCRSPACE_KEY, "url": image_url, "OCREngine": 2, "language": "eng"},
            timeout=60,
        )
        j = resp.json()
    except Exception as e:
        raise HTTPException(502, f"OCR network error: {e}")
    if not j.get("IsErroredOnProcessing") and j.get("ParsedResults"):
        return " \n".join(pr.get("ParsedText","") for pr in j["ParsedResults"])
    raise HTTPException(502, f"OCR failed: {j.get('ErrorMessage') or j}")

def _ocrspace_from_file(file_bytes: bytes) -> str:
    if not OCRSPACE_KEY:
        raise HTTPException(503, "OCRSPACE_KEY not configured")
    try:
        resp = requests.post(
            OCRSPACE_URL,
            files={"file": ("bill.jpg", file_bytes)},
            data={"apikey": OCRSPACE_KEY, "OCREngine": 2, "language": "eng"},
            timeout=120,
        )
        j = resp.json()
    except Exception as e:
        raise HTTPException(502, f"OCR network error: {e}")
    if not j.get("IsErroredOnProcessing") and j.get("ParsedResults"):
        return " \n".join(pr.get("ParsedText","") for pr in j["ParsedResults"])
    raise HTTPException(502, f"OCR failed: {j.get('ErrorMessage') or j}")

# ---------------- Regex parser ----------------
DATE_PATTERNS = [
    r"\b(\d{1,2}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{2,4})\b",
    r"\b(\d{1,2}\s*-\s*[A-Za-z]{3}\s*-\s*\d{4})\b",
    r"\b([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})\b",
]
TIME_PATTERNS = [r"(?<!\d)(\d{1,2}\s*[:：]\s*\d{2}(?:\s*[:：]\s*\d{2})?)(?!\d)"]
AMOUNT_PATTERNS = [
    r"(?:Amount|Amt|Total|Grand\s*Total|Preset\s*Value)\s*[:=]?\s*₹?\s*([0-9][\d,]*\.?\d{0,2})\b",
    r"\b(?:INR|Rs\.?|₹)\s*([0-9][\d,]*\.?\d{0,2})\b",
]
REF_PATTERNS = [
    r"(?:TXN\s*NO|Txn\s*Id|TXN\s*Id|Transaction\s*Id)\s*[:#]?\s*([A-Z0-9\-\/]{5,})",
    r"(?:Invoice\s*No\.?|Receipt\s*No\.?|Rcpt\s*No\.?)\s*[:#]?\s*([A-Z0-9\-\/]{5,})",
]
RATE_PATTERNS = [r"(?:Rate|Rate\(Rs/?L\)|Rate/Ltr\.?)\s*[:=]?\s*([0-9]+(?:\.[0-9]{1,2})?)"]
VOL_PATTERNS  = [r"(?:Volume|Vol|Qty)\s*(?:\(|L|Ltr|Litres|Ltrs)?\.?\s*[:=]?\s*0*([0-9]+(?:\.[0-9]{1,3})?)"]
SOURCE_PATTERNS = [
    (r"Indian\s*Oil|IndianOil|IOC", "IndianOil"),
    (r"\bHP\b|Hindustan\s*Petroleum", "HP"),
    (r"Bharat\s*Petroleum|BPCL", "BPCL"),
]

def _norm(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s*/\s*", "/", s)
    return s

def _first(patterns, text, flags=re.IGNORECASE):
    for p in patterns:
        m = re.search(p, text, flags)
        if m:
            return m.group(1).strip()
    return None

def _norm_amount_str(x: str | None) -> float | None:
    if not x: return None
    x = x.replace(",", "")
    try: return float(x)
    except: return None

def _normalize_date_any(cand: str | None) -> str | None:
    if not cand: return None
    cand = cand.strip().replace("IUL", "JUL").replace("O", "0")
    for fmt in ("%d/%m/%Y","%d-%m-%Y","%d/%m/%y","%d-%m-%y","%d/%b/%Y","%d-%b-%Y","%b %d, %Y","%d-%b-%y"):
        try:
            return datetime.strptime(cand, fmt).strftime("%Y-%m-%d")
        except: continue
    m = re.match(r"(\d{1,2})-([A-Za-z]{3})-(\d{2,4})", cand)
    if m:
        d, mon, y = m.groups()
        if len(y)==2: y = ("20" if int(y)<50 else "19")+y
        try:
            return datetime.strptime(f"{d}-{mon}-{y}", "%d-%b-%Y").strftime("%Y-%m-%d")
        except: pass
    return None

def _normalize_time(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip().replace("：", ":")
    s = re.sub(r"\s*:\s*", ":", s)
    m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", s)
    if not m:
        return s
    hh, mm, ss = m.group(1), m.group(2), m.group(3) or None
    try:
        h = int(hh); m_ = int(mm); s_ = int(ss) if ss is not None else None
        if not (0 <= h <= 23 and 0 <= m_ <= 59 and (s_ is None or 0 <= s_ <= 59)):
            return s
    except:
        return s
    return f"{h:02d}:{m_:02d}:{(s_ if s_ is not None else 0):02d}" if ss is not None else f"{h:02d}:{m_:02d}"

def _source_from_text(t: str) -> str | None:
    for pat, label in SOURCE_PATTERNS:
        if re.search(pat, t, re.IGNORECASE):
            return label
    return None

def _parse_bill_text_strict(text: str):
    t = _norm(text)
    txn_date_raw = _first(DATE_PATTERNS, t)
    txn_date = _normalize_date_any(txn_date_raw)
    ref_txn = _first(REF_PATTERNS, t)
    amt_raw = _first(AMOUNT_PATTERNS, t)
    total_amount = _norm_amount_str(amt_raw)
    rate = _first(RATE_PATTERNS, t)
    rate_val = _norm_amount_str(rate)
    vol = _first(VOL_PATTERNS, t)
    vol_val = _norm_amount_str(vol)
    time_raw = _first(TIME_PATTERNS, t)
    time_norm = _normalize_time(time_raw)
    if total_amount is None and rate_val is not None and vol_val is not None:
        total_amount = round(rate_val * vol_val, 2)
    return {
        "txn_date": txn_date,
        "reference_no": ref_txn,
        "time": time_norm,
        "total_amount": total_amount,
        "rate_rs_per_l": rate_val,
        "volume_l": vol_val,
        "source": _source_from_text(t),
        "preview": t[:600],
    }

# ---------------- Upload constraints ----------------
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_BYTES = 10 * 1024 * 1024  # 10 MB

# ---------------- OCR endpoint ----------------
@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    try:
        if not file:
            raise HTTPException(400, "Missing file field 'file'")
        ctype = (file.content_type or "").lower()
        if ctype not in ALLOWED_TYPES:
            raise HTTPException(415, f"Unsupported content type: {file.content_type}")
        data = await file.read()
        if not data:
            raise HTTPException(400, "Empty file")
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"File too large (> {MAX_BYTES} bytes)")

        if shutil.which("tesseract"):
            suffix = os.path.splitext(file.filename or "")[-1] or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            try:
                res = parse_receipt_image(tmp_path)
                return asdict(res)
            finally:
                try: os.remove(tmp_path)
                except Exception: pass
        else:
            if not OCRSPACE_KEY:
                raise HTTPException(503, "Tesseract not available and OCRSPACE_KEY not set.")
            text = _ocrspace_from_file(data)
            f = _parse_bill_text_strict(text)
            return {
                "transaction_id": f.get("reference_no"),
                "txn_date": f.get("txn_date"),
                "time": f.get("time"),
                "total_amount": f.get("total_amount"),
                "source": f.get("source"),
                "rate_rs_per_l": f.get("rate_rs_per_l"),
                "volume_l": f.get("volume_l"),
                "raw_text": text[:5000],
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("extract failed")
        raise HTTPException(status_code=500, detail=f"/extract failed: {type(e).__name__}: {e}")

# ---------------- Claims API ----------------
@app.post("/upload-claim")
def upload_claim(claim: ClaimIn):
    valid_claim_types = {"fuel", "driver_salary", "insurance", "service", "accessories"}
    if claim.claim_type not in valid_claim_types:
        raise HTTPException(400, detail=f"Invalid claim_type. Must be one of {valid_claim_types}")

    user_id = get_user_id(claim.telegram_user_id)
    row = {
        "user_id": user_id,
        "claim_type": claim.claim_type,
        "claim_date": claim.claim_date,
        "claim_time": claim.claim_time,
        "total_rs": claim.total_rs,
        "station": claim.station,
        "reference_no": claim.reference_no,
        "rate_rs_per_l": claim.rate_rs_per_l,
        "volume_l": claim.volume_l,
        "notes": claim.notes,
        "created_at": datetime.utcnow().isoformat(),
    }

    try:
        insert_res = _sb().table("claims").insert(row).execute()
    except Exception:
        logger.exception("Claims insert failed. Row=%s", row)
        raise HTTPException(status_code=500, detail="Insert failed")

    if not insert_res.data:
        raise HTTPException(status_code=500, detail="Insert failed")

    if claim.claim_type == "fuel":
        ack = (
            f"✅ Fuel claim saved\n"
            f"Date {claim.claim_date} {claim.claim_time or ''}\n"
            f"Station: {claim.station or 'N/A'}\n"
            f"Ref: {claim.reference_no or 'N/A'}\n"
            f"Rate: {claim.rate_rs_per_l or '—'} | Vol: {claim.volume_l or '—'} L\n"
            f"Amount: ₹{claim.total_rs:,.2f}"
        )
    else:
        labels = {"driver_salary": "Driver Salary","insurance": "Insurance","service": "Service","accessories": "Accessories"}
        kind = labels.get(claim.claim_type, claim.claim_type.title())
        ack = (
            f"✅ {kind} claim saved\n"
            f"Date: {claim.claim_date}\n"
            f"Payee: {claim.station or 'N/A'}\n"
            f"Ref: {claim.reference_no or 'N/A'}\n"
            f"Amount: ₹{claim.total_rs:,.2f}"
        )

    prefs = get_user_prefs(user_id)
    p_start, p_end = current_period(prefs["cutoff_day"], prefs["tz"])
    p_total, by_type, last_txn = period_totals(user_id, p_start, p_end)
    summary = (
        f"\n—\nPeriod {p_start.strftime('%d-%b')} → {p_end.strftime('%d-%b')}"
        f"\nTotal this period: ₹{p_total:,.2f}" +
        "".join(f"\n  • {k}: ₹{v:,.2f}" for k, v in by_type.items())
    )
    if last_txn:
        summary += f"\nLast bill in period: {last_txn['claim_date']} ₹{last_txn['total_rs']:,.2f}"
    ack += summary

    try:
        send_tg(claim.telegram_chat_id, ack)
    except Exception as e:
        logger.warning("Non-fatal: telegram send failed: %r", e)

    return {"ack": ack, "chat_id": claim.telegram_chat_id}

# ---------------- Reminders ----------------
@app.post("/reminders/weekly")
def weekly_reminders():
    users = list_users()
    sent = 0
    for u in users:
        uid = u["user_id"]; chat = u["telegram_chat_id"]
        prefs = get_user_prefs(uid)
        if not prefs.get("weekly_reminder", True):
            continue
        start, end = current_period(prefs["cutoff_day"], prefs["tz"])
        total, by_type, last_txn = period_totals(uid, start, end)
        msg = (f"🗓️ Weekly reminder\nPeriod {start:%d-%b} → {end:%d-%b}\n"
               f"Total so far: ₹{total:,.2f}" +
               "".join(f"\n• {k}: ₹{v:,.2f}" for k, v in by_type.items()))
        if last_txn:
            msg += f"\nLast bill: {last_txn['claim_date']} ₹{last_txn['total_rs']:,.2f}"
        msg += "\n\nPlease capture any pending bills."
        send_tg(chat, msg); sent += 1
    return {"sent": sent}

@app.post("/reminders/precutoff")
def precutoff_reminders():
    users = list_users()
    sent = 0
    for u in users:
        uid = u["user_id"]; chat = u["telegram_chat_id"]
        prefs = get_user_prefs(uid)
        if not prefs.get("daily_pre_cutoff", True):
            continue
        tz = zoneinfo.ZoneInfo(prefs["tz"]); today = datetime.now(tz).date()
        start, end = current_period(prefs["cutoff_day"], prefs["tz"])
        if end - timedelta(days=5) <= today <= end:
            total, _, last_txn = period_totals(uid, start, end)
            last_line = (f"Last bill: {last_txn['claim_date']} ₹{last_txn['total_rs']:,.2f}"
                         if last_txn else "No bills yet this period.")
            msg = (f"⏰ Cut-off approaching ({end:%d-%b}).\n"
                   f"Total this period: ₹{total:,.2f}\n{last_line}\n"
                   f"Please capture any pending bills today.")
            send_tg(chat, msg); sent += 1
    return {"sent": sent}

# ---------------- Diagnostics ----------------
@app.get("/diag")
def diag():
    return {
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        "opencv": getattr(cv2, '__version__', None),
        "tesseract_binary": shutil.which("tesseract"),
        "tesseract_version": str(pytesseract.get_tesseract_version()) if shutil.which("tesseract") else None,
    }

# ---------------- OCR.space direct endpoint ----------------
@app.post("/extract-bill")
async def extract_bill(
    image_url: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
    telegram_chat_id: Optional[int] = Form(default=None),
):
    if not image_url and not file:
        raise HTTPException(400, "Provide image_url or file")
    try:
        text = _ocrspace_from_url(image_url) if image_url else _ocrspace_from_file(await file.read())
        f = _parse_bill_text_strict(text)
        if telegram_chat_id:
            msg = (
                "🧾 *Bill parsed*\n"
                f"Ref: `{f.get('reference_no') or '—'}`\n"
                f"Date: `{f.get('txn_date') or '—'}`\n"
                f"Total: `{f.get('total_amount') or '—'}`"
            )
            send_tg(telegram_chat_id, msg)
        return JSONResponse({"ok": True, "fields": f})
    except HTTPException:
        raise
    except Exception:
        logger.exception("extract-bill failed")
        raise HTTPException(500, "OCR/parse failed")

# ---------------- Telegram Webhook Route ----------------
@app.post("/tg/")
async def tg_webhook(request: Request):
    """
    Telegram posts updates here. We convert JSON -> Update and hand to the PTB Application.
    """
    if not _ptb_app:
        raise HTTPException(503, "Telegram bot not initialized")
    data = await request.json()
    update = Update.de_json(data, _ptb_app.bot)
    await _ptb_app.process_update(update)
    return {"ok": True}

# ---------------- Telegram Bot Logic ----------------
def build_tg_app() -> Application:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN not set")
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", tg_start))
    app.add_handler(CommandHandler("help", tg_help))
    app.add_handler(CommandHandler("health", tg_health))
    app.add_handler(CommandHandler("claimtype", tg_set_claim_type))
    app.add_handler(MessageHandler(filters.PHOTO | (filters.Document.IMAGE), tg_handle_image))
    return app

# ---- TG handlers (async) ----
async def tg_start(update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Hi! Send me a *photo or image file* of your bill.\n"
        "I’ll extract details and log your claim automatically.\n\n"
        "• /help — usage\n"
        "• /health — API health\n"
        "• /claimtype fuel|service|insurance|accessories|driver_salary — set default type (per chat)"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def tg_help(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send a bill *photo* or *image file* (jpg/png/webp). I’ll parse and save it.\n"
        "Use /claimtype to switch from fuel to others.",
        parse_mode=ParseMode.MARKDOWN,
    )

async def tg_health(update, context: ContextTypes.DEFAULT_TYPE):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{API_BASE}/health")
            await update.message.reply_text(r.text)
    except Exception as e:
        await update.message.reply_text(f"Health check failed: {e}")

async def tg_set_claim_type(update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text("Usage: /claimtype fuel|service|insurance|accessories|driver_salary")
    ct = context.args[0].strip().lower()
    valid = {"fuel", "service", "insurance", "accessories", "driver_salary"}
    if ct not in valid:
        return await update.message.reply_text(f"Invalid. Choose one of: {', '.join(sorted(valid))}")
    context.chat_data["claim_type"] = ct
    await update.message.reply_text(f"Default claim_type set to *{ct}*", parse_mode=ParseMode.MARKDOWN)

async def tg_handle_image(update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user = update.effective_user
    chat = update.effective_chat

    # Highest-res photo or image-doc
    file_id = None
    filename = "bill.jpg"
    if msg.photo:
        file_id = msg.photo[-1].file_id
    elif msg.document and (msg.document.mime_type or "").startswith("image/"):
        file_id = msg.document.file_id
        filename = msg.document.file_name or "bill.jpg"
    else:
        return await msg.reply_text("Please send a *photo* or *image file* (jpg/png/webp).", parse_mode=ParseMode.MARKDOWN)

    try:
        await msg.chat.send_action(ChatAction.UPLOAD_PHOTO)
        tg_file = await context.bot.get_file(file_id)
        image_bytes = await tg_file.download_as_bytearray()

        # 1) /extract
        async with httpx.AsyncClient(timeout=120) as client:
            files = {"file": (filename, bytes(image_bytes), "image/jpeg")}
            er = await client.post(f"{API_BASE}/extract", files=files)
            er.raise_for_status()
            fields = er.json()

        # default claim_type per chat
        claim_type = context.chat_data.get("claim_type", "fuel")

        # 2) /upload-claim
        payload = {
            "telegram_user_id": user.id,
            "telegram_chat_id": chat.id,
            "claim_type": claim_type,
            "claim_date": fields.get("txn_date"),
            "claim_time": fields.get("time"),
            "total_rs": fields.get("total_amount"),
            "station": fields.get("source"),
            "reference_no": fields.get("transaction_id"),
            "rate_rs_per_l": fields.get("rate_rs_per_l"),
            "volume_l": fields.get("volume_l"),
            "notes": "auto-uploaded from TG",
        }
        async with httpx.AsyncClient(timeout=60) as client:
            ur = await client.post(f"{API_BASE}/upload-claim", json=payload)
            if ur.status_code == 404:
                return await msg.reply_text("You’re not registered. Please link your Telegram in the app.")
            ur.raise_for_status()
            upload_ack = ur.json()

        # 3) ACK message
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

        if isinstance(upload_ack, dict) and upload_ack.get("ack"):
            parts.append("\n—\n" + upload_ack["ack"])

        await msg.reply_text("\n".join(parts), parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)

    except httpx.HTTPStatusError as he:
        detail = he.response.text
        try:
            j = he.response.json()
            if isinstance(j, dict) and "detail" in j:
                detail = j["detail"]
        except Exception:
            pass
        logger.exception("API error: %s", detail)
        await msg.reply_text(f"❌ API error: {detail}")
    except Exception as e:
        logger.exception("Unexpected TG handler error")
        await msg.reply_text(f"❌ Failed: {e}")

# ---------------- Local run ----------------
if __name__ == "__main__":
    # For local dev, this still runs the API on :8000
    # Webhook requires a public URL (Render). For local testing of the bot:
    #  - use polling in a separate script, or
    #  - tunnel (ngrok) and set WEBHOOK_URL to that.
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
