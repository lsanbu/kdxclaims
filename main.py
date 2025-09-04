# main.py  — KDxClaims API (consolidated)
# --------------------------------------
# Endpoints:
#   GET  /                  -> service ping
#   GET  /health            -> health check
#   POST /extract           -> OCR (image file) via local Tesseract (receipt_ocr.py)
#   POST /upload-claim      -> persist a claim (supports fuel & others)
#   POST /reminders/weekly  -> send weekly summaries to all users
#   POST /reminders/precutoff -> send daily reminders 5 days before cut-off
#   POST /extract-bill      -> (optional) OCR.space-based parser; requires OCRSPACE_KEY

from __future__ import annotations

import os
import re
import logging
import tempfile
import shutil
from datetime import datetime, date, timedelta
import zoneinfo
from typing import Optional, Dict, Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supabase import create_client, Client

# ---------- App setup ----------
load_dotenv()
app = FastAPI()

# CORS (tighten allow_origins to your FE domain when you deploy)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"ok": True, "service": "kdxclaims-api"}

@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}

# ---------- Supabase + Config ----------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BOT_TOKEN     = os.getenv("BOT_TOKEN")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY. Check your env vars.")

supabase: Optional[Client] = None

@app.on_event("startup")
def _startup():
    global supabase
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def _sb() -> Client:
    if supabase is None:
        raise RuntimeError("Supabase client not initialized yet")
    return supabase

# ---------- Models ----------
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

# ---------- Middleware ----------
logger = logging.getLogger("uvicorn.error")

@app.middleware("http")
async def log_errors(request, call_next):
    try:
        return await call_next(request)
    except Exception:
        logger.exception(f"Unhandled error on {request.method} {request.url.path}")
        raise

# ---------- Helpers ----------
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
    return res.data[0] if res.data else {"cutoff_day": 15, "tz": "Asia/Kolkata", "weekly_reminder": True, "daily_pre_cutoff": True}

def current_period(cutoff_day: int, tzname: str) -> tuple[date, date]:
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
        # start is the day after last period's cutoff
        start_day = min(cutoff_day + 1, 28 if start_m.month == 2 and not (start_m.year % 4 == 0 and (start_m.year % 100 != 0 or start_m.year % 400 == 0)) else 31)
        start = start_m.replace(day=start_day)
        end = now.replace(day=cutoff_day)
    else:
        start = now.replace(day=cutoff_day + 1)
        nm = month_add(now.replace(day=1), 1)
        end = nm.replace(day=cutoff_day)
    return start, end

def period_totals(user_id: str, start: date, end: date) -> tuple[float, Dict[str, float], Optional[Dict[str, Any]]]:
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
        print("TG send skipped: BOT_TOKEN missing")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=10)
        if r.status_code != 200:
            print("TG send error:", r.status_code, r.text)
    except Exception as e:
        print("TG send exception:", repr(e))

def list_users():
    res = _sb().table("user_telegram").select("user_id, telegram_chat_id").execute()
    return res.data or []

# ---------- OCR (local Tesseract via receipt_ocr.py) ----------
from dataclasses import asdict
from receipt_ocr import parse_receipt_image

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    """OCR a single image and return normalized fields."""
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename or '')[-1] or ".jpg", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    res = parse_receipt_image(tmp_path)
    return asdict(res)

# ---------- Claims API ----------
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

    # Build ACK
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
        labels = {
            "driver_salary": "Driver Salary",
            "insurance": "Insurance",
            "service": "Service",
            "accessories": "Accessories",
        }
        kind = labels.get(claim.claim_type, claim.claim_type.title())
        ack = (
            f"✅ {kind} claim saved\n"
            f"Date: {claim.claim_date}\n"
            f"Payee: {claim.station or 'N/A'}\n"
            f"Ref: {claim.reference_no or 'N/A'}\n"
            f"Amount: ₹{claim.total_rs:,.2f}"
        )

    # Period summary
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

    # Send Telegram (best-effort)
    try:
        send_tg(claim.telegram_chat_id, ack)
    except Exception as e:
        logger.warning("Non-fatal: telegram send failed: %s", repr(e))

    return {"ack": ack, "chat_id": claim.telegram_chat_id}

# ---------- Reminders ----------
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

# ---------- Optional: OCR.space fallback ----------
OCRSPACE_KEY = os.getenv("OCRSPACE_KEY")
OCRSPACE_URL = "https://api.ocr.space/parse/image"

date_patterns = [
    r"\b(\d{4}[-/]\d{2}[-/]\d{2})\b",
    r"\b(\d{2}[-/]\d{2}[-/]\d{4})\b",
    r"\b(\d{1,2}\s*[A-Za-z]{3,9}\s*\d{2,4})\b",
]
amount_patterns = [
    r"(?:Total(?:\s*Amount)?|Amount\s*Payable|Grand\s*Total)[^\d]{0,10}([\ ₹₹Rs\.]*\d[\d,]*\.?\d{0,2})",
    r"\b(?:INR|Rs\.?|₹)\s*([\d,]*\.?\d{1,2})\b",
]
ref_patterns = [
    r"(?:Ref(?:erence)?|Invoice|Bill|Receipt|Txn|Transaction|Voucher)\s*(?:No\.?|#|:)?\s*([A-Z0-9\-\/]{5,})",
]

def _norm_amount(s: str) -> Optional[float]:
    if not s: 
        return None
    s = s.replace("₹", "").replace("Rs.", "").replace("Rs", "").replace(" ", "").replace(",", "")
    try:
        return float(s)
    except:
        return None

def _find_first(patterns, text, flags=re.IGNORECASE):
    for p in patterns:
        m = re.search(p, text, flags)
        if m:
            return m.group(1).strip()
    return None

def _ocrspace_from_url(image_url: str) -> str:
    if not OCRSPACE_KEY:
        raise HTTPException(500, "OCRSPACE_KEY not configured")
    resp = requests.post(
        OCRSPACE_URL,
        data={"apikey": OCRSPACE_KEY, "url": image_url, "OCREngine": 2, "language": "eng"},
        timeout=30,
    )
    j = resp.json()
    if not j.get("IsErroredOnProcessing") and j.get("ParsedResults"):
        return " \n".join(pr.get("ParsedText","") for pr in j["ParsedResults"])
    raise HTTPException(502, f"OCR failed: {j.get('ErrorMessage')}")

def _ocrspace_from_file(file_bytes: bytes) -> str:
    if not OCRSPACE_KEY:
        raise HTTPException(500, "OCRSPACE_KEY not configured")
    resp = requests.post(
        OCRSPACE_URL,
        files={"file": ("bill.jpg", file_bytes)},
        data={"apikey": OCRSPACE_KEY, "OCREngine": 2, "language": "eng"},
        timeout=30,
    )
    j = resp.json()
    if not j.get("IsErroredOnProcessing") and j.get("ParsedResults"):
        return " \n".join(pr.get("ParsedText","") for pr in j["ParsedResults"])
    raise HTTPException(502, f"OCR failed: {j.get('ErrorMessage')}")

def _parse_bill_text(text: str):
    t = re.sub(r"[ \t]+", " ", text)
    txn_date  = _find_first(date_patterns, t)
    ref_no    = _find_first(ref_patterns, t)
    amt_raw   = _find_first(amount_patterns, t)
    total_amt = _norm_amount(amt_raw)
    return {
        "txn_date": txn_date,
        "reference_no": ref_no,
        "total_amount": total_amt,
        "raw_amount_match": amt_raw,
        "preview": t[:600],
    }

@app.post("/extract-bill")
async def extract_bill(
    image_url: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
    telegram_chat_id: Optional[int] = Form(default=None),
):
    """OCR.space fallback (URL or file)."""
    if not image_url and not file:
        raise HTTPException(400, "Provide image_url or file")
    try:
        text = _ocrspace_from_url(image_url) if image_url else _ocrspace_from_file(await file.read())
        result = _parse_bill_text(text)
        if telegram_chat_id:
            msg = (
                "🧾 *Bill parsed*\n"
                f"Ref: `{result.get('reference_no') or '—'}`\n"
                f"Date: `{result.get('txn_date') or '—'}`\n"
                f"Total: `{result.get('total_amount') or result.get('raw_amount_match') or '—'}`"
            )
            send_tg(telegram_chat_id, msg)
        return JSONResponse({"ok": True, "fields": result})
    except HTTPException:
        raise
    except Exception:
        logger.exception("extract-bill failed")
        raise HTTPException(500, "OCR/parse failed")

# ---------- Local run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
