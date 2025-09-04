from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from datetime import datetime, date, timedelta
import zoneinfo
import requests

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# Init Supabase client
app = FastAPI()
load_dotenv()

@app.get("/")
def root():
    return {"ok": True, "service": "kdxclaims-api"}

@app.get("/health")
def health():
    return {"ok": True}

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your domain(s)
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY. Check your .env and CWD.")

supabase: Client | None = None

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup():
    global supabase
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)  # 👈 set global here

def _sb() -> Client:
    if supabase is None:
        raise RuntimeError("Supabase client not initialized yet")
    return supabase

# ---- Pydantic model for request ----
class ClaimIn(BaseModel):
    telegram_user_id: int
    telegram_chat_id: int
    claim_type: str   # "fuel", "driver_salary", "insurance", "service", "accessories"
    claim_date: str   # YYYY-MM-DD
    claim_time: str | None = None
    total_rs: float
    station: str | None = None          # fuel station / workshop / insurer / shop
    reference_no: str | None = None     # invoice no / policy no / voucher no
    rate_rs_per_l: float | None = None  # only for fuel
    volume_l: float | None = None       # only for fuel
    notes: str | None = None

# ---- Helper: get app_users.id from telegram_user_id ----
def get_user_id(telegram_user_id: int) -> str:
    result = (
        supabase.table("user_telegram")
        .select("user_id")
        .eq("telegram_user_id", telegram_user_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="User not registered")
    return result.data[0]["user_id"]

def get_user_prefs(user_id: str):
    res = supabase.table("user_prefs").select("*").eq("user_id", user_id).execute()
    prefs = res.data[0] if res.data else {"cutoff_day": 15, "tz": "Asia/Kolkata"}
    return prefs

def current_period(cutoff_day: int, tzname: str):
    tz = zoneinfo.ZoneInfo(tzname)
    now = datetime.now(tz).date()

    def month_add(d: date, months: int):
        y = d.year + (d.month - 1 + months) // 12
        m = (d.month - 1 + months) % 12 + 1
        day = min(d.day, [31,29 if y%4==0 and (y%100!=0 or y%400==0) else 28,31,30,31,30,31,31,30,31,30,31][m-1])
        return date(y,m,day)

    if now.day <= cutoff_day:
        start_m = month_add(now.replace(day=1), -1)
        start = start_m.replace(day=min(cutoff_day+1, 28 if start_m.month==2 and not (start_m.year%4==0 and (start_m.year%100!=0 or start_m.year%400==0)) else 31))
        end = now.replace(day=cutoff_day)
    else:
        start = now.replace(day=cutoff_day+1)
        nm = month_add(now.replace(day=1), 1)
        end = nm.replace(day=cutoff_day)
    return start, end  # inclusive window

def period_totals(user_id: str, start: date, end: date):
    q = (supabase.table("claims")
         .select("claim_type,total_rs,claim_date")
         .eq("user_id", user_id)
         .gte("claim_date", str(start))
         .lte("claim_date", str(end))
         .execute())
    rows = q.data or []
    total = sum(r["total_rs"] for r in rows)
    by_type = {}
    for r in rows:
        by_type[r["claim_type"]] = by_type.get(r["claim_type"], 0) + r["total_rs"]
    last_txn = max(rows, key=lambda r: r["claim_date"]) if rows else None
    return total, by_type, last_txn

# ---- API endpoint ----
@app.post("/upload-claim")
def upload_claim(claim: ClaimIn):
    # Resolve to internal user_id
    user_id = get_user_id(claim.telegram_user_id)

    # Insert claim
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
    }

    insert_res = supabase.table("claims").insert(row).execute()

    if not insert_res.data:
        raise HTTPException(status_code=500, detail="Insert failed")

    # Build acknowledgement
    valid_claim_types = {"fuel", "driver_salary", "insurance", "service", "accessories"}
    if claim.claim_type not in valid_claim_types:
        raise HTTPException(400, detail=f"Invalid claim_type. Must be one of {valid_claim_types}")

    if claim.claim_type == "fuel":
        ack = (
            f"✅ Fuel claim saved\n"
            f"Date {claim.claim_date} {claim.claim_time or ''}\n"
            f"Station: {claim.station}\n"
            f"Ref: {claim.reference_no}\n"
            f"Rate: {claim.rate_rs_per_l} | Vol: {claim.volume_l} L\n"
            f"Amount: ₹{claim.total_rs}"
        )
    else:
        labels = {
            "driver_salary": "Driver Salary",
            "insurance": "Insurance",
            "service": "Service",
            "accessories": "Accessories",
        }
        labels = {"fuel":"Fuel","driver_salary":"Driver Salary","insurance":"Insurance","service":"Service","accessories":"Accessories"}
        kind = labels.get(claim.claim_type, claim.claim_type.title())
        
        ack = (
            f"✅ {kind} claim saved\n"
            f"Date: {claim.claim_date}\n"
            f"Payee: {claim.station or 'N/A'}\n"
            f"Ref: {claim.reference_no or 'N/A'}\n"
            f"Amount: ₹{claim.total_rs}"
        )

    # after insert_res success
    prefs = get_user_prefs(user_id)
    p_start, p_end = current_period(prefs["cutoff_day"], prefs["tz"])
    p_total, by_type, last_txn = period_totals(user_id, p_start, p_end)

    summary = (
        f"\n—\nPeriod {p_start.strftime('%d-%b')} → {p_end.strftime('%d-%b')}"
        f"\nTotal this period: ₹{p_total:,.2f}"
        + "".join(f"\n  • {k}: ₹{v:,.2f}" for k,v in by_type.items())
    )
    if last_txn:
        summary += f"\nLast bill in period: {last_txn['claim_date']} ₹{last_txn['total_rs']:,.2f}"

    ack += summary

    # ✅ send the ack to Telegram
    if BOT_TOKEN:  # only if token is configured
        send_tg(claim.telegram_chat_id, ack)

    return {"ack": ack, "chat_id": claim.telegram_chat_id}

def send_tg(chat_id: int, text: str):
    if not BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id": chat_id,
            "text": text
            # "parse_mode": "HTML",  # enable if you format messages in HTML
            # "disable_web_page_preview": True
        }, timeout=10)
        if r.status_code != 200:
            # helpful log if something is wrong (401 bad token, 400 chat not found, etc.)
            print("TG send error:", r.status_code, r.text)
    except Exception as e:
        print("TG send exception:", e)

def list_users():
    res = supabase.table("user_telegram").select("user_id, telegram_chat_id").execute()
    return res.data or []

@app.post("/reminders/weekly")
def weekly_reminders():
    users = list_users()
    sent = 0
    for u in users:
        uid = u["user_id"]; chat = u["telegram_chat_id"]
        prefs = get_user_prefs(uid)
        if not prefs.get("weekly_reminder", True): continue
        start, end = current_period(prefs["cutoff_day"], prefs["tz"])
        total, by_type, last_txn = period_totals(uid, start, end)
        msg = (f"🗓️ Weekly reminder\nPeriod {start:%d-%b} → {end:%d-%b}\n"
               f"Total so far: ₹{total:,.2f}" +
               "".join(f"\n• {k}: ₹{v:,.2f}" for k,v in by_type.items()))
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
        if not prefs.get("daily_pre_cutoff", True): continue
        tz = zoneinfo.ZoneInfo(prefs["tz"]); today = datetime.now(tz).date()
        start, end = current_period(prefs["cutoff_day"], prefs["tz"])
        # window: cutoff_day-5 ... cutoff_day inclusive
        if end - timedelta(days=5) <= today <= end:
            total, _, last_txn = period_totals(uid, start, end)
            last_line = (f"Last bill: {last_txn['claim_date']} ₹{last_txn['total_rs']:,.2f}"
                         if last_txn else "No bills yet this period.")
            msg = (f"⏰ Cut-off approaching ({end:%d-%b}).\n"
                   f"Total this period: ₹{total:,.2f}\n{last_line}\n"
                   f"Please capture any pending bills today.")
            send_tg(chat, msg); sent += 1
    return {"sent": sent}
