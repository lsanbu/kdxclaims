"""
KDxClaims – Receipt OCR & Field Parser
- Extracts: transaction_id, txn_date, time, total_amount
- Also returns: source, rate, volume, raw_text for audit
"""

import re
import cv2
import pytesseract
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import statistics
from calendar import month_abbr

# ------------- CONFIG (Windows users: set your Tesseract path if needed) -------------
# Example: pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# -------------------------------------------------------------------------------------

DATE_PATTERNS = [
    r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",     # 01/09/25, 13-09-2024
    r"\b(\d{1,2}-[A-Za-z]{3}-\d{4})\b",         # 14-Jul-2025
    r"\b([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})\b",  # Jul 14, 2025
]

TIME_PATTERNS = [
    r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b"           # 19:56 or 17:50:49
]

AMOUNT_PATTERNS = [
    r"(?:Amount|Amt|Total|Preset\s*Value)\s*[:=]?\s*₹?\s*([0-9]+(?:\.[0-9]{1,2})?)\b",
    r"\b₹\s*([0-9]+(?:\.[0-9]{1,2})?)\b",
    r"\bRs\.?\s*([0-9]+(?:\.[0-9]{1,2})?)\b",
    r"\b([0-9]{2,6}\.[0-9]{2})\b"               # last resort decimal catch
]

TRANS_ID_HINTS = [
    r"(?:Receipt\s*No\.?|Rcpt\s*No\.?)\s*[:=]?\s*([A-Za-z0-9-]+)",
    r"(?:TXN\s*NO|Txn\s*Id|TXN\s*Id|Transaction\s*Id)\s*[:=]?\s*([A-Za-z0-9-]+)",
    r"(?:Invoice\s*No\.?)\s*[:=]?\s*([A-Za-z0-9-]+)",
    r"(?:FCC\s*ID)\s*[:=]?\s*([A-Za-z0-9-]+)",
    r"(?:No\.?\s*P)\s*([A-Za-z0-9-]+)",  # HP handwritten slips
]

RATE_PATTERNS = [
    r"(?:Rate|Rate\(Rs/?L\)|Rate/Ltr\.?)\s*[:=]?\s*([0-9]+(?:\.[0-9]{1,2})?)"
]

VOL_PATTERNS = [
    r"(?:Volume|Vol|Qty)\s*(?:\(|L|Ltr|Litres|Ltrs)?\.?\s*[:=]?\s*0*([0-9]+(?:\.[0-9]{1,3})?)"
]

# ----------------------- Helpers -----------------------

def _find_all(patterns, text, flags=re.IGNORECASE):
    vals = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags):
            vals.append(m.group(1).strip())
    return vals

def _pick_numeric(nums):
    vals = []
    for n in nums:
        try:
            vals.append(float(n.lstrip("0") or "0"))
        except:
            continue
    if not vals:
        return None
    return round(statistics.median(vals), 2)

def _normalize_date_any(cand: str) -> str | None:
    cand = cand.strip().replace("IUL", "JUL").replace("O", "0")
    m = re.match(r"(\d{1,2})[-/ ]([A-Za-z]{3}|\d{1,2})[-/ ](\d{2,4})", cand)
    if m:
        d, mth, y = m.groups()
        try:
            d_i = int(d)
            if d_i > 31 and d.startswith("4"):
                d = "1" + d[1:]  # fix 44 -> 14
            if not mth.isdigit():
                mth_num = [i for i,a in enumerate(month_abbr) if a and a.upper()==mth[:3].upper()]
                if mth_num:
                    mth = f"{mth_num[0]:02d}"
            if len(y) == 2:
                y = ("20" if int(y) < 50 else "19") + y
            return datetime.strptime(f"{d}-{mth}-{y}", "%d-%m-%Y").strftime("%Y-%m-%d")
        except:
            pass
    for fmt in ("%d/%m/%y","%d/%m/%Y","%d-%m-%Y","%d-%m-%y","%d-%b-%Y","%d-%b-%y","%b %d, %Y"):
        try:
            return datetime.strptime(cand, fmt).strftime("%Y-%m-%d")
        except:
            continue
    return None

def _best_date(text: str) -> str | None:
    for c in _find_all(DATE_PATTERNS, text):
        norm = _normalize_date_any(c)
        if norm:
            return norm
    return None

def _best_time(text: str) -> str | None:
    cands = _find_all(TIME_PATTERNS, text)
    return sorted(cands, key=lambda s: -len(s))[0] if cands else None

def _best_amount(text: str) -> float | None:
    return _pick_numeric(_find_all(AMOUNT_PATTERNS, text))

def _best_rate(text: str) -> tuple[float | None, list[float]]:
    cands = _find_all(RATE_PATTERNS, text)
    vals = []
    for n in cands:
        try:
            vals.append(float(n.lstrip("0") or "0"))
        except:
            pass
    val = _pick_numeric(cands)
    if val is not None and not (75.0 <= val <= 150.0):
        val = None
    return val, vals

def _best_volume(text: str) -> float | None:
    return _pick_numeric(_find_all(VOL_PATTERNS, text))

def _compose_txn_id(text: str) -> str | None:
    ids = []
    for pat in TRANS_ID_HINTS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            val = m.group(1).strip().rstrip(".:,;")
            if val and val not in ids:
                ids.append(val)
    return "; ".join(ids) if ids else None

def _best_source(text: str) -> str | None:
    mapping = [
        (r"Indian\s*Oil|IndianOil|IOC", "IndianOil"),
        (r"HP\b|Hindustan\s*Petroleum", "HP"),
        (r"Bharat\s*Petroleum|BPCL", "BPCL"),
        (r"Vasanth\s*Enterpr", "Vasanth Enterprises"),
        (r"AG\s*Agency", "AG Agency"),
        (r"Tharun\s*Enterprises", "Tharun Enterprises"),
        (r"S\.?\s*T\.?\s*Arasu", "S. T. Arasu & Co."),
        (r"RAM\s*FILLING\s*STATION", "RAM Filling Station"),
    ]
    for pat, label in mapping:
        if re.search(pat, text, re.IGNORECASE):
            return label
    return None

# ----------------------- OCR Core -----------------------

def _preprocess_for_ocr(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )
    return thr

def _run_tesseract(img, psm=6):
    config = f"--psm {psm} -l eng"
    return pytesseract.image_to_string(img, config=config)

# ----------------------- Public API -----------------------

@dataclass
class OCRResult:
    transaction_id: Optional[str]
    txn_date: Optional[str]
    time: Optional[str]
    total_amount: Optional[float]
    source: Optional[str]
    rate_rs_per_l: Optional[float]
    volume_l: Optional[float]
    raw_text: str

def _closest_to(target: float, candidates: list[float]) -> float | None:
    return min(candidates, key=lambda x: abs(x - target)) if candidates else None

def _reconcile_rate(primary_rate: float | None,
                    rate_candidates: list[float],
                    amount: float | None,
                    volume: float | None) -> float | None:
    if amount is None or volume is None or volume == 0:
        return primary_rate
    implied = round(amount / volume, 2)
    plausible = [r for r in rate_candidates if 75.0 <= r <= 150.0]
    if primary_rate and 75.0 <= primary_rate <= 150.0 and primary_rate not in plausible:
        plausible.append(primary_rate)
    chosen = _closest_to(implied, plausible)
    if chosen is None:
        return implied
    return chosen if abs(chosen - implied) <= 0.75 else implied

def parse_receipt_image(image_path: str) -> OCRResult:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(image_path)

    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    text1 = _run_tesseract(img, psm=6)
    prep = _preprocess_for_ocr(img)
    text2 = _run_tesseract(prep, psm=6)
    text3 = _run_tesseract(prep, psm=4)

    text = "\n".join([text1, text2, text3])
    text_clean = re.sub(r"[^\S\r\n]+", " ", text)

    txn_id   = _compose_txn_id(text_clean)
    txn_date = _best_date(text_clean)
    time_raw = _best_time(text_clean)
    amount   = _best_amount(text_clean)
    rate_med, rate_all = _best_rate(text_clean)
    volume   = _best_volume(text_clean)
    source   = _best_source(text_clean)

    if amount is None and rate_med and volume:
        amount = round(rate_med * volume, 2)

    rate = _reconcile_rate(rate_med, rate_all, amount, volume)

    return OCRResult(
        transaction_id=txn_id,
        txn_date=txn_date,
        time=time_raw,
        total_amount=amount,
        source=source,
        rate_rs_per_l=rate,
        volume_l=volume,
        raw_text=text_clean.strip()
    )

def parse_folder_to_rows(folder: str) -> Dict[str, Dict[str, Any]]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    out = {}
    for f in Path(folder).glob("*"):
        if f.suffix.lower() in exts:
            res = parse_receipt_image(str(f))
            out[f.name] = asdict(res)
    return out

if __name__ == "__main__":
    import json, sys
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    if Path(target).is_dir():
        rows = parse_folder_to_rows(target)
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        res = parse_receipt_image(target)
        print(json.dumps(asdict(res), indent=2, ensure_ascii=False))
