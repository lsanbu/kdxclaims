# tg_webhook_asgi.py — ASGI entry to forward updates to PTB Application
import os
import asyncio
from fastapi import FastAPI, Request
from bot import build_app  # imports Application & handlers

app = FastAPI()
_ptb_app = None

@app.on_event("startup")
async def _startup():
    global _ptb_app
    _ptb_app = build_app()
    await _ptb_app.initialize()
    await _ptb_app.start()

@app.post("/tg/webhook")
async def tg_webhook(request: Request):
    # PTB provides a convenience to process updates:
    data = await request.json()
    await _ptb_app.update_queue.put(data)  # enqueue raw update dict
    return {"ok": True}
