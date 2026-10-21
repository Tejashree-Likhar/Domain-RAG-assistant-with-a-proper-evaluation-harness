"""FastAPI service. Config via env vars (see Config.from_env). Run: uvicorn ragkit.api:app"""
from __future__ import annotations
import logging, statistics, threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .pipeline import build_pipeline, Config

logging.basicConfig(level=logging.INFO, format="%(message)s")
STATE: dict = {"lat": [], "n": 0, "refused": 0}
LOCK = threading.Lock()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int | None = Field(default=None, ge=1, le=20)


@asynccontextmanager
async def lifespan(app):
    STATE["cfg"] = Config.from_env()
    STATE["pipe"] = build_pipeline(STATE["cfg"])
    yield


app = FastAPI(title="Domain RAG assistant", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "config": {k: v for k, v in vars(STATE["cfg"]).items()}}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        out = STATE["pipe"].answer(req.question, req.k)
    except Exception as e:                      # surface generator/API failures cleanly
        raise HTTPException(502, f"pipeline error: {type(e).__name__}: {e}")
    with LOCK:
        STATE["n"] += 1; STATE["refused"] += int(out["refused"])
        STATE["lat"] = (STATE["lat"] + [out["latency_ms"]["total_ms"]])[-1000:]
    return out


@app.get("/stats")
def stats():
    lat = sorted(STATE["lat"])
    if not lat:
        return {"requests": 0}
    q = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]
    return {"requests": STATE["n"], "refusal_rate": STATE["refused"] / STATE["n"],
            "latency_ms": {"p50": q(0.5), "p95": q(0.95), "mean": statistics.mean(lat)}}
