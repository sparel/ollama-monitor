"""
Ollama Monitor — FastAPI app.
Proxies requests to Ollama on :11434, logs metrics to SQLite,
serves dashboard and API endpoints.
"""
import asyncio
import json
import time
import threading
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse

import db
import health
from models import (
    StatsResponse,
    RequestRow,
    HealthRow,
    ModelInfo,
    HourlyVolume,
    ModelDistEntry,
    DailyTokens,
)

# ── config ──────────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://127.0.0.1:11434"
PROXY_PORT = 11436
HEALTH_INTERVAL = 15  # seconds

app = FastAPI(title="Ollama Monitor")

# ── health poller ─────────────────────────────────────────────────────────────
def _poll_health():
    models = health.get_ollama_models()
    ollama_model = models[0]["name"] if models else None
    ollama_model_size = models[0].get("size", 0) / (1024**3) if models and "size" in models[0] else None
    h = health.get_system_health(ollama_model, ollama_model_size)
    db.insert_health(
        gpu_name=h["gpu_name"],
        gpu_temp=h["gpu_temp"] or 0,
        gpu_util=h["gpu_util"] or 0,
        gpu_mem_used=h["gpu_mem_used"] or 0,
        gpu_mem_total=h["gpu_mem_total"] or 0,
        gpu_power_draw=h["gpu_power_draw"] or 0,
        cpu_util=h["cpu_util"] or 0,
        mem_used=h["mem_used"] or 0,
        mem_total=h["mem_total"] or 0,
        disk_used=h["disk_used"] or 0,
        disk_total=h["disk_total"] or 0,
        ollama_model=h["ollama_model"],
        ollama_model_size=h["ollama_model_size"],
    )

def _health_loop():
    while True:
        try:
            _poll_health()
        except Exception as e:
            print(f"[health] poll error: {e}")
        time.sleep(HEALTH_INTERVAL)

_health_thread = threading.Thread(target=_health_loop, daemon=True)
_health_thread.start()

# ── SSE broadcaster ────────────────────────────────────────────────────────────
_event_queue: asyncio.Queue = asyncio.Queue()


# ── proxy: stream + log ────────────────────────────────────────────────────────
async def _proxy_stream(request: Request, path: str):
    """
    Forward to Ollama, intercept SSE, measure TTFT, log to DB.
    DB insert happens AFTER the stream finishes (client already has full response).
    Uses Ollama's own eval_count / prompt_eval_count for accurate token counts.
    """
    body = await request.json()
    model = body.get("model", "unknown")
    client_ip = request.client.host if request.client else "unknown"
    t0 = time.perf_counter()
    ttft_ms = None
    prompt_tokens = 0
    response_tokens = 0
    total_ms = None

    async def generator():
        nonlocal ttft_ms, prompt_tokens, response_tokens, total_ms
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=15.0),
            limits=httpx.Limits(max_keepalive_connections=1),
        ) as client:
            async with client.stream("POST", f"{OLLAMA_URL}{path}", json=body) as resp:
                async for raw_line in resp.aiter_lines():
                    line = raw_line.rstrip()
                    if not line or line == "[DONE]":
                        yield line
                        continue

                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str and data_str != "[DONE]":
                            try:
                                chunk = json.loads(data_str)
                                # TTFT: first chunk with actual response content
                                if ttft_ms is None and chunk.get("response"):
                                    ttft_ms = (time.perf_counter() - t0) * 1000

                                # done=true chunk has final metrics
                                if chunk.get("done"):
                                    prompt_tokens = chunk.get("prompt_eval_count", 0)
                                    response_tokens = chunk.get("eval_count", 0)
                                    total_ms = (time.perf_counter() - t0) * 1000
                            except json.JSONDecodeError:
                                pass

                    yield line

    async def wrapped():
        nonlocal prompt_tokens, response_tokens, total_ms, ttft_ms
        lines_seen = 0
        async for line in generator():
            lines_seen += 1
            yield line
        Path('/tmp/wrapped_done').write_text(
            f'lines={lines_seen} prompt={prompt_tokens} resp={response_tokens} ttft={ttft_ms}ms total={total_ms}ms\n'
        )
        # Stream consumed — now log to DB
        try:
            tps = (response_tokens / total_ms * 1000) if total_ms and total_ms > 0 else 0
            db.insert_request(
                model=model,
                prompt_tokens=prompt_tokens,
                response_tokens=max(response_tokens, 1),
                ttft_ms=ttft_ms,
                total_ms=total_ms,
                client_ip=client_ip,
                status="ok",
            )
            await _event_queue.put({
                "type": "request",
                "model": model,
                "timestamp": datetime.utcnow().isoformat(),
            })
        except Exception as e:
            print(f"[db] insert error: {e}")

    return StreamingResponse(wrapped(), media_type="text/event-stream")


# ── API routes ─────────────────────────────────────────────────────────────────
@app.get("/api/stats", response_model=StatsResponse)
def stats():
    return db.get_stats_24h()


@app.get("/api/requests", response_model=list[RequestRow])
def list_requests(limit: int = 100):
    return db.get_recent_requests(limit)


@app.get("/api/health", response_model=HealthRow)
def latest_health():
    h = db.get_latest_health()
    if not h:
        return {"id": 0, "timestamp": "", "gpu_index": 0}
    return h


@app.get("/api/health/history")
def health_history(hours: int = 24):
    return db.get_health_history(hours)


@app.get("/api/models", response_model=list[ModelInfo])
def list_models():
    models = health.get_ollama_models()
    return [
        ModelInfo(
            name=m.get("name", ""),
            size=(m.get("size", 0) / (1024**3)) if "size" in m else None,
            modified=m.get("modified_at"),
        )
        for m in models
    ]


@app.get("/api/requests/stream")
async def requests_stream():
    async def gen():
        while True:
            payload = await _event_queue.get()
            yield {"event": "update", "data": json.dumps(payload)}
    return EventSourceResponse(gen())


@app.get("/api/volume", response_model=list[HourlyVolume])
def volume(hours: int = 24):
    return db.get_hourly_volume(hours)


@app.get("/api/distribution", response_model=list[ModelDistEntry])
def distribution(hours: int = 24):
    return db.get_model_distribution(hours)


@app.get("/api/tokens", response_model=list[DailyTokens])
def tokens(days: int = 7):
    return db.get_daily_tokens(days)


# ── dashboard (before catch-all) ────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (Path(__file__).parent / "dashboard.html").read_text()


# ── proxy routes ────────────────────────────────────────────────────────────────
@app.post("/api/chat")
async def proxy_chat(request: Request):
    return await _proxy_stream(request, "/api/chat")


@app.post("/api/generate")
async def proxy_generate(request: Request):
    return await _proxy_stream(request, "/api/generate")


# ── pass-through ───────────────────────────────────────────────────────────────
@app.api_route("/{path:path}", methods=["GET", "POST", "DELETE", "PUT", "PATCH"])
async def ollama_proxy(path: str, request: Request):
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        url = f"{OLLAMA_URL}/{path}"
        body = await request.body()
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("host", "content-length")}
        resp = await client.request(request.method, url, content=body, headers=headers)
        return Response(content=resp.content, status_code=resp.status_code,
                       headers=dict(resp.headers))


# ── startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    print(f"[ollama-monitor] Starting on :{PROXY_PORT}")
    print(f"[ollama-monitor] Proxying to Ollama at {OLLAMA_URL}")
    try:
        _poll_health()
    except Exception as e:
        print(f"[health] initial poll failed: {e}")
