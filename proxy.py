"""HTTP proxy that intercepts Ollama /api/chat and /api/generate streams."""
import json
import httpx
import time
from datetime import datetime
from typing import AsyncIterator


OLLAMA_URL = "http://127.0.0.1:11435"
TIMEOUT = httpx.Timeout(60.0, connect=10.0)


async def proxy_chat(body: dict, client_ip: str) -> AsyncIterator[bytes]:
    """
    Forward a /api/chat request to Ollama, yield SSE tokens.
    Measures TTFT on first token, returns (model, prompt_tokens, response_tokens, ttft_ms, total_ms, status).
    """
    model = body.get("model", "unknown")
    # Estimate prompt tokens (rough: count chars / 4)
    prompt_tokens = body.get("num_predict", 0) or len(str(body.get("messages", ""))) // 4

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        t0 = time.perf_counter()
        ttft_ms = None
        response_tokens = 0

        try:
            async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=body) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str or data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000

                    # Count response tokens from chunk
                    if "message" in chunk:
                        content = chunk["message"].get("content", "")
                        response_tokens += len(content) // 4  # rough estimate

                    yield line.encode()

                total_ms = (time.perf_counter() - t0) * 1000
                tps = (response_tokens / total_ms * 1000) if total_ms > 0 else 0

                return model, prompt_tokens, response_tokens, ttft_ms, total_ms, "ok"

        except Exception as e:
            total_ms = (time.perf_counter() - t0) * 1000
            yield f'{{"error": "{e}"}}\n'.encode()
            return model, prompt_tokens, response_tokens, ttft_ms, total_ms, "error"


async def proxy_generate(body: dict, client_ip: str) -> AsyncIterator[bytes]:
    """
    Forward a /api/generate request to Ollama, yield SSE tokens.
    """
    model = body.get("model", "unknown")
    prompt_tokens = len(str(body.get("prompt", ""))) // 4

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        t0 = time.perf_counter()
        ttft_ms = None
        response_tokens = 0

        try:
            async with client.stream("POST", f"{OLLAMA_URL}/api/generate", json=body) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str or data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000

                    if "response" in chunk:
                        response_tokens += len(chunk["response"]) // 4

                    yield line.encode()

                total_ms = (time.perf_counter() - t0) * 1000
                tps = (response_tokens / total_ms * 1000) if total_ms > 0 else 0

                return model, prompt_tokens, response_tokens, ttft_ms, total_ms, "ok"

        except Exception as e:
            total_ms = (time.perf_counter() - t0) * 1000
            yield f'{{"error": "{e}"}}\n'.encode()
            return model, prompt_tokens, response_tokens, ttft_ms, total_ms, "error"
