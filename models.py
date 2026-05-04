"""Pydantic response models for API endpoints."""
from pydantic import BaseModel
from typing import Optional


class StatsResponse(BaseModel):
    total_requests: int
    total_tokens: int
    avg_ttft_ms: float
    avg_tps: float


class RequestRow(BaseModel):
    id: int
    timestamp: str
    model: str
    prompt_tokens: int
    response_tokens: int
    total_tokens: int
    ttft_ms: Optional[float]
    total_ms: Optional[float]
    tps: Optional[float]
    client_ip: Optional[str]
    status: str


class HealthRow(BaseModel):
    id: int
    timestamp: str
    gpu_index: int
    gpu_name: Optional[str]
    gpu_temp: Optional[float]
    gpu_util: Optional[float]
    gpu_mem_used: Optional[float]
    gpu_mem_total: Optional[float]
    gpu_power_draw: Optional[float]
    cpu_util: Optional[float]
    mem_used: Optional[float]
    mem_total: Optional[float]
    disk_used: Optional[float]
    disk_total: Optional[float]
    ollama_model: Optional[str]
    ollama_model_size: Optional[float]


class ModelInfo(BaseModel):
    name: str
    size: Optional[float] = None
    modified: Optional[str] = None


class HourlyVolume(BaseModel):
    bucket: str
    count: int


class ModelDistEntry(BaseModel):
    model: str
    count: int


class DailyTokens(BaseModel):
    day: str
    prompt: int
    response: int


class SSEPayload(BaseModel):
    type: str
    data: dict
