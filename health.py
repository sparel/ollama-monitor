"""GPU + system health polling via nvidia-smi and psutil."""
import re
import subprocess
import psutil
from typing import Optional


def _parse_nvidia_smi() -> dict:
    """Parse nvidia-smi for GPU metrics."""
    result = {
        "gpu_name": "NVIDIA GB10",
        "gpu_temp": None,
        "gpu_util": None,
        "gpu_mem_used": None,
        "gpu_mem_total": None,
        "gpu_power_draw": None,
    }
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,temperature.gpu,utilization.gpu,"
                "memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            parts = [p.strip() for p in proc.stdout.strip().split(",")]
            # Handle [N/A] for memory
            result["gpu_name"] = parts[0] if parts[0] else "NVIDIA GB10"
            result["gpu_temp"] = float(parts[1]) if parts[1] not in ("[N/A]", "") else None
            result["gpu_util"] = float(parts[2].replace("%", "")) if parts[2] not in ("[N/A]", "") else None
            result["gpu_mem_used"] = float(parts[3]) if parts[3] not in ("[N/A]", "") else None
            result["gpu_mem_total"] = float(parts[4]) if parts[4] not in ("[N/A]", "") else None
            result["gpu_power_draw"] = float(parts[5]) if parts[5] not in ("[N/A]", "") else None
    except Exception:
        pass
    return result


def get_system_health(ollama_model: Optional[str] = None, ollama_model_size: Optional[float] = None) -> dict:
    """Collect all health metrics."""
    gpu = _parse_nvidia_smi()
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    cpu_util = psutil.cpu_percent(interval=0.5)

    return {
        "gpu_name": gpu["gpu_name"],
        "gpu_temp": gpu["gpu_temp"],
        "gpu_util": gpu["gpu_util"],
        "gpu_mem_used": gpu["gpu_mem_used"],
        "gpu_mem_total": gpu["gpu_mem_total"],
        "gpu_power_draw": gpu["gpu_power_draw"],
        "cpu_util": cpu_util,
        "mem_used": mem.used / (1024 * 1024),  # MB
        "mem_total": mem.total / (1024 * 1024),  # MB
        "disk_used": disk.used / (1024 * 1024 * 1024),  # GB
        "disk_total": disk.total / (1024 * 1024 * 1024),  # GB
        "ollama_model": ollama_model,
        "ollama_model_size": ollama_model_size,
    }


def get_ollama_models() -> list:
    """Fetch available Ollama models from /api/tags."""
    import httpx
    try:
        resp = httpx.get("http://127.0.0.1:11435/api/tags", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("models", [])
    except Exception:
        pass
    return []
