"""SQLite connection, schema, and query helpers."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

DATABASE = Path(__file__).parent / "ollama_monitor.db"

_conn: Optional[sqlite3.Connection] = None


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(DATABASE), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _setup(_conn)
    return _conn


def _setup(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_tokens INTEGER DEFAULT 0,
            response_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            ttft_ms REAL,
            total_ms REAL,
            tps REAL,
            client_ip TEXT,
            status TEXT DEFAULT 'ok'
        );

        CREATE TABLE IF NOT EXISTS health (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            gpu_index INTEGER DEFAULT 0,
            gpu_name TEXT,
            gpu_temp REAL,
            gpu_util REAL,
            gpu_mem_used REAL,
            gpu_mem_total REAL,
            gpu_power_draw REAL,
            cpu_util REAL,
            mem_used REAL,
            mem_total REAL,
            disk_used REAL,
            disk_total REAL,
            ollama_model TEXT,
            ollama_model_size REAL
        );

        CREATE INDEX IF NOT EXISTS idx_requests_time ON requests(timestamp);
        CREATE INDEX IF NOT EXISTS idx_health_time ON health(timestamp);
        CREATE INDEX IF NOT EXISTS idx_requests_model_time ON requests(model, timestamp);
    """)


@contextmanager
def get_cursor():
    conn = get_conn()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def update_request_row(row_id: int, prompt_tokens: int, response_tokens: int,
                        ttft_ms: float, total_ms: float) -> None:
    with get_cursor() as cur:
        cur.execute(
            """UPDATE requests SET
               prompt_tokens=?, response_tokens=?,
               total_tokens=?, ttft_ms=?,
               total_ms=?, tps=?, status='ok'
               WHERE id=?""",
            (
                prompt_tokens,
                response_tokens,
                prompt_tokens + response_tokens,
                ttft_ms,
                total_ms,
                (response_tokens / total_ms * 1000) if total_ms and total_ms > 0 else 0,
                row_id,
            ),
        )


def insert_request(
    model: str,
    prompt_tokens: int,
    response_tokens: int,
    ttft_ms: Optional[float],
    total_ms: Optional[float],
    client_ip: str,
    status: str = "ok",
) -> int:
    with get_cursor() as cur:
        cur.execute(
            """INSERT INTO requests
               (timestamp, model, prompt_tokens, response_tokens, total_tokens,
                ttft_ms, total_ms, tps, client_ip, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.utcnow().isoformat(),
                model,
                prompt_tokens,
                response_tokens,
                prompt_tokens + response_tokens,
                ttft_ms,
                total_ms,
                (response_tokens / total_ms * 1000) if total_ms and total_ms > 0 else None,
                client_ip,
                status,
            ),
        )
        return cur.lastrowid


def insert_health(
    gpu_name: str,
    gpu_temp: float,
    gpu_util: float,
    gpu_mem_used: float,
    gpu_mem_total: float,
    gpu_power_draw: float,
    cpu_util: float,
    mem_used: float,
    mem_total: float,
    disk_used: float,
    disk_total: float,
    ollama_model: Optional[str] = None,
    ollama_model_size: Optional[float] = None,
) -> int:
    with get_cursor() as cur:
        cur.execute(
            """INSERT INTO health
               (timestamp, gpu_index, gpu_name, gpu_temp, gpu_util,
                gpu_mem_used, gpu_mem_total, gpu_power_draw,
                cpu_util, mem_used, mem_total, disk_used, disk_total,
                ollama_model, ollama_model_size)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.utcnow().isoformat(),
                0,
                gpu_name,
                gpu_temp,
                gpu_util,
                gpu_mem_used,
                gpu_mem_total,
                gpu_power_draw,
                cpu_util,
                mem_used,
                mem_total,
                disk_used,
                disk_total,
                ollama_model,
                ollama_model_size,
            ),
        )
        return cur.lastrowid


def get_stats_24h() -> dict:
    cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
    with get_cursor() as cur:
        cur.execute(
            """SELECT
                COUNT(*) as total_requests,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                COALESCE(AVG(ttft_ms), 0) as avg_ttft_ms,
                COALESCE(AVG(tps), 0) as avg_tps
               FROM requests WHERE timestamp >= ?""",
            (cutoff,),
        )
        row = cur.fetchone()
        return dict(row) if row else {}


def get_recent_requests(limit: int = 100) -> list:
    with get_cursor() as cur:
        cur.execute(
            """SELECT * FROM requests
               ORDER BY id DESC LIMIT ?""",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


def get_latest_health() -> Optional[dict]:
    with get_cursor() as cur:
        cur.execute(
            """SELECT * FROM health ORDER BY id DESC LIMIT 1"""
        )
        row = cur.fetchone()
        return dict(row) if row else None


def get_health_history(hours: int = 24) -> list:
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    with get_cursor() as cur:
        cur.execute(
            """SELECT * FROM health WHERE timestamp >= ?
               ORDER BY id ASC""",
            (cutoff,),
        )
        return [dict(r) for r in cur.fetchall()]


def get_hourly_volume(hours: int = 24) -> list:
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    with get_cursor() as cur:
        cur.execute(
            """SELECT
                strftime('%Y-%m-%dT%H:00:00', timestamp) as bucket,
                COUNT(*) as count
               FROM requests WHERE timestamp >= ?
               GROUP BY bucket ORDER BY bucket ASC""",
            (cutoff,),
        )
        return [dict(r) for r in cur.fetchall()]


def get_model_distribution(hours: int = 24) -> list:
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    with get_cursor() as cur:
        cur.execute(
            """SELECT model, COUNT(*) as count
               FROM requests WHERE timestamp >= ?
               GROUP BY model""",
            (cutoff,),
        )
        return [dict(r) for r in cur.fetchall()]


def get_daily_tokens(days: int = 7) -> list:
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    with get_cursor() as cur:
        cur.execute(
            """SELECT
                strftime('%Y-%m-%d', timestamp) as day,
                SUM(prompt_tokens) as prompt,
                SUM(response_tokens) as response
               FROM requests WHERE timestamp >= ?
               GROUP BY day ORDER BY day ASC""",
            (cutoff,),
        )
        return [dict(r) for r in cur.fetchall()]
