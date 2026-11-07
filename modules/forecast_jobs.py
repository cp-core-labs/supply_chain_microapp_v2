from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from modules.forecaster import forecast_all_skus
from supply_chain_microapp.config import APP_CONFIG


def _connect() -> sqlite3.Connection:
    """Abre una conexión independiente a SQLite."""
    db_path = Path(APP_CONFIG.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def initialize_forecast_jobs() -> None:
    """Crea la tabla de trabajos de forecast si todavía no existe."""
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS forecast_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                horizon_days INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                error_message TEXT,
                result_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_forecast_jobs_status
            ON forecast_jobs(status)
            """
        )
        conn.commit()
    finally:
        conn.close()


def build_forecast_job_id(daily_sales: pd.DataFrame, horizon: int) -> str:
    """
    Genera una clave estable para reutilizar el resultado si los datos
    y el horizonte del pronóstico no han cambiado.
    """
    data_hash = pd.util.hash_pandas_object(daily_sales, index=True).to_numpy().tobytes()
    columns = "|".join(str(column) for column in daily_sales.columns).encode("utf-8")
    horizon_bytes = str(int(horizon)).encode("utf-8")

    digest = hashlib.sha256(data_hash + columns + horizon_bytes).hexdigest()
    return digest


def get_forecast_job(job_id: str) -> dict[str, Any] | None:
    """Devuelve el estado y resultado de un trabajo."""
    initialize_forecast_jobs()
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT job_id, status, horizon_days, created_at, updated_at,
                   error_message, result_json
            FROM forecast_jobs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _set_job_failed(job_id: str, error_message: str) -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            UPDATE forecast_jobs
            SET status = 'FAILED',
                error_message = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
            """,
            (error_message[:2000], job_id),
        )
        conn.commit()
    finally:
        conn.close()


def _run_forecast(
    job_id: str,
    daily_sales: pd.DataFrame,
    horizon: int,
) -> None:
    """Función ejecutada en segundo plano. No debe llamar a Streamlit."""
    try:
        result_df = forecast_all_skus(daily_sales, horizon=horizon)

        if not isinstance(result_df, pd.DataFrame):
            raise TypeError("forecast_all_skus debe devolver un pandas DataFrame.")

        result_json = result_df.to_json(
            orient="records",
            date_format="iso",
        )

        conn = _connect()
        try:
            conn.execute(
                """
                UPDATE forecast_jobs
                SET status = 'READY',
                    result_json = ?,
                    error_message = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
                """,
                (result_json, job_id),
            )
            conn.commit()
        finally:
            conn.close()

    except Exception as exc:
        _set_job_failed(job_id, f"{type(exc).__name__}: {exc}")


def submit_forecast_job(
    daily_sales: pd.DataFrame,
    horizon: int,
    executor: Any,
    force: bool = False,
) -> str:
    """
    Registra y lanza un trabajo. Si ya hay un resultado válido, lo reutiliza,
    excepto cuando force=True.
    """
    initialize_forecast_jobs()
    job_id = build_forecast_job_id(daily_sales, horizon)

    conn = _connect()
    should_submit = False

    try:
        conn.execute("BEGIN IMMEDIATE")

        row = conn.execute(
            "SELECT status FROM forecast_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()

        if row is None:
            conn.execute(
                """
                INSERT INTO forecast_jobs (
                    job_id, status, horizon_days, updated_at
                )
                VALUES (?, 'PENDING', ?, CURRENT_TIMESTAMP)
                """,
                (job_id, int(horizon)),
            )
        elif row["status"] == "COMPUTING":
            conn.commit()
            return job_id
        elif row["status"] == "READY" and not force:
            conn.commit()
            return job_id
        else:
            conn.execute(
                """
                UPDATE forecast_jobs
                SET status = 'PENDING',
                    result_json = NULL,
                    error_message = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
                """,
                (job_id,),
            )

        claimed = conn.execute(
            """
            UPDATE forecast_jobs
            SET status = 'COMPUTING',
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ? AND status = 'PENDING'
            """,
            (job_id,),
        )
        should_submit = claimed.rowcount == 1
        conn.commit()

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    if should_submit:
        try:
            # La copia pertenece al trabajo; no se guarda en session_state.
            executor.submit(
                _run_forecast,
                job_id,
                daily_sales.copy(deep=True),
                int(horizon),
            )
        except Exception as exc:
            _set_job_failed(job_id, f"No se pudo iniciar el trabajo: {exc}")
            raise

    return job_id


def forecast_dataframe_from_job(job: dict[str, Any]) -> pd.DataFrame:
    """Convierte el resultado persistido en DataFrame."""
    result_json = job.get("result_json")
    if not result_json:
        return pd.DataFrame()

    from io import StringIO

    return pd.read_json(StringIO(result_json), orient="records")