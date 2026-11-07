from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from supply_chain_microapp.config import APP_CONFIG


def _connect() -> sqlite3.Connection:
    """Abre una conexión independiente a la base SQLite del proyecto."""
    db_path = Path(APP_CONFIG.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def initialize_risk_tables() -> None:
    """Crea las tablas de alertas e historial si todavía no existen."""
    conn = _connect()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS risk_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                sku TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                confidence TEXT NOT NULL,
                provisional INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'OPEN',
                first_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                resolved_at TEXT,
                UNIQUE(scope, fingerprint)
            );

            CREATE INDEX IF NOT EXISTS idx_risk_alerts_scope_status
            ON risk_alerts(scope, status);

            CREATE TABLE IF NOT EXISTS risk_alert_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                event_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                details TEXT,
                FOREIGN KEY(alert_id) REFERENCES risk_alerts(id)
            );

            CREATE INDEX IF NOT EXISTS idx_risk_alert_events_alert_id
            ON risk_alert_events(alert_id);
            """)
        conn.commit()
    finally:
        conn.close()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _to_number(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if pd.notna(result) else None


def _confidence_for_row(row: pd.Series) -> tuple[str, int, list[str]]:
    """Evalúa integridad básica de los campos que necesita el análisis de riesgo."""
    score = 100
    reasons: list[str] = []

    required_numeric = {
        "avg_daily_demand": "average demand",
        "avg_lead_time_days": "lead time",
        "stockout_probability": "stockout probability",
        "days_of_coverage": "days of coverage",
    }

    for column, label in required_numeric.items():
        if column not in row.index:
            score -= 20
            reasons.append(f"Missing field {label}.")
            continue

        value = _to_number(row[column])
        if value is None:
            score -= 20
            reasons.append(f"Field {label} is empty or non-numeric.")
        elif value < 0:
            score -= 15
            reasons.append(f"Field {label} has a negative value.")

    if "stockout_probability" in row.index:
        probability = _to_number(row["stockout_probability"])
        if probability is not None and probability > 1:
            score -= 20
            reasons.append("Stockout probability exceeds 1.")

    if _is_missing(row.get("sku")) or not str(row.get("sku", "")).strip():
        score -= 30
        reasons.append("Missing SKU identifier.")

    score = max(0, min(100, score))

    if score >= 80:
        confidence = "HIGH"
    elif score >= 55:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    if not reasons:
        reasons.append("Main fields present and within basic ranges.")

    return confidence, score, reasons


def assess_inventory_data(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """Añade confianza y motivos a cada fila del plan de inventario."""
    result = inventory_df.copy()

    if result.empty:
        result["confidence"] = pd.Series(dtype="object")
        result["confidence_score"] = pd.Series(dtype="int64")
        result["confidence_reasons"] = pd.Series(dtype="object")
        return result

    assessments = result.apply(_confidence_for_row, axis=1)
    result["confidence"] = assessments.apply(lambda item: item[0])
    result["confidence_score"] = assessments.apply(lambda item: item[1])
    result["confidence_reasons"] = assessments.apply(lambda item: " ".join(item[2]))
    return result


def build_risk_alerts(
    inventory_df: pd.DataFrame,
    scope: str = "default",
) -> pd.DataFrame:
    """Crea alertas explicables a partir del inventario evaluado."""
    if inventory_df.empty:
        return pd.DataFrame(
            columns=[
                "scope",
                "sku",
                "alert_type",
                "severity",
                "message",
                "confidence",
                "provisional",
            ]
        )

    assessed = assess_inventory_data(inventory_df)
    alerts: list[dict[str, Any]] = []

    lead_times = pd.to_numeric(
        assessed.get("avg_lead_time_days", pd.Series(dtype=float)),
        errors="coerce",
    )
    valid_lead_times = lead_times[lead_times > 0]
    lead_time_threshold = (
        float(valid_lead_times.quantile(0.90)) if not valid_lead_times.empty else None
    )

    for _, row in assessed.iterrows():
        sku_value = row.get("sku")
        sku = str(sku_value).strip() if not _is_missing(sku_value) else "UNKNOWN"

        confidence = str(row.get("confidence", "LOW"))
        provisional = confidence == "LOW"

        def add_alert(
            alert_type: str,
            severity: str,
            message: str,
            *,
            is_provisional: bool = provisional,
        ) -> None:
            alerts.append(
                {
                    "scope": scope,
                    "sku": sku,  # noqa: B023
                    "alert_type": alert_type,
                    "severity": severity,
                    "message": message,
                    "confidence": confidence,  # noqa: B023
                    "provisional": bool(is_provisional),
                }
            )

        confidence_reasons = str(row.get("confidence_reasons", ""))
        if confidence in {"LOW", "MEDIUM"}:
            add_alert(
                "DATA_QUALITY",
                "HIGH" if confidence == "LOW" else "MEDIUM",
                f"Confidence {confidence} ({row.get('confidence_score', 0)}/100). "
                f"{confidence_reasons}",
                is_provisional=False,
            )

        stockout_probability = _to_number(row.get("stockout_probability"))
        days_of_coverage = _to_number(row.get("days_of_coverage"))
        risk_score = _to_number(row.get("risk_score"))
        lead_time = _to_number(row.get("avg_lead_time_days"))

        if stockout_probability is not None and stockout_probability >= 0.30:
            add_alert(
                "STOCKOUT_RISK",
                "HIGH",
                f"High stockout risk: probability "
                f"{stockout_probability:.0%}."
                + (" Review data quality before taking action." if provisional else ""),
            )

        if days_of_coverage is not None and days_of_coverage < 2:
            add_alert(
                "LOW_COVERAGE",
                "HIGH" if days_of_coverage < 1 else "MEDIUM",
                f"Low coverage: {days_of_coverage:.1f} days."
                + (" Provisional alert due to low confidence." if provisional else ""),
            )

        if risk_score is not None and risk_score >= 70:
            add_alert(
                "HIGH_RISK_SCORE",
                "HIGH",
                f"High risk score: {risk_score:.1f}/100."
                + (" Provisional alert due to low confidence." if provisional else ""),
            )

        if (
            lead_time_threshold is not None
            and lead_time is not None
            and lead_time >= lead_time_threshold
        ):
            add_alert(
                "LONG_LEAD_TIME",
                "MEDIUM",
                f"High lead time in this set: {lead_time:.1f} days "
                f"(90th percentile: {lead_time_threshold:.1f})."
                + (" Provisional alert due to low confidence." if provisional else ""),
            )

    result = pd.DataFrame(alerts)
    if result.empty:
        return result

    return result.drop_duplicates(subset=["scope", "sku", "alert_type"]).reset_index(drop=True)


def _fingerprint(scope: str, sku: str, alert_type: str) -> str:
    raw = f"{scope}|{sku}|{alert_type}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def persist_risk_alerts(
    alerts_df: pd.DataFrame,
    scope: str = "default",
) -> None:
    """Actualiza alertas abiertas y resuelve las que ya no están presentes."""
    initialize_risk_tables()

    current_alerts = alerts_df.copy()
    if not current_alerts.empty:
        current_alerts = current_alerts[current_alerts["scope"].astype(str) == str(scope)]

    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")

        current_fingerprints: set[str] = set()

        for _, row in current_alerts.iterrows():
            sku = str(row["sku"])
            alert_type = str(row["alert_type"])
            fingerprint = _fingerprint(scope, sku, alert_type)
            current_fingerprints.add(fingerprint)

            existing = conn.execute(
                """
                SELECT id, severity, message, confidence, provisional, status
                FROM risk_alerts
                WHERE scope = ? AND fingerprint = ?
                """,
                (scope, fingerprint),
            ).fetchone()

            severity = str(row["severity"])
            message = str(row["message"])
            confidence = str(row["confidence"])
            provisional = int(bool(row["provisional"]))

            if existing is None:
                cursor = conn.execute(
                    """
                    INSERT INTO risk_alerts (
                        scope, fingerprint, sku, alert_type, severity,
                        message, confidence, provisional, status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
                    """,
                    (
                        scope,
                        fingerprint,
                        sku,
                        alert_type,
                        severity,
                        message,
                        confidence,
                        provisional,
                    ),
                )
                last_id = cursor.lastrowid
                alert_id = int(last_id) if last_id is not None else 0
                conn.execute(
                    """
                    INSERT INTO risk_alert_events (alert_id, event_type, details)
                    VALUES (?, 'CREATED', ?)
                    """,
                    (
                        alert_id,
                        json.dumps(
                            {"severity": severity, "message": message},
                            ensure_ascii=False,
                        ),
                    ),
                )
                continue

            alert_id = int(existing["id"])
            changed = (
                existing["severity"] != severity
                or existing["message"] != message
                or existing["confidence"] != confidence
                or int(existing["provisional"]) != provisional
                or existing["status"] != "OPEN"
            )

            conn.execute(
                """
                UPDATE risk_alerts
                SET severity = ?,
                    message = ?,
                    confidence = ?,
                    provisional = ?,
                    status = 'OPEN',
                    last_seen = CURRENT_TIMESTAMP,
                    resolved_at = NULL
                WHERE id = ?
                """,
                (severity, message, confidence, provisional, alert_id),
            )

            if changed:
                conn.execute(
                    """
                    INSERT INTO risk_alert_events (alert_id, event_type, details)
                    VALUES (?, 'UPDATED', ?)
                    """,
                    (
                        alert_id,
                        json.dumps(
                            {"severity": severity, "message": message},
                            ensure_ascii=False,
                        ),
                    ),
                )

        open_rows = conn.execute(
            """
            SELECT id, fingerprint
            FROM risk_alerts
            WHERE scope = ? AND status = 'OPEN'
            """,
            (scope,),
        ).fetchall()

        for row in open_rows:
            if row["fingerprint"] in current_fingerprints:
                continue

            conn.execute(
                """
                UPDATE risk_alerts
                SET status = 'RESOLVED',
                    resolved_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (row["id"],),
            )
            conn.execute(
                """
                INSERT INTO risk_alert_events (alert_id, event_type)
                VALUES (?, 'RESOLVED')
                """,
                (row["id"],),
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def load_open_risk_alerts(scope: str = "default") -> pd.DataFrame:
    """Carga alertas abiertas persistidas para mostrarlas en la app."""
    initialize_risk_tables()
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT sku, alert_type, severity, message, confidence,
                   provisional, status, first_seen, last_seen
            FROM risk_alerts
            WHERE scope = ? AND status = 'OPEN'
            ORDER BY
                CASE severity
                    WHEN 'HIGH' THEN 1
                    WHEN 'MEDIUM' THEN 2
                    ELSE 3
                END,
                last_seen DESC
            """,
            (scope,),
        ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])
    finally:
        conn.close()
