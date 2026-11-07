from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .config import APP_CONFIG


def ensure_db_dir() -> Path:
    db_dir = APP_CONFIG.db_dir
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir


def get_connection() -> sqlite3.Connection:
    ensure_db_dir()
    conn = sqlite3.connect(APP_CONFIG.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def initialize_database() -> None:
    conn = get_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS raw_staged_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                error_message TEXT
            );

            CREATE TABLE IF NOT EXISTS suppliers (
                supplier_code TEXT PRIMARY KEY,
                supplier_name TEXT,
                region TEXT,
                avg_lead_time_days REAL,
                lead_time_std_days REAL,
                service_level_target REAL,
                on_time_delivery_rate REAL,
                shipping_cost_per_unit REAL,
                risk_factor REAL,
                criticality TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT NOT NULL,
                product_name TEXT,
                category TEXT,
                supplier_code TEXT,
                supplier_name TEXT,
                region TEXT,
                date TEXT NOT NULL,
                units_sold REAL,
                unit_price REAL,
                unit_cost REAL,
                revenue REAL,
                gross_margin REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS inventory_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date TEXT DEFAULT CURRENT_TIMESTAMP,
                sku TEXT NOT NULL,
                product_name TEXT,
                category TEXT,
                avg_daily_demand REAL,
                demand_std REAL,
                avg_lead_time_days REAL,
                lead_time_std_days REAL,
                safety_stock REAL,
                reorder_point REAL,
                eoq REAL,
                stockout_probability REAL,
                risk_score REAL,
                alert_level TEXT
            );

            CREATE TABLE IF NOT EXISTS forecast_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                horizon_days INTEGER,
                selected_model TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                payload TEXT NOT NULL,
                confidence_score REAL
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                sku TEXT NOT NULL,
                product_name TEXT,
                category TEXT,
                alert_level TEXT,
                priority_label TEXT,
                risk_score REAL,
                stockout_probability REAL,
                message TEXT
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def set_setting(key: str, value: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO settings (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key)
            DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
            """,
            (key, value),
        )
        conn.commit()
    finally:
        conn.close()


def get_setting(key: str) -> str | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None
    finally:
        conn.close()


def save_raw_staged_data(source_name: str, payload: dict[str, Any], status: str = "pending", error_message: str | None = None) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO raw_staged_data (source_name, payload, status, error_message)
            VALUES (?, ?, ?, ?)
            """,
            (
                source_name,
                json.dumps(payload, default=str),
                status,
                error_message,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def upsert_supplier(row: dict[str, Any]) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO suppliers (
                supplier_code, supplier_name, region, avg_lead_time_days,
                lead_time_std_days, service_level_target, on_time_delivery_rate,
                shipping_cost_per_unit, risk_factor, criticality
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(supplier_code)
            DO UPDATE SET
                supplier_name = excluded.supplier_name,
                region = excluded.region,
                avg_lead_time_days = excluded.avg_lead_time_days,
                lead_time_std_days = excluded.lead_time_std_days,
                service_level_target = excluded.service_level_target,
                on_time_delivery_rate = excluded.on_time_delivery_rate,
                shipping_cost_per_unit = excluded.shipping_cost_per_unit,
                risk_factor = excluded.risk_factor,
                criticality = excluded.criticality
            """,
            (
                row.get("supplier_code"),
                row.get("supplier_name"),
                row.get("region"),
                row.get("avg_lead_time_days"),
                row.get("lead_time_std_days"),
                row.get("service_level_target"),
                row.get("on_time_delivery_rate"),
                row.get("shipping_cost_per_unit"),
                row.get("risk_factor"),
                row.get("criticality"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def save_sales_batch(rows: Iterable[dict[str, Any]]) -> None:
    conn = get_connection()
    try:
        conn.executemany(
            """
            INSERT INTO sales (
                sku, product_name, category, supplier_code, supplier_name, region,
                date, units_sold, unit_price, unit_cost, revenue, gross_margin
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.get("sku"),
                    row.get("product_name"),
                    row.get("category"),
                    row.get("supplier_code"),
                    row.get("supplier_name"),
                    row.get("region"),
                    row.get("date"),
                    row.get("units_sold"),
                    row.get("unit_price"),
                    row.get("unit_cost"),
                    row.get("revenue"),
                    row.get("gross_margin"),
                )
                for row in rows
            ],
        )
        conn.commit()
    finally:
        conn.close()


def save_inventory_snapshot(row: dict[str, Any]) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO inventory_snapshots (
                snapshot_date, sku, product_name, category, avg_daily_demand,
                demand_std, avg_lead_time_days, lead_time_std_days,
                safety_stock, reorder_point, eoq, stockout_probability, risk_score, alert_level
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("snapshot_date"),
                row.get("sku"),
                row.get("product_name"),
                row.get("category"),
                row.get("avg_daily_demand"),
                row.get("demand_std"),
                row.get("avg_lead_time_days"),
                row.get("lead_time_std_days"),
                row.get("safety_stock"),
                row.get("reorder_point"),
                row.get("eoq"),
                row.get("stockout_probability"),
                row.get("risk_score"),
                row.get("alert_level"),
            ),
        )
        conn.commit()
    finally:
        conn.close()