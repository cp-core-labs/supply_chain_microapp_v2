from __future__ import annotations

from typing import Any

from .storage import get_connection


class BaseRepository:
    def __init__(self, table_name: str):
        self.table_name = table_name

    def fetch_all(self) -> list[dict[str, Any]]:
        conn = get_connection()
        try:
            rows = conn.execute(f"SELECT * FROM {self.table_name}").fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def fetch_by_id(self, table_key: str, value: str) -> dict[str, Any] | None:
        conn = get_connection()
        try:
            row = conn.execute(
                f"SELECT * FROM {self.table_name} WHERE {table_key} = ?", (value,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


class SupplierRepository(BaseRepository):
    def __init__(self):
        super().__init__("suppliers")


class SalesRepository(BaseRepository):
    def __init__(self):
        super().__init__("sales")


class InventoryRepository(BaseRepository):
    def __init__(self):
        super().__init__("inventory_snapshots")


class ForecastRepository(BaseRepository):
    def __init__(self):
        super().__init__("forecast_runs")


class AlertRepository(BaseRepository):
    def __init__(self):
        super().__init__("alerts")
