from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def export_inventory_excel(df: pd.DataFrame, output_path: str | Path) -> str:
    """Exporta un DataFrame de inventario a Excel."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)
    return str(path)


def export_forecast_excel(df: pd.DataFrame, output_path: str | Path) -> str:
    """Exporta una tabla de forecast a Excel."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)
    return str(path)


def export_csv(df: pd.DataFrame, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def export_summary_report(
    inventory_df: pd.DataFrame,
    forecast_df: Optional[pd.DataFrame] = None,
    output_path: str | Path = "exports/summary_report.xlsx",
) -> str:
    """Exporta un reporte consolidado con inventario y forecast."""
    export_df = inventory_df.copy()

    if forecast_df is not None and not forecast_df.empty:
        latest_forecast = (
            forecast_df[forecast_df["selected_model"] != "historical"]
            .sort_values("date")
            .drop_duplicates(subset=["sku", "date"], keep="last")
        )
        forecast_columns = [
            "sku",
            "date",
            "forecast_units",
            "lower_bound",
            "upper_bound",
        ]
        export_df = export_df.merge(
            latest_forecast[forecast_columns].rename(
                columns={"forecast_units": "forecast_units_latest"}
            ),
            on="sku",
            how="left",
        )

    return export_inventory_excel(export_df, output_path)