from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _safe_pick_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the real column name if one of the candidates exists."""
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for name in candidates:
        key = str(name).strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns to lowercase and trim spaces."""
    out = df.copy()
    out.columns = [str(col).strip().lower() for col in out.columns]
    return out


def clean_supplier_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean supplier data without breaking if optional columns are missing."""
    out = _normalize_columns(df)

    required_cols = {"supplier_code"}
    missing_required = required_cols - set(out.columns)
    if missing_required:
        raise ValueError(f"Faltan columnas esenciales en proveedores: {missing_required}")

    numeric_like = [
        "avg_lead_time_days",
        "lead_time_std_days",
        "service_level_target",
        "on_time_delivery_rate",
        "shipping_cost_per_unit",
        "risk_factor",
    ]

    for col in numeric_like:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["supplier_name", "region", "criticality"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()

    out = out.dropna(subset=["supplier_code"]).copy()
    out = out.drop_duplicates(subset=["supplier_code"]).reset_index(drop=True)
    return out


def clean_sales_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean sales data using tolerant validation."""
    out = _normalize_columns(df)

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).copy()

    if "supplier_code" not in out.columns:
        out["supplier_code"] = "UNKNOWN"

    for col in [
        "sku",
        "product_name",
        "category",
        "supplier_code",
        "supplier_name",
        "region",
    ]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()

    numeric_cols = [
        "units_sold",
        "unit_price",
        "unit_cost",
        "revenue",
        "gross_margin",
        "avg_daily_demand",
        "reorder_point",
        "service_level_target",
    ]

    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "units_sold" in out.columns:
        out["units_sold"] = out["units_sold"].clip(lower=0)

    if "revenue" in out.columns:
        out["revenue"] = out["revenue"].fillna(0.0)

    if "gross_margin" in out.columns:
        out["gross_margin"] = out["gross_margin"].fillna(0.0)

    out = out.drop_duplicates().sort_values(["sku", "date"]).reset_index(drop=True)
    return out


def merge_supplier_context(
    sales_df: pd.DataFrame,
    supplier_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge supplier context without assuming all columns exist."""
    sales = clean_sales_dataframe(sales_df)
    suppliers = clean_supplier_dataframe(supplier_df)

    alias_map = {
        "supplier_name": "supplier_name_supplier",
        "region": "region_supplier",
        "avg_lead_time_days": "avg_lead_time_days",
        "lead_time_std_days": "lead_time_std_days",
        "service_level_target": "service_level_target",
        "on_time_delivery_rate": "on_time_delivery_rate",
        "shipping_cost_per_unit": "shipping_cost_per_unit",
        "risk_factor": "risk_factor",
        "criticality": "criticality",
    }

    rename_map = {src: dst for src, dst in alias_map.items() if src in suppliers.columns}
    supplier_subset = suppliers.rename(columns=rename_map)

    merged = sales.merge(
        supplier_subset,
        on="supplier_code",
        how="left",
        suffixes=("_sales", "_supplier"),
    )

    final_cols = {
        "supplier_name": ["supplier_name", "supplier_name_supplier", "supplier_name_sales"],
        "region": ["region", "region_supplier", "region_sales"],
        "avg_lead_time_days": ["avg_lead_time_days"],
        "lead_time_std_days": ["lead_time_std_days"],
        "service_level_target": ["service_level_target"],
        "on_time_delivery_rate": ["on_time_delivery_rate"],
        "shipping_cost_per_unit": ["shipping_cost_per_unit"],
        "risk_factor": ["risk_factor"],
        "criticality": ["criticality"],
    }

    for target, candidates in final_cols.items():
        source_col = _safe_pick_column(merged, candidates)

        if source_col is None:
            merged[target] = pd.NA
        else:
            if source_col != target:
                merged = merged.rename(columns={source_col: target})

            if target in ["supplier_name", "region"]:
                merged[target] = merged[target].fillna("Unknown")
            else:
                merged[target] = merged[target].fillna(pd.NA)

    merged["supplier_name"] = merged["supplier_name"].fillna("Unknown Supplier")
    merged["region"] = merged["region"].fillna("Unknown Region")

    numeric_columns = [
        "avg_lead_time_days",
        "lead_time_std_days",
        "service_level_target",
        "on_time_delivery_rate",
        "shipping_cost_per_unit",
        "risk_factor",
    ]
    for col in numeric_columns:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    return merged.sort_values(["sku", "date"]).reset_index(drop=True)


def process_supply_chain_data(sales_path: str | Path, supplier_path: str | Path) -> dict:
    """Process a supply chain dataset to produce the summary tables used by the app."""
    sales_df = pd.read_csv(sales_path)
    supplier_df = pd.read_csv(supplier_path)

    sales_clean = clean_sales_dataframe(sales_df)
    suppliers_clean = clean_supplier_dataframe(supplier_df)
    merged_data = merge_supplier_context(sales_df, supplier_df)

    if "sku" in merged_data.columns:
        agg_dict = {
            "units_sold": "sum",
            "unit_price": "mean",
            "unit_cost": "mean",
        }
        for col in [
            "avg_lead_time_days",
            "lead_time_std_days",
            "service_level_target",
            "on_time_delivery_rate",
            "shipping_cost_per_unit",
            "risk_factor",
        ]:
            if col in merged_data.columns:
                agg_dict[col] = "first"

        sku_summary = merged_data.groupby("sku").agg(agg_dict).reset_index()

        if "units_sold" in sku_summary.columns and "unit_price" in sku_summary.columns:
            sku_summary["total_revenue"] = sku_summary["units_sold"] * sku_summary["unit_price"]
        else:
            sku_summary["total_revenue"] = 1.0

        sku_summary = sku_summary.sort_values(
            by="total_revenue",
            ascending=False,
        ).reset_index(drop=True)

        total_rev = sku_summary["total_revenue"].sum()
        if total_rev > 0:
            sku_summary["cum_percentage"] = sku_summary["total_revenue"].cumsum() / total_rev
            sku_summary["abc_class"] = sku_summary["cum_percentage"].apply(
                lambda x: "A" if x <= 0.80 else ("B" if x <= 0.95 else "C")
            )
            sku_summary = sku_summary.drop(columns=["total_revenue", "cum_percentage"])
        else:
            sku_summary["abc_class"] = "B"
            if "total_revenue" in sku_summary.columns:
                sku_summary = sku_summary.drop(columns=["total_revenue"])
    else:
        sku_summary = merged_data.copy()
        sku_summary["abc_class"] = "B"

    sku_summary["abc_class"] = sku_summary["abc_class"].astype(str)

    return {
        "sales_clean": sales_clean,
        "suppliers_clean": suppliers_clean,
        "merged": merged_data,
        "daily_sales": merged_data,
        "sku_summary": sku_summary,
    }
