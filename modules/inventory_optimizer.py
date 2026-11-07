from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd


def _service_level_z(service_level: float) -> float:
    """Aproxima el z-score para un nivel de servicio.
    Para un SL del 95%, z ≈ 1.645
    """
    service_level = float(service_level)
    mapping = {
        0.90: 1.2816,
        0.91: 1.3408,
        0.92: 1.4051,
        0.93: 1.4758,
        0.94: 1.5548,
        0.95: 1.6449,
        0.96: 1.7507,
        0.97: 1.8808,
        0.98: 2.0537,
        0.99: 2.3263,
    }

    if service_level in mapping:
        return mapping[service_level]

    # Aproximación general para valores no tabulados
    if service_level <= 0 or service_level >= 1:
        raise ValueError("El service level debe estar entre 0 y 1.")

    z = 0.0
    if service_level >= 0.5:
        z = 1.0
        for _ in range(100):
            z = z + 0.01
        z = (service_level - 0.5) * 20.0
        z = max(0.0, z)
    return max(0.0, z + 0.5)


def calculate_safety_stock(
    avg_daily_demand: float,
    demand_std: float,
    lead_time_days: float,
    lead_time_std_days: float = 0.0,
    service_level: float = 0.95,
) -> float:
    """Calcula Safety Stock:
    SS = z * sqrt( (LT * sigma_d)^2 + (d * sigma_LT)^2 )
    """
    avg_daily_demand = float(avg_daily_demand)
    demand_std = float(demand_std)
    lead_time_days = float(lead_time_days)
    lead_time_std_days = float(lead_time_std_days)
    service_level = float(service_level)

    if avg_daily_demand <= 0:
        return 0.0

    z = _service_level_z(service_level)
    variance_term = (lead_time_days * demand_std) ** 2
    variance_term_2 = (avg_daily_demand * lead_time_std_days) ** 2
    sigma_during_lead_time = math.sqrt(variance_term + variance_term_2)

    return max(0.0, z * sigma_during_lead_time)


def calculate_reorder_point(
    avg_daily_demand: float,
    lead_time_days: float,
    safety_stock: float,
) -> float:
    """ROP = demanda promedio durante el lead time + safety stock"""
    return max(0.0, (avg_daily_demand * lead_time_days) + safety_stock)


def calculate_eoq(
    annual_demand: float,
    ordering_cost: float,
    holding_cost_per_unit_year: float,
) -> float:
    """EOQ = sqrt(2 * D * S / H)"""
    annual_demand = float(annual_demand)
    ordering_cost = float(ordering_cost)
    holding_cost_per_unit_year = float(holding_cost_per_unit_year)

    if annual_demand <= 0 or ordering_cost <= 0 or holding_cost_per_unit_year <= 0:
        return 0.0

    return math.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit_year)


def days_of_coverage(current_inventory: float, avg_daily_demand: float) -> float:
    """Días de cobertura: inventario actual / demanda promedio diaria"""
    if avg_daily_demand <= 0:
        return 0.0
    return max(0.0, current_inventory / avg_daily_demand)


def assign_abc_classification(
    df: pd.DataFrame, value_col: str = "annual_dollar_usage", sku_col: str = "sku"
) -> pd.DataFrame:
    """Asigna clase ABC usando Pareto:
    - A: 80% acumulado
    - B: siguiente 15%
    - C: restante 5%
    """
    out = df.copy()
    if value_col not in out.columns:
        out[value_col] = 0.0

    out = out.sort_values(value_col, ascending=False).reset_index(drop=True)
    total = out[value_col].sum()
    if total <= 0:
        out["abc_class"] = "C"
        return out.sort_values(sku_col).reset_index(drop=True)

    cumulative = 0.0
    abc = []
    for _, row in out.iterrows():
        cumulative += float(row[value_col])
        share = cumulative / total

        if share <= 0.80:
            cls = "A"
        elif share <= 0.95:
            cls = "B"
        else:
            cls = "C"

        abc.append(cls)

    out["abc_class"] = abc
    out = out.sort_values(sku_col).reset_index(drop=True)
    return out


def assign_ved_classification(
    df: pd.DataFrame, criticality_col: str = "criticality", sku_col: str = "sku"
) -> pd.DataFrame:
    """Clasifica por criticidad tipo VED:
    - V = Vital
    - E = Essential
    - D = Desirable
    """
    out = df.copy()
    if criticality_col not in out.columns:
        out[criticality_col] = "Essential"

    def map_value(val: str) -> str:
        if pd.isna(val):
            return "E"
        v = str(val).strip().lower()
        if v in ["vital", "v"]:
            return "V"
        if v in ["essential", "e"]:
            return "E"
        if v in ["desirable", "d"]:
            return "D"
        return "E"

    out["ved_class"] = out[criticality_col].apply(map_value)
    out = out.sort_values(sku_col).reset_index(drop=True)
    return out


def simulate_lead_time_uncertainty(
    avg_daily_demand: float,
    demand_std: float,
    lead_time_mean: float,
    lead_time_std: float,
    reorder_point: float,
    n_simulations: int = 5000,
    seed: int = 42,
) -> Dict[str, float]:
    """Simulación Monte Carlo de demanda durante el lead time."""
    rng = np.random.default_rng(seed)

    stockout_count = 0
    demand_values = []
    lead_times = []

    for _ in range(n_simulations):
        lead_time_sample = max(0.0, rng.normal(lead_time_mean, lead_time_std))
        lead_times.append(lead_time_sample)

        average_demand = avg_daily_demand * lead_time_sample
        demand_variance = lead_time_sample * demand_std**2
        demand_sample = rng.normal(average_demand, math.sqrt(max(demand_variance, 0.0)))

        demand_values.append(demand_sample)
        if demand_sample > reorder_point:
            stockout_count += 1

    stockout_probability = stockout_count / n_simulations if n_simulations else 0.0
    mean_lead_time = float(np.mean(lead_times)) if lead_times else lead_time_mean
    mean_demand_during_lt = float(np.mean(demand_values)) if demand_values else 0.0

    return {
        "stockout_probability": float(stockout_probability),
        "mean_lead_time": mean_lead_time,
        "mean_demand_during_lt": mean_demand_during_lt,
        "recommended_safety_stock": max(0.0, mean_demand_during_lt - reorder_point * 0.5),
    }


def build_inventory_plan(
    processed_data: Dict[str, pd.DataFrame],
    service_level: float = 0.95,
    ordering_cost: float = 150.0,
    holding_cost_pct: float = 0.22,
) -> pd.DataFrame:
    """Genera un plan operativo de inventario a nivel SKU."""
    sku_summary = processed_data.get("sku_summary")
    if sku_summary is None:
        raise ValueError("processed_data no contiene 'sku_summary'.")

    df = sku_summary.copy()

    required_cols = [
        "sku",
        "product_name",
        "category",
        "avg_daily_demand",
        "demand_std",
        "avg_lead_time_days",
        "lead_time_std_days",
        "avg_unit_price",
        "service_level_target",
        "criticality",
        "reorder_point",
    ]

    for col in required_cols:
        if col not in df.columns:
            if col in [
                "demand_std",
                "lead_time_std_days",
                "avg_lead_time_days",
                "avg_daily_demand",
                "avg_unit_price",
                "service_level_target",
            ]:
                df[col] = 0.0
            else:
                df[col] = ""

    df["avg_daily_demand"] = pd.to_numeric(df["avg_daily_demand"], errors="coerce").fillna(0.0)
    df["demand_std"] = pd.to_numeric(df["demand_std"], errors="coerce").fillna(0.0)
    df["avg_lead_time_days"] = pd.to_numeric(df["avg_lead_time_days"], errors="coerce").fillna(0.0)
    df["lead_time_std_days"] = pd.to_numeric(df["lead_time_std_days"], errors="coerce").fillna(0.0)
    df["avg_unit_price"] = pd.to_numeric(df["avg_unit_price"], errors="coerce").fillna(0.0)
    df["service_level_target"] = pd.to_numeric(
        df["service_level_target"], errors="coerce"
    ).fillna(0.95)

    df["annual_demand"] = df["avg_daily_demand"] * 365.0
    df["annual_dollar_usage"] = df["annual_demand"] * df["avg_unit_price"]

    df = assign_abc_classification(df, value_col="annual_dollar_usage", sku_col="sku")
    df = assign_ved_classification(df, criticality_col="criticality", sku_col="sku")

    df["safety_stock"] = df.apply(
        lambda row: calculate_safety_stock(
            row["avg_daily_demand"],
            row["demand_std"],
            row["avg_lead_time_days"],
            row["lead_time_std_days"],
            service_level=row.get("service_level_target", service_level),
        ),
        axis=1,
    )

    df["reorder_point"] = df.apply(
        lambda row: calculate_reorder_point(
            row["avg_daily_demand"],
            row["avg_lead_time_days"],
            row["safety_stock"],
        ),
        axis=1,
    )

    df["holding_cost_per_unit_year"] = df["avg_unit_price"] * holding_cost_pct
    df["eoq"] = df.apply(
        lambda row: calculate_eoq(
            row["annual_demand"],
            ordering_cost=ordering_cost,
            holding_cost_per_unit_year=row["holding_cost_per_unit_year"],
        ),
        axis=1,
    )

    df["recommended_order_quantity"] = np.maximum(df["eoq"], df["reorder_point"] * 0.4)

    df["days_of_coverage"] = df.apply(
        lambda row: days_of_coverage(row.get("reorder_point", 0.0), row["avg_daily_demand"]),
        axis=1,
    )

    simulation_rows = []
    for _, row in df.iterrows():
        sim = simulate_lead_time_uncertainty(
            avg_daily_demand=row["avg_daily_demand"],
            demand_std=row["demand_std"],
            lead_time_mean=row["avg_lead_time_days"],
            lead_time_std=row["lead_time_std_days"],
            reorder_point=row["reorder_point"],
            n_simulations=2000,
            seed=int(abs(hash(str(row["sku"]))) % 100000),
        )
        simulation_rows.append(sim)

    sim_df = pd.DataFrame(simulation_rows)
    df["stockout_probability"] = sim_df["stockout_probability"].values
    df["risk_score"] = (df["stockout_probability"] * 100) * 0.7 + (
        1 / (df["days_of_coverage"] + 1)
    ) * 100 * 0.3

    if "supplier_code" in df.columns:
        df["supplier_code"] = df["supplier_code"].fillna("UNKNOWN")

    return df.sort_values(
        ["risk_score", "annual_dollar_usage"], ascending=[False, False]
    ).reset_index(drop=True)


def summarize_inventory_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen de inventario por categoría."""
    if df.empty:
        return pd.DataFrame()

    group = (
        df.groupby("category", as_index=False)
        .agg(
            sku_count=("sku", "nunique"),
            avg_daily_demand=("avg_daily_demand", "mean"),
            total_annual_demand=("annual_demand", "sum"),
            avg_risk_score=("risk_score", "mean"),
            total_safety_stock=("safety_stock", "sum"),
            total_reorder_point=("reorder_point", "sum"),
        )
        .sort_values("avg_risk_score", ascending=False)
    )
    return group


if __name__ == "__main__":
    d = {
        "sku_summary": pd.DataFrame(
            [
                {
                    "sku": "SKU-001",
                    "product_name": "Widget A",
                    "category": "A",
                    "avg_daily_demand": 110,
                    "demand_std": 18,
                    "avg_lead_time_days": 12,
                    "lead_time_std_days": 3,
                    "avg_unit_price": 35,
                    "service_level_target": 0.95,
                    "criticality": "Vital",
                    "reorder_point": 1100,
                },
                {
                    "sku": "SKU-002",
                    "product_name": "Widget B",
                    "category": "B",
                    "avg_daily_demand": 40,
                    "demand_std": 9,
                    "avg_lead_time_days": 8,
                    "lead_time_std_days": 2,
                    "avg_unit_price": 18,
                    "service_level_target": 0.92,
                    "criticality": "Essential",
                    "reorder_point": 700,
                },
            ]
        )
    }

    plan = build_inventory_plan(d, service_level=0.95)
    print(
        plan[
            [
                "sku",
                "abc_class",
                "ved_class",
                "safety_stock",
                "reorder_point",
                "eoq",
                "stockout_probability",
            ]
        ]
    )