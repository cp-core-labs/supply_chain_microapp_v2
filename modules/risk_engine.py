from __future__ import annotations

import pandas as pd


def build_alerts(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """Genera una tabla de alertas tipo semáforo:
    - Verde: saludable
    - Amarillo: revisar
    - Rojo: urgente
    """
    df = inventory_df.copy()

    if df.empty:
        return df

    def alert_level(row):
        if row.get("stockout_probability", 0.0) > 0.30:
            return "Rojo"
        if row.get("days_of_coverage", 99) < 2:
            return "Amarillo"
        if row.get("risk_score", 0.0) > 50:
            return "Amarillo"
        return "Verde"

    df["alert_level"] = df.apply(alert_level, axis=1)

    # Prioridad de compra
    df["priority_score"] = (
        df.get("stockout_probability", 0.0) * 100
        + df.get("risk_score", 0.0)
        + (100 / (df.get("days_of_coverage", 1) + 1))
    )

    df["priority_label"] = df["priority_score"].apply(
        lambda x: "Urgente" if x > 70 else ("Prioridad media" if x > 40 else "Normal")
    )

    return df.sort_values(["priority_score", "risk_score"], ascending=[False, False]).reset_index(
        drop=True
    )


def evaluate_what_if_scenario(
    inventory_df: pd.DataFrame,
    demand_change_pct: float = 0.20,
    lead_time_change_days: float = 3.0,
    service_level: float = 0.95,
) -> pd.DataFrame:
    """Analiza escenarios de 'qué pasaría si':
    - cambio de demanda
    - expansión del lead time
    - ajuste del nivel de servicio
    """
    df = inventory_df.copy()

    if df.empty:
        return df

    demand_factor = 1.0 + float(demand_change_pct)
    lead_time_factor = float(lead_time_change_days)

    df["scenario_demand_change_pct"] = demand_change_pct
    df["scenario_lead_time_change_days"] = lead_time_change_days

    df["scenario_avg_daily_demand"] = df["avg_daily_demand"] * demand_factor
    df["scenario_lead_time_days"] = df["avg_lead_time_days"] + lead_time_factor

    # Recalcular seguridad bajo nuevo escenario
    df["scenario_safety_stock"] = df.apply(
        lambda row: (
            max(
                0.0,
                row["scenario_avg_daily_demand"]
                * row["demand_std"]
                * row["scenario_lead_time_days"],
            )
            * 0.8
        ),
        axis=1,
    )

    df["scenario_reorder_point"] = (
        df["scenario_avg_daily_demand"] * df["scenario_lead_time_days"]
        + df["scenario_safety_stock"]
    )

    # Impacto sobre riesgo (Corregido con .clip(lower=1) para evitar errores de ambigüedad en max())
    avg_lead_time_safe = df["avg_lead_time_days"].clip(lower=1)
    df["scenario_risk_score"] = (
        df["stockout_probability"] * 100 * (1.0 + abs(demand_change_pct))
    ) + ((lead_time_factor / avg_lead_time_safe) * 35)

    df["scenario_alert_level"] = df["scenario_risk_score"].apply(
        lambda x: "Rojo" if x > 70 else ("Amarillo" if x > 45 else "Verde")
    )

    return df.sort_values("scenario_risk_score", ascending=False).reset_index(drop=True)


def build_risk_summary(alerts_df: pd.DataFrame) -> pd.DataFrame:
    """Resumen ejecutivo de riesgos por clasificación."""
    if alerts_df.empty:
        return pd.DataFrame()

    summary = (
        alerts_df.groupby("alert_level", as_index=False)
        .agg(
            sku_count=("sku", "nunique"),
            total_risk_score=("risk_score", "sum"),
            avg_stockout_probability=("stockout_probability", "mean"),
        )
        .sort_values("total_risk_score", ascending=False)
    )

    return summary


if __name__ == "__main__":
    sample = pd.DataFrame(
        [
            {
                "sku": "SKU-001",
                "risk_score": 62,
                "stockout_probability": 0.34,
                "days_of_coverage": 1.5,
                "avg_daily_demand": 120,
                "avg_lead_time_days": 10,
                "demand_std": 20,
            },
            {
                "sku": "SKU-002",
                "risk_score": 28,
                "stockout_probability": 0.09,
                "days_of_coverage": 4.5,
                "avg_daily_demand": 50,
                "avg_lead_time_days": 8,
                "demand_std": 10,
            },
        ]
    )

    alerts = build_alerts(sample)
    print(alerts[["sku", "alert_level", "priority_label"]])