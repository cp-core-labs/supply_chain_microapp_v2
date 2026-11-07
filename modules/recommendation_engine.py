from __future__ import annotations

import pandas as pd

RECOMMENDATION_COLUMNS = [
    "sku",
    "product_name",
    "supplier_code",
    "action",
    "urgency",
    "reason",
    "confidence",
    "risk_score",
    "stockout_probability",
]


def _number(row: pd.Series, column: str) -> float | None:
    if column not in row.index or pd.isna(row[column]):
        return None

    try:
        value = float(row[column])
    except (TypeError, ValueError):
        return None

    return value if pd.notna(value) else None


def build_recommendations(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea recomendaciones de apoyo a la decisión usando el plan de inventario.

    No recomienda cantidades de compra: para eso se necesita inventario
    disponible real y, preferiblemente, órdenes de compra en tránsito.
    """
    if inventory_df.empty:
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)

    recommendations = []

    for _, row in inventory_df.iterrows():
        sku = str(row.get("sku", "")).strip()
        confidence = str(row.get("confidence", "UNKNOWN")).upper()

        risk_score = _number(row, "risk_score")
        stockout_probability = _number(row, "stockout_probability")
        lead_time = _number(row, "avg_lead_time_days")
        current_inventory = _number(row, "current_inventory")
        demand = _number(row, "avg_daily_demand")

        action = "MONITOR"
        urgency = "LOW"
        reason = "No se detectó una condición que requiera acción inmediata."

        # Si la calidad no es alta, primero se debe revisar la información.
        if confidence == "LOW":
            action = "VALIDATE_DATA"
            urgency = "HIGH"
            reason = (
                "La confianza de los datos es baja. "
                "Valida los datos antes de tomar una decisión de compra."
            )

        else:
            has_actual_inventory = current_inventory is not None
            coverage_days = None

            if has_actual_inventory and demand is not None and demand > 0:
                coverage_days = max(0.0, current_inventory / demand)

            # Solo se compara cobertura contra lead time si hay stock real.
            if coverage_days is not None and lead_time is not None and coverage_days < lead_time:
                action = "EXPEDITE_OR_REVIEW"
                urgency = "HIGH"
                reason = (
                    f"El inventario cubre aproximadamente {coverage_days:.1f} días, "
                    f"menos que el lead time de {lead_time:.1f} días."
                )

            elif stockout_probability is not None and stockout_probability >= 0.30:
                action = "REVIEW_REPLENISHMENT"
                urgency = "HIGH"
                reason = (
                    f"Probabilidad de stockout elevada "
                    f"({stockout_probability:.0%}). Revisa la reposición."
                )

            elif risk_score is not None and risk_score >= 60:
                action = "REVIEW_SUPPLIER_AND_POLICY"
                urgency = "MEDIUM"
                reason = (
                    f"Riesgo alto ({risk_score:.1f}/100). "
                    "Revisa lead time, proveedor y política de inventario."
                )

            elif stockout_probability is not None and stockout_probability >= 0.15:
                action = "MONITOR_CLOSER"
                urgency = "MEDIUM"
                reason = (
                    f"Riesgo de stockout moderado "
                    f"({stockout_probability:.0%}). Aumenta el seguimiento."
                )

            if confidence == "MEDIUM" and action != "MONITOR":
                reason += " La confianza es media; valida los datos antes de actuar."

        recommendations.append(
            {
                "sku": sku or "UNKNOWN",
                "product_name": row.get("product_name", ""),
                "supplier_code": row.get("supplier_code", ""),
                "action": action,
                "urgency": urgency,
                "reason": reason,
                "confidence": confidence,
                "risk_score": risk_score,
                "stockout_probability": stockout_probability,
            }
        )

    result = pd.DataFrame(recommendations)

    urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    result["_urgency_order"] = result["urgency"].map(urgency_order).fillna(3)

    return (
        result.sort_values(
            ["_urgency_order", "risk_score"],
            ascending=[True, False],
            na_position="last",
        )
        .drop(columns="_urgency_order")
        .reset_index(drop=True)
    )
