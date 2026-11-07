from __future__ import annotations

# ruff: noqa: E402
import sys
from pathlib import Path

# Añadir la carpeta 'src' al sys.path para resolver las importaciones del paquete
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st

from modules.forecast_jobs import (
    build_forecast_job_id,
    forecast_dataframe_from_job,
    get_forecast_job,
    initialize_forecast_jobs,
    submit_forecast_job,
)
from modules.recommendation_engine import build_recommendations
from modules.risk_monitor import (
    build_risk_alerts,
    load_open_risk_alerts,
    persist_risk_alerts,
)
from supply_chain_microapp.storage import initialize_database

initialize_database()
initialize_forecast_jobs()

try:
    import plotly.express as px
except Exception:
    px = None

from data_generator import generate_synthetic_supply_chain_data
from modules.data_processor import process_supply_chain_data
from modules.inventory_optimizer import build_inventory_plan
from modules.risk_engine import build_alerts, build_risk_summary, evaluate_what_if_scenario

st.set_page_config(
    page_title="Supply Chain Micro App v2.0",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# Premium CSS / Supply Chain Control Tower Theme (v2.0.0)
# ==========================================================
st.markdown(
    """
    <style>
        :root {
            --navy: #102A43;
            --navy-soft: #173B57;
            --teal: #0F766E;
            --teal-light: #CCFBF1;
            --ink: #172B4D;
            --muted: #627D98;
            --canvas: #F3F6FA;
            --line: #D9E2EC;
            --white: #FFFFFF;
        }

        /* Page */
        .stApp {
            background: var(--canvas);
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--navy) 0%, #0B1F33 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #F3F7FA;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #C5D3E0;
        }

        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            color: #E6EDF3;
            font-weight: 600;
        }

        /* Main page heading */
        .premium-header {
            position: relative;
            overflow: hidden;
            padding: 1.7rem 2rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(15, 118, 110, 0.12);
            border-radius: 20px;
            background:
                radial-gradient(circle at 90% 15%, rgba(45, 212, 191, 0.24), transparent 28%),
                linear-gradient(120deg, #FFFFFF 0%, #EAF7F5 100%);
            box-shadow: 0 12px 30px rgba(16, 42, 67, 0.07);
        }

        .premium-header::after {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 6px;
            background: var(--teal);
        }

        .premium-eyebrow {
            margin-bottom: 0.45rem;
            color: var(--teal);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .premium-title {
            margin: 0;
            color: var(--navy);
            font-size: clamp(1.55rem, 3vw, 2.15rem);
            font-weight: 800;
            letter-spacing: -0.04em;
            line-height: 1.15;
        }

        .premium-subtitle {
            margin-top: 0.55rem;
            max-width: 760px;
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.5;
        }

        /* Section navigation buttons */
        div[data-testid="stHorizontalBlock"] button {
            min-height: 2.7rem;
            border: 1px solid var(--line);
            border-radius: 11px;
            background: var(--white);
            color: var(--navy);
            font-weight: 700;
            transition: all 150ms ease;
        }

        div[data-testid="stHorizontalBlock"] button:hover {
            border-color: var(--teal);
            color: var(--teal);
            box-shadow: 0 4px 12px rgba(15, 118, 110, 0.10);
        }

        div[data-testid="stHorizontalBlock"] button[kind="primary"] {
            border-color: var(--teal);
            background: var(--teal);
            color: white;
        }

        /* KPI cards */
        [data-testid="stMetric"] {
            min-height: 126px;
            padding: 1rem 1.15rem;
            border: 1px solid var(--line);
            border-radius: 16px;
            background: var(--white);
            box-shadow: 0 6px 18px rgba(16, 42, 67, 0.045);
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 700;
        }

        [data-testid="stMetricValue"] {
            color: var(--navy);
            font-size: 1.8rem;
            font-weight: 800;
        }

        /* Headings and text */
        h1, h2, h3, h4 {
            color: var(--navy) !important;
            letter-spacing: -0.025em;
        }

        [data-testid="stMarkdownContainer"] p {
            color: var(--ink);
        }

        /* Buttons */
        .stButton button,
        .stDownloadButton button {
            border-radius: 10px;
            font-weight: 700;
            transition: all 150ms ease;
        }

        .stButton button[kind="primary"],
        .stDownloadButton button {
            border-color: var(--teal);
            background: var(--teal);
            color: white;
        }

        .stButton button[kind="primary"]:hover,
        .stDownloadButton button:hover {
            border-color: #0B5F59;
            background: #0B5F59;
            color: white;
        }

        /* Data tables and alerts */
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            overflow: hidden;
            border: 1px solid var(--line);
            border-radius: 14px;
            background: var(--white);
        }

        [data-testid="stAlert"] {
            border-radius: 12px;
        }

        /* Inputs */
        [data-testid="stSelectbox"] label,
        [data-testid="stSlider"] label,
        [data-testid="stTextInput"] label {
            color: var(--navy);
            font-weight: 700;
        }

        /* Responsive spacing */
        @media (max-width: 768px) {
            .block-container {
                padding: 1rem 1rem 2rem;
            }

            .premium-header {
                padding: 1.3rem 1.25rem;
                border-radius: 15px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# Data loading & Pipelines with Cache
# ==========================================================
@st.cache_data
def load_or_generate_data() -> dict:
    sales_path = Path("data/historical_sales.csv")
    supplier_path = Path("data/suppliers_lead_time.csv")

    if not sales_path.exists() or not supplier_path.exists():
        generate_synthetic_supply_chain_data(
            output_dir="data",
            n_skus=40,
            n_days=730,
            seed=42,
        )

    return process_supply_chain_data(
        sales_path=sales_path,
        supplier_path=supplier_path,
    )


@st.cache_resource
def get_forecast_executor() -> ThreadPoolExecutor:
    """A single shared executor during the Streamlit process lifecycle."""
    return ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="forecast",
    )


@st.cache_data
def compute_inventory_pipeline(processed_data: dict, service_level: float = 0.95) -> pd.DataFrame:
    return build_inventory_plan(processed_data, service_level=service_level)


# ==========================================================
# Render sections
# ==========================================================
def render_overview(inventory_df: pd.DataFrame):
    st.subheader("Executive Overview")

    inventory_df = build_alerts(inventory_df)

    total_skus = inventory_df["sku"].nunique()
    avg_daily_demand = float(inventory_df["avg_daily_demand"].mean())
    annual_demand = float(inventory_df["annual_demand"].sum())
    avg_risk = float(inventory_df["risk_score"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SKUs", total_skus)
    c2.metric("Avg Daily Demand", f"{avg_daily_demand:.1f}")
    c3.metric("Annual Demand", f"{annual_demand:,.0f}")
    c4.metric("Avg Risk Score", f"{avg_risk:.1f}")

    col_a, col_b = st.columns(2)

    abc_counts = (
        inventory_df["abc_class"]
        .value_counts()
        .reindex(["A", "B", "C"], fill_value=0)
        .reset_index()
    )
    abc_counts.columns = ["abc_class", "sku_count"]
    abc_counts["abc_class"] = abc_counts["abc_class"].astype(str)

    if px is not None:
        fig_abc = px.bar(
            abc_counts,
            x="abc_class",
            y="sku_count",
            color="abc_class",
            color_discrete_sequence=["#0F766E", "#22c55e", "#f59e0b"],
            title="ABC Classification",
        )
        fig_abc.update_layout(
            paper_bgcolor="rgba(255, 255, 255, 1)",
            plot_bgcolor="rgba(255, 255, 255, 1)",
            font={"color": "#172B4D"},
            legend={"font": {"color": "#172B4D"}},
        )
        col_a.plotly_chart(fig_abc, use_container_width=True)
    else:
        col_a.dataframe(abc_counts, use_container_width=True)

    risk_summary = build_risk_summary(inventory_df)
    if not risk_summary.empty and px is not None:
        fig_risk = px.bar(
            risk_summary,
            x="alert_level",
            y="total_risk_score",
            color="alert_level",
            color_discrete_map={
                "Green": "#22c55e",
                "Yellow": "#f59e0b",
                "Red": "#ef4444",
                "Verde": "#22c55e",
                "Amarillo": "#f59e0b",
                "Rojo": "#ef4444",
            },
            title="Operational Risk by Alert Level",
        )
        fig_risk.update_layout(
            paper_bgcolor="rgba(255, 255, 255, 1)",
            plot_bgcolor="rgba(255, 255, 255, 1)",
            font={"color": "#172B4D"},
            legend={"font": {"color": "#172B4D"}},
        )
        col_b.plotly_chart(fig_risk, use_container_width=True)
    else:
        col_b.dataframe(risk_summary, use_container_width=True)

    st.markdown("### SKU Operating Plan")
    display_df = inventory_df[
        [
            "sku",
            "product_name",
            "category",
            "abc_class",
            "ved_class",
            "avg_daily_demand",
            "annual_dollar_usage",
            "safety_stock",
            "reorder_point",
            "eoq",
            "days_of_coverage",
            "stockout_probability",
            "alert_level",
            "risk_score",
        ]
    ].sort_values("risk_score", ascending=False)

    st.dataframe(display_df, use_container_width=True, height=420)


def render_forecast_lazy(daily_sales: pd.DataFrame, horizon: int) -> None:
    """Displays the forecast and triggers it only when requested by the user."""
    initialize_forecast_jobs()

    st.subheader("Demand Forecast")
    st.caption(f"Selected Horizon: {horizon} days")

    job_id = build_forecast_job_id(daily_sales, horizon)
    job = get_forecast_job(job_id)

    button_label = (
        "Regenerate Forecast" if job and job["status"] == "READY" else "Generate Forecast"
    )

    if st.button(button_label, key="start_forecast"):
        try:
            submit_forecast_job(
                daily_sales=daily_sales,
                horizon=horizon,
                executor=get_forecast_executor(),
                force=bool(job and job["status"] == "READY"),
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to start forecast: {exc}")

    # Read status back after starting the job
    job = get_forecast_job(job_id)

    if not job:
        st.info("Click «Generate Forecast» to start calculation.")
        return

    status = job["status"]

    if status in {"PENDING", "COMPUTING"}:
        st.info(
            "Forecast is currently being calculated in the background. "
            "You can navigate to other sections and return here."
        )
        if st.button("Refresh Status", key="refresh_forecast_status"):
            st.rerun()
        return

    if status == "FAILED":
        st.error(f"Forecast calculation failed: {job.get('error_message')}")
        st.caption("You can try again using the button above.")
        return

    if status != "READY":
        st.warning(f"Unrecognized forecast status: {status}")
        return

    forecast_df = forecast_dataframe_from_job(job)
    if forecast_df.empty:
        st.warning("The job finished, but there are no results to display.")
        return

    if "sku" not in forecast_df.columns:
        st.error("The forecast result does not contain the 'sku' column.")
        return

    sku_list = sorted(forecast_df["sku"].dropna().astype(str).unique())
    if not sku_list:
        st.warning("No SKUs available in the result.")
        return

    selected_sku = st.selectbox("Select a SKU", sku_list)
    sku_forecast = forecast_df[forecast_df["sku"].astype(str) == selected_sku].copy()

    if "date" in sku_forecast.columns:
        sku_forecast["date"] = pd.to_datetime(
            sku_forecast["date"],
            errors="coerce",
        )
        sku_forecast = sku_forecast.dropna(subset=["date"]).sort_values("date")

    if sku_forecast.empty:
        st.warning("No forecast available for this SKU.")
        return

    chart_columns = [
        column
        for column in ("forecast_units", "lower_bound", "upper_bound")
        if column in sku_forecast.columns
    ]

    if "date" in sku_forecast.columns and chart_columns:
        if px is not None:
            fig = px.line(
                sku_forecast,
                x="date",
                y=chart_columns,
                title=f"Forecast for {selected_sku}",
                labels={"date": "Date", "value": "Units"},
            )
            fig.update_layout(
                paper_bgcolor="rgba(255, 255, 255, 1)",
                plot_bgcolor="rgba(255, 255, 255, 1)",
                font={"color": "#172B4D"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(sku_forecast.set_index("date")[chart_columns])

    display_columns = [
        column
        for column in (
            "date",
            "sku",
            "forecast_units",
            "lower_bound",
            "upper_bound",
            "selected_model",
        )
        if column in sku_forecast.columns
    ]
    st.dataframe(
        sku_forecast[display_columns].tail(30),
        use_container_width=True,
    )


def render_inventory(inventory_df: pd.DataFrame):
    st.subheader("Inventory & Optimization")

    inventory_df = build_alerts(inventory_df)

    display_cols = [
        "sku",
        "product_name",
        "category",
        "abc_class",
        "ved_class",
        "avg_daily_demand",
        "annual_dollar_usage",
        "safety_stock",
        "reorder_point",
        "eoq",
        "days_of_coverage",
        "stockout_probability",
        "alert_level",
        "risk_score",
    ]

    st.dataframe(
        inventory_df[display_cols].sort_values("risk_score", ascending=False),
        use_container_width=True,
        height=520,
    )

    summary = (
        inventory_df.groupby("category", as_index=False)
        .agg(
            sku_count=("sku", "nunique"),
            avg_daily_demand=("avg_daily_demand", "mean"),
            total_safety_stock=("safety_stock", "sum"),
            total_reorder_point=("reorder_point", "sum"),
            avg_stockout_probability=("stockout_probability", "mean"),
        )
        .sort_values("avg_stockout_probability", ascending=False)
    )
    st.dataframe(summary, use_container_width=True)


def render_risk(alerts_df: pd.DataFrame):
    st.subheader("Operational Alerts")

    risk_summary = build_risk_summary(alerts_df)
    if not risk_summary.empty:
        st.dataframe(risk_summary, use_container_width=True)

    st.markdown("### Open Alerts (Persisted)")
    persisted_alerts = load_open_risk_alerts(scope="default")
    if persisted_alerts.empty:
        st.success("No open alerts.")
    else:
        if "provisional" in persisted_alerts.columns:
            persisted_alerts["provisional"] = persisted_alerts["provisional"].map(
                {0: "No", 1: "Yes"}
            )
        st.dataframe(
            persisted_alerts,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### SKU Operating Plan Alerts")
    st.dataframe(
        alerts_df[
            [
                "sku",
                "product_name",
                "category",
                "abc_class",
                "ved_class",
                "risk_score",
                "stockout_probability",
                "days_of_coverage",
                "alert_level",
                "priority_label",
            ]
        ].sort_values("risk_score", ascending=False),
        use_container_width=True,
        height=400,
    )


def render_scenarios(inventory_df: pd.DataFrame):
    st.subheader("Scenario Analysis")

    col1, col2 = st.columns(2)
    demand_change_pct = (
        col1.slider(
            "Demand Change (%)",
            min_value=-50,
            max_value=50,
            value=20,
            step=5,
        )
        / 100.0
    )

    lead_time_change_days = col2.slider(
        "Lead Time Change (days)",
        min_value=-10,
        max_value=20,
        value=3,
        step=1,
    )

    scenario_df = evaluate_what_if_scenario(
        inventory_df,
        demand_change_pct=demand_change_pct,
        lead_time_change_days=lead_time_change_days,
        service_level=0.95,
    )

    st.dataframe(
        scenario_df[
            [
                "sku",
                "product_name",
                "avg_daily_demand",
                "scenario_avg_daily_demand",
                "avg_lead_time_days",
                "scenario_lead_time_days",
                "scenario_reorder_point",
                "scenario_risk_score",
                "scenario_alert_level",
            ]
        ].sort_values("scenario_risk_score", ascending=False),
        use_container_width=True,
        height=500,
    )


def render_recommendations(inventory_df: pd.DataFrame) -> None:
    st.subheader("Action Recommendations")
    st.caption(
        "Decision support only: review the data and approve actions "
        "according to your operational process."
    )

    recommendations_df = build_recommendations(inventory_df)

    if recommendations_df.empty:
        st.info("No recommendations are available.")
        return

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    urgency_options = ["All"] + sorted(recommendations_df["urgency"].dropna().unique().tolist())
    selected_urgency = filter_col1.selectbox(
        "Urgency",
        urgency_options,
        key="recommendations_urgency",
    )

    sku_options = ["All"] + sorted(recommendations_df["sku"].dropna().astype(str).unique().tolist())
    selected_sku = filter_col2.selectbox(
        "SKU",
        sku_options,
        key="recommendations_sku",
    )

    supplier_options = ["All"] + sorted(
        recommendations_df["supplier_code"]
        .fillna("")
        .astype(str)
        .replace("", "Unknown")
        .unique()
        .tolist()
    )
    selected_supplier = filter_col3.selectbox(
        "Supplier",
        supplier_options,
        key="recommendations_supplier",
    )

    filtered = recommendations_df.copy()

    if selected_urgency != "All":
        filtered = filtered[filtered["urgency"] == selected_urgency]

    if selected_sku != "All":
        filtered = filtered[filtered["sku"].astype(str) == selected_sku]

    if selected_supplier != "All":
        supplier_series = filtered["supplier_code"].fillna("").astype(str).replace("", "Unknown")
        filtered = filtered[supplier_series == selected_supplier]

    high_count = int((filtered["urgency"] == "HIGH").sum())
    medium_count = int((filtered["urgency"] == "MEDIUM").sum())

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Recommendations", len(filtered))
    metric2.metric("High urgency", high_count)
    metric3.metric("Medium urgency", medium_count)

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download recommendations (.csv)",
        data=csv_data,
        file_name="supply_chain_recommendations.csv",
        mime="text/csv",
        key="download_recommendations",
    )


def render_export(alerts_df: pd.DataFrame):
    st.subheader("Export")

    export_df = alerts_df[
        [
            "sku",
            "product_name",
            "category",
            "abc_class",
            "ved_class",
            "avg_daily_demand",
            "safety_stock",
            "reorder_point",
            "eoq",
            "stockout_probability",
            "risk_score",
            "alert_level",
            "priority_label",
        ]
    ]

    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Inventory and Alerts (.csv)",
        data=csv_data,
        file_name="inventory_alerts.csv",
        mime="text/csv",
    )


# ==========================================================
# Main flow
# ==========================================================
def main():
    processed_data = load_or_generate_data()
    daily_sales = processed_data["daily_sales"]

    st.sidebar.title("Configuration")
    st.sidebar.caption("Supply Chain Monitoring")

    forecast_horizon = st.sidebar.slider(
        "Forecast Horizon (days)",
        min_value=7,
        max_value=180,
        value=30,
        step=7,
    )

    service_level = st.sidebar.slider(
        "Target Service Level",
        min_value=0.80,
        max_value=0.99,
        value=0.95,
        step=0.01,
    )

    if "active_section" not in st.session_state:
        st.session_state.active_section = "Overview"

    # Control Tower header (v2.0)
    st.markdown(
        """
        <div class="premium-header">
            <div class="premium-eyebrow">Supply chain intelligence · v2.0</div>
            <div class="premium-title">Plan with clarity. Respond with confidence.</div>
            <div class="premium-subtitle">
                Demand, inventory and operational risk in one decision-support workspace.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sections = [
        "Overview",
        "Forecast",
        "Inventory",
        "Risk",
        "Scenarios",
        "Recommendations",
        "Export",
    ]
    cols = st.columns(len(sections))

    for i, item in enumerate(sections):
        with cols[i]:
            is_active = st.session_state.active_section == item
            btn_type = "primary" if is_active else "secondary"
            if st.button(item, use_container_width=True, type=btn_type):
                st.session_state.active_section = item
                st.rerun()

    active_section = st.session_state.active_section

    st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)

    raw_inventory_df = compute_inventory_pipeline(processed_data, service_level=service_level)
    inventory_df = build_alerts(raw_inventory_df)
    alerts_df = inventory_df

    # Sprint 3: Risk alerts assessment and persistence
    risk_alerts_df = build_risk_alerts(
        inventory_df,
        scope="default",
    )
    persist_risk_alerts(
        risk_alerts_df,
        scope="default",
    )

    if active_section == "Overview":
        render_overview(inventory_df)
    elif active_section == "Forecast":
        render_forecast_lazy(daily_sales, forecast_horizon)
    elif active_section == "Inventory":
        render_inventory(inventory_df)
    elif active_section == "Risk":
        render_risk(alerts_df)
    elif active_section == "Scenarios":
        render_scenarios(inventory_df)
    elif active_section == "Recommendations":
        render_recommendations(inventory_df)
    elif active_section == "Export":
        render_export(alerts_df)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
