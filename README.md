<div align="center">

# 📦 Supply Chain Intelligence

### Plan with clarity. Respond with confidence.

An AI-powered MVP for exploring demand forecasts, inventory plans, and operational risk — with recommendations that help teams decide what to review next.

<br />

[![Version](https://img.shields.io/badge/version-2.0.0-0F766E?style=for-the-badge)](https://github.com/cp-core-labs/supply_chain_microapp_v2)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br />

<div align="center">
  <a href="https://supply-chain-microapp.streamlit.app/">
    <img
      src="https://img.shields.io/badge/🚀_Open_Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"
      alt="Open the live demo"
    />
  </a>
</div>

<br />

[Features](#-what-it-does) · [Getting Started](#-getting-started) · [Architecture](#-how-it-works) · [Limitations](#-project-status-and-limitations)

</div>

---

## A clearer view of inventory decisions

Supply Chain Intelligence brings demand planning, inventory analysis, operational risk, and decision-support recommendations into one Streamlit application.

Use the included sample data to explore SKU-level forecasts, replenishment indicators, risk alerts, and what-if scenarios. Recommendations are intended to guide review — not to place orders or replace operational judgment.

> **This project is an MVP decision-support and analytics tool for exploring demand forecasting, inventory planning, and operational risk. It is not yet a production-ready, multi-tenant platform or a replacement for an ERP system.**

## ✨ What it does

| Capability | Description |
| --- | --- |
| **Demand forecasting** | Generate SKU-level forecasts when requested, with forecast results saved for reuse. |
| **Inventory planning** | Explore ABC/VED classification, safety stock, reorder points, EOQ estimates, and coverage indicators. |
| **Operational risk** | Review risk scores and alerts for potential stockout, low coverage, lead-time, and data-quality concerns. |
| **Confidence signals** | See basic data-quality assessments alongside risk alerts to support cautious interpretation. |
| **Recommendations** | Review suggested next steps, such as validating data or reviewing replenishment and supplier policies. |
| **Scenario analysis** | Explore how demand and lead-time changes may affect inventory planning indicators. |
| **Exports** | Download selected planning and recommendation data as CSV. |
| **Updated interface** | Navigate an executive overview and focused sections in a refreshed control-tower style UI. |

## 🧭 How it works

1. The app loads the included sales and supplier sample data, generating synthetic sample data if the expected CSV files are unavailable.
2. It processes sales and supplier information and builds an inventory plan.
3. Risk rules and confidence checks produce alerts for review.
4. Forecast calculations can be started on demand.
5. The user explores recommendations, scenarios, and downloadable outputs.

### Technology

- **Application:** Python, Streamlit
- **Data processing:** Pandas, NumPy
- **Forecasting and analytics:** Scikit-learn, Statsmodels, Prophet, XGBoost
- **Charts:** Plotly
- **Local persistence:** SQLite
- **Quality checks:** Pytest, Ruff, Black

## 🚀 Getting Started

### Requirements

- Python 3.10 or later
- Git

### Run locally

```bash
git clone https://github.com/cp-core-labs/supply_chain_microapp_v2.git
cd supply_chain_microapp_v2

python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies and start the app:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL in the terminal, typically `http://localhost:8501`.

### Run tests and quality checks

Install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

Then run:

```bash
pytest -q
ruff check .
black --check .
```

## 🗂️ Project Structure

```text
supply_chain_microapp_v2/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .streamlit/
├── assets/
├── data/
│   ├── db/
│   ├── historical_sales.csv
│   └── suppliers_lead_time.csv
├── modules/
│   ├── data_processor.py
│   ├── forecast_jobs.py
│   ├── forecaster.py
│   ├── inventory_optimizer.py
│   ├── recommendation_engine.py
│   ├── report_exporter.py
│   ├── risk_engine.py
│   ├── risk_monitor.py
│   └── utils.py
├── scripts/
├── src/
│   └── supply_chain_microapp/
│       ├── __init__.py
│       ├── config.py
│       ├── repositories.py
│       └── storage.py
├── tests/
│   ├── test_data_processor.py
│   ├── test_forecaster.py
│   ├── test_inventory_optimizer.py
│   └── test_risk_engine.py
├── app.py
├── data_generator.py
├── Dockerfile
├── pyproject.toml
├── requirements-dev.txt
└── requirements.txt
```

## 🧱 How it works

The code is organized around a Streamlit app, analytical modules, and a small SQLite persistence layer:

- `app.py` provides the user interface and connects the analysis workflows.
- `modules/` contains data processing, forecasting, inventory, risk, and recommendation logic.
- `src/supply_chain_microapp/` contains application configuration and SQLite storage/repository helpers.
- `data/` contains the sample CSV inputs and the local database directory.
- `tests/` contains automated tests for core processing and analytical modules.

## ⚠️ Project status and limitations

This is an **MVP for exploration and decision support**, not a production-ready multi-user service.

- SQLite data is stored locally; persistence depends on the environment retaining the database file.
- Forecast work runs in a thread managed by the Streamlit process. It is not a distributed job queue and may not survive a process restart.
- The app does not currently provide tenant isolation, user roles, or enterprise authentication.
- Recommendations are suggestions for human review. They do not create purchase orders or execute operational actions.
- Forecast and risk outputs depend on the quality, completeness, and freshness of the input data.

Do not use the sample data or unvalidated recommendations as the sole basis for live purchasing or other critical operational decisions.

## 🗺️ Roadmap

Possible future work, based on user feedback:

- improve forecast and recommendation validation
- expand automated tests for persistence, alerts, and recommendations
- refine data import and validation workflows
- evaluate production hosting and multi-user requirements when there is a concrete deployment need

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

**Built as an MVP to make supply-chain analysis more actionable.**

</div>