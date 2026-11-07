<div align="left">
  <a href="https://github.com/cp-core-labs/supply_chain_microapp_v2/releases">
    <img src="https://img.shields.io/badge/Release-v2.0.0-orange?style=for-the-badge&logo=github&logoColor=white" alt="Release v2.0.0" />
  </a>

  <a href="https://github.com/cp-core-labs/supply_chain_microapp_v2/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License" />
  </a>
</div>

<h1 align="left">
  <img src="https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/1f680.svg" width="42" alt="rocket" />
  Supply Chain & Inventory Intelligence Platform v2.0
</h1>

<p align="left">
  AI-powered supply chain intelligence platform with persistent inventory data, on-demand demand forecasting, risk alerts, and actionable replenishment recommendations.
</p>

<p align="left">
  This project is an MVP decision-support and analytics tool for exploring demand forecasting, inventory planning, and operational risk. It is designed to support better planning decisions, not to replace a full ERP system.
</p>

<div align="left">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
  </a>

  <a href="https://scikit-learn.org/stable/">
    <img src="https://img.shields.io/badge/ML_Engine-Scikit%20Learn-purple?style=for-the-badge" alt="ML Engine" />
  </a>

  <a href="https://facebook.github.io/prophet/">
    <img src="https://img.shields.io/badge/Forecasting-Prophet-9cf?style=for-the-badge" alt="Prophet" />
  </a>
</div>

---

## Overview

This project helps teams explore:
- historical sales and demand patterns
- SKU-level forecasting
- inventory optimization
- ABC/VED segmentation
- safety stock and reorder policy estimation
- operational risk evaluation
- scenario analysis
- recommendations for review and action

This is an MVP decision-support and analytics tool for demand planning and inventory intelligence. It is not yet a production-ready multi-tenant platform or a replacement for an ERP system.

## Highlights of v2.0.0

- persistent storage with SQLite
- on-demand forecasting instead of blocking the UI
- risk monitoring with confidence scoring
- actionable recommendations for review
- improved interface design for a more operational control-tower feeling
- better organization of logic into data, forecasting, risk, and recommendation modules

## Main Features

- **On-demand forecasting:** forecast heavy calculations are started only when the user requests them.
- **Persistent state:** data and forecasting results are stored locally to improve continuity across sessions.
- **Risk monitoring:** detect operational risk, stockout risk, and low coverage areas with confidence scoring.
- **Actionable recommendations:** suggest review actions such as validating data, reviewing replenishment, or monitoring supplier risk.
- **Inventory planning:** ABC/VED segmentation, safety stock, EOQ estimation, reorder points and scenario simulation.
- **CSV export:** export planning output and key operational signals for downstream workflows.

## Tech Stack

- **Core Engine & Data Processing:** Python, Pandas, NumPy
- **Forecasting & ML:** Scikit-learn, Statsmodels, Prophet, XGBoost
- **Dashboard & Visualization:** Streamlit, Plotly
- **Persistence:** SQLite
- **Quality & Validation:** Pytest, Ruff, Black, Mypy

## Project Structure

```text
supply_chain_microapp_v2/
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
├── LICENSE
├── README.md
├── app.py
├── data/
│   ├── historical_sales.csv
│   ├── suppliers_lead_time.csv
│   └── db/
├── modules/
│   ├── __init__.py
│   ├── data_processor.py
│   ├── forecaster.py
│   ├── forecast_jobs.py
│   ├── inventory_optimizer.py
│   ├── recommendation_engine.py
│   ├── risk_engine.py
│   ├── risk_monitor.py
│   └── utils.py
├── src/
│   └── supply_chain_microapp/
│       ├── config.py
│       ├── init.py
│       ├── repositories.py
│       └── storage.py
├── tests/
│   ├── test_data_processor.py
│   ├── test_forecaster.py
│   ├── test_inventory_optimizer.py
│   └── test_risk_engine.py
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── pytest.ini
├── Dockerfile
└── .github/
    └── workflows/
        └── ci.yml
```

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Development Setup

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
black --check .
```

## Notes

This project is intended as a decision-support MVP for inventory and supply chain analysis. It does not yet represent a production multi-tenant platform, a full ERP replacement, or an enterprise integration layer.

The forecasting and risk outputs should be reviewed with operational context before any business action is taken. Data quality, lead time assumptions, and supplier information directly affect the reliability of the recommendations.

## License

MIT