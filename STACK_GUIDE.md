# NTB Predictor Stack Guide

This guide maps the app into three practical stages:
- local build
- free hosted build
- production build

The goal is to help move from a single-user prototype to an internal treasury tool with live data, storage, and a usable front end.

---

## 1) Local Build

### Best use case
Use this when you want to build fast, test logic, and validate the prediction flow on your laptop.

### Stack
- **Frontend:** Streamlit
- **Backend:** FastAPI
- **Database:** SQLite
- **Model storage:** local `.pkl` files in the repo

### Why this is the right local stack
- zero setup cost
- fastest development path
- no cloud complexity
- easiest place to debug model loading and feature calculations

### Recommended files
- `frontend_app_live.py`
- `backend_main.py`
- `backend_db.py`
- `backend_models.py`
- `backend_predictor.py`

### Environment
```env
DATABASE_URL=sqlite:///./ntb_live.db
API_BASE_URL=http://localhost:8000
```

### Run locally
Backend:
```bash
uvicorn backend_main:app --reload
```

Frontend:
```bash
streamlit run frontend_app_live.py
```

### What you get
- manual snapshot entry
- prediction history
- local DB file
- ability to test prediction workflow end to end

### Limitation
This is not true live infrastructure. It is only the best local working base.

---

## 2) Free Hosted Build

### Best use case
Use this when you want a working online MVP with low or zero cost.

### Recommended stack
- **Frontend:** Streamlit Community Cloud or another low-cost Streamlit host
- **Backend:** Render / Railway / Fly-style free or starter FastAPI hosting
- **Database:** PostgreSQL on a free-tier provider such as Supabase or Neon
- **Model storage:** keep `.pkl` files in repo or in controlled object storage later

### Why this is the best free-first setup
- Streamlit remains fast for internal analytics tools
- PostgreSQL is much better than SQLite once hosted
- FastAPI gives a clean API layer between data, model logic, and UI
- you do not waste time building a React app too early

### Environment
```env
DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME
API_BASE_URL=https://your-backend-service-url
```

### Suggested free-hosted architecture
1. frontend calls backend API
2. backend reads latest market snapshots from PostgreSQL
3. backend runs the saved model files
4. backend stores prediction history in PostgreSQL
5. frontend displays latest snapshots and results

### What you get
- hosted internal MVP
- central DB
- shared access for your team
- history preserved outside your laptop

### Limitation
Free tiers can sleep, throttle, or change limits. This setup is for MVP and internal testing, not guaranteed mission-critical uptime.

---

## 3) Production Build

### Best use case
Use this when you want an internal GTI-grade tool with stronger reliability, controlled access, and room for scale.

### Recommended stack
- **Frontend:** Next.js or React app
- **Backend:** FastAPI
- **Database:** PostgreSQL
- **Background jobs:** scheduled ingestion and sync workers
- **Model storage:** versioned model artifacts in controlled storage
- **Hosting:** managed cloud services with secrets, logs, monitoring, and backups

### Why this is the production path
- cleaner user experience
- better role-based access and authentication options
- more control over API, audit trail, and scaling
- easier to separate market data ingestion from user interactions

### Production additions
- auth and user roles
- audit logging
- scheduled ETL jobs
- monitoring and alerts
- backups
- model version control
- staging and production environments

### Production architecture
1. ingestion job pulls market and auction inputs on schedule
2. backend validates and stores them in PostgreSQL
3. model service runs predictions
4. backend writes results and history
5. frontend shows latest rates, predictions, scenarios, and logs

### What you get
- stronger reliability
- cleaner UI
- enterprise-friendly structure
- easier scaling and governance

### Limitation
This takes more time and more setup than the free-hosted MVP path.

---

## What I recommend for this repo right now

### Immediate next step
Use the **free hosted build** path.

### Exact recommendation
- **Frontend:** Streamlit
- **Backend:** FastAPI
- **Database:** PostgreSQL

### Why
This gives the best balance of speed, cost, and upgrade path.
It is enough for:
- live data feed integration
- prediction history
- internal user testing
- scenario overrides
- future migration to a more polished UI later

---

## Practical path for you

### Phase 1
Build and validate locally with SQLite.

### Phase 2
Move DB to hosted PostgreSQL and deploy backend + frontend.

### Phase 3
Add live data ingestion and scheduled updates.

### Phase 4
Upgrade frontend only if you outgrow Streamlit.

---

## Simple decision table

| Need | Best choice |
|---|---|
| Fastest build | Streamlit + FastAPI + SQLite |
| Free hosted MVP | Streamlit + FastAPI + PostgreSQL |
| Long-term production | Next.js/React + FastAPI + PostgreSQL |

---

## Final recommendation

For this NTB predictor, do **not** start with a heavy frontend.
Start with:
- **Streamlit frontend**
- **FastAPI backend**
- **PostgreSQL database**

Then upgrade the frontend only after the live data and DB workflow is stable.
