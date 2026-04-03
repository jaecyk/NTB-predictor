# Live Data + Database + Frontend Plan

This repo can be upgraded from a single-file Streamlit app into a small production-style stack.

## Target architecture

### 1) Data ingestion layer
Responsible for pulling fresh market and auction data into the system.

Suggested input groups:
- auction inputs by tenor
- prior stop rates by tenor
- secondary market tenor proxy rates
- system liquidity snapshot
- macro snapshots such as MPR and inflation

This layer should run on a schedule and write every new snapshot into the database.

### 2) Database layer
Recommended production database: **PostgreSQL**

Local development fallback: **SQLite**

Core tables:
- `market_snapshots`
- `prediction_runs`

`market_snapshots` stores the latest observed auction and market input values.
`prediction_runs` stores what the model predicted and when.

### 3) Backend API
Recommended backend: **FastAPI**

Responsibilities:
- expose health endpoint
- expose latest market snapshot
- expose prediction endpoint
- expose prediction history endpoint
- accept controlled manual input overrides when needed

### 4) Frontend
Current fastest path: **Streamlit** as the frontend

The frontend should no longer be the system of record.
Instead it should:
- read latest data from the backend
- trigger predictions through the backend
- show latest prediction and history
- allow manual override inputs for scenario testing

## Suggested folder structure

```text
backend/
  db.py
  models.py
  schemas.py
  predictor.py
  main.py
frontend/
  app_live.py
docs/
  LIVE_STACK_PLAN.md
requirements_live.txt
```

## Environment variables

```env
DATABASE_URL=sqlite:///./ntb_live.db
# production example
# DATABASE_URL=postgresql+psycopg2://user:password@host:5432/ntb_live
```

## Live data reality

A real live system still needs approved source connectors.
This scaffold gives you the stack.
You will still need to connect actual market/auction feeds or a scheduled ETL process.

## Recommended rollout order

1. Stand up the backend with SQLite locally
2. Store manual snapshots and validate predictions
3. Swap SQLite for PostgreSQL in deployment
4. Add scheduled ingestion jobs
5. Point the Streamlit frontend to the backend API

## What this repo change adds

- backend API scaffold
- database models
- prediction service wrapping your existing pickle models
- a frontend app wired to the backend

## What it does not yet add

- real source credentials
- a production scheduler
- authentication
- hosted PostgreSQL instance

Those are deployment decisions and access-controlled items.
