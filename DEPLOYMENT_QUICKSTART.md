# Deployment Quickstart

This guide prepares the app for a shareable link so other users can open the frontend in their browser and submit inputs.

## Deployment shape
- **Frontend:** Streamlit app
- **Backend:** FastAPI app
- **Database:** hosted PostgreSQL

## Files added for hosting
- `backend_main_hosted.py`
- `requirements_backend.txt`
- `requirements_frontend.txt`
- `streamlit_app.py`
- `hosted_env.example`

## 1) Backend deployment
Deploy the backend as a Python web service on your preferred host.

### Build command
```bash
pip install -r requirements_backend.txt
```

### Start command
```bash
uvicorn backend_main_hosted:app --host 0.0.0.0 --port $PORT
```

### Required backend environment variables
- `DATABASE_URL`
- `ALLOWED_ORIGINS`

Example:
```env
DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME
ALLOWED_ORIGINS=https://your-frontend-url.streamlit.app
```

## 2) Database deployment
Use a hosted PostgreSQL database and copy its connection string into `DATABASE_URL`.

The backend already uses SQLAlchemy and will create the tables on startup.

## 3) Frontend deployment
Deploy the Streamlit frontend using:
- main file: `streamlit_app.py`
- Python dependencies file: `requirements_frontend.txt`

### Required frontend secret or environment variable
- `API_BASE_URL`

Example:
```env
API_BASE_URL=https://your-backend-service-url
```

## 4) Why `backend_main_hosted.py` exists
This wrapper adds CORS handling so the hosted frontend can call the hosted backend across domains.

## 5) After deployment
Test these in order:

### Backend health
```text
https://your-backend-service-url/health
```

### Frontend
```text
https://your-frontend-url.streamlit.app
```

## 6) Shareable workflow
Once both are deployed:
1. users open the frontend link
2. they submit snapshots or use your ingestion flow
3. frontend calls the backend API
4. backend stores data in PostgreSQL
5. predictions and history are shared across users

## 7) Recommended first security step
Before sharing widely, restrict backend CORS to only your frontend URL using `ALLOWED_ORIGINS`.

## 8) Recommended next upgrade
Add authentication before broad internal rollout if multiple users will be entering data.
