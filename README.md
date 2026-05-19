# SemDisc Demo

SemDisc is now split into a Flask API backend and a Vue 3 frontend.

## Backend

```bash
cd backend
python -m pip install -r requirements.txt
python app.py
```

The API runs on `http://127.0.0.1:5000` by default. Key endpoints include:

- `GET /api/health`
- `GET /api/demo/datalake`
- `POST /api/preprocess`
- `GET /api/datasets/<dataset_id>/graph`
- `GET /api/datasets/<dataset_id>/paths`
- `GET /api/demo/query`
- `POST /api/query`
- `POST /api/materialize`

## Frontend

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server runs on `http://127.0.0.1:5173` and proxies `/api` requests to Flask.

## Tests

```bash
cd backend
pytest

cd ../frontend
npm test
```

## Project Layout

```text
backend/      Flask app, API blueprints, SemDisc domain logic, backend tests
frontend/     Vite/Vue app, workflow views, components, frontend tests
datalake/     Bundled CSV demo data
plan.md       Original migration plan
```
