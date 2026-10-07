# IntelliSum Frontend

React + Vite + Tailwind CSS dashboard for IntelliSum.

## Requirements

- Node.js 20.19+ or 22.12+ (developed on Node 22)

## Setup and run

From the `frontend/` directory:

```bash
npm install
npm run dev
```

Open <http://localhost:5173>. The backend must be running on port 8000. During development, Vite proxies
every `/api/*` request to `http://127.0.0.1:8000` (see `vite.config.js`), so no CORS setup is needed.
To point the proxy at a different backend, set `VITE_BACKEND_URL` before `npm run dev`.

## Build

```bash
npm run build     # outputs to dist/
npm run preview   # serve the production build locally
```

For a production build served separately from the API, set `VITE_API_BASE_URL` (e.g. `http://127.0.0.1:8000`)
at build time and add the frontend origin to the backend's `INTELLISUM_CORS_ORIGINS`.

## Structure

```
src/
├── components/   Reusable UI pieces (status pill, upload zone, stat cards …)
├── pages/        Page-level views
├── services/     Axios API client
└── utils/        Formatting helpers, local history
```
