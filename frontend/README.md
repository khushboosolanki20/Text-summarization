# IntelliSum Frontend

React + Vite + Tailwind CSS dashboard for IntelliSum.

## Requirements

- Node.js 20.19+ or 22.12+ (developed on Node 22)
- The backend running on port 8000 (see `../backend/README.md`)

## Setup and run

From the `frontend/` directory:

```bash
npm install
npm run dev
```

Open <http://localhost:5173>. During development, Vite proxies every `/api/*` request to
`http://127.0.0.1:8000` (see `vite.config.js`), so no CORS setup is needed. To point the proxy at a different
backend, set `VITE_BACKEND_URL` before `npm run dev`.

## Build

```bash
npm run lint      # oxlint
npm run build     # outputs to dist/
npm run preview   # serve the production build locally
```

For a production build served separately from the API, set `VITE_API_BASE_URL` (e.g. `http://127.0.0.1:8000`)
at build time and add the frontend origin to the backend's `INTELLISUM_CORS_ORIGINS`.

## Pages

| Route | Page |
|---|---|
| `/` | **Summarize**: paste text or drag-and-drop a PDF/DOCX/TXT, choose method and length, optional reference summary, live progress, results |
| `/history` | Last 25 summaries, stored only in this browser (`localStorage`) |
| `/about` | How the four methods and the evaluation work |

## How a summary is requested

1. The form validates input in the browser (non-empty, minimum length, file type and size from `GET /api/config`).
   Choosing BART or Hybrid calls `POST /api/warmup` so the model starts loading early.
2. `POST /api/summarize/{text|file}/async` starts a job; `GET /api/jobs/{id}` is polled every 0.7 s and its
   `progress` (stage, done/total) drives the progress bar. Cancel stops polling.
3. The result is shown and added to the local history.

## Results view

- **Original vs summary**: word counts, compression meter, stat tiles (time, sentences selected or chunks, ROUGE).
- **Summary**: copy, download `.txt`; for BART/Hybrid, sentences flagged by the faithfulness check are marked.
- **Potentially unsupported content (experimental)**: flagged sentences, reasons, closest source passage.
- **Evaluation (ROUGE)**: grouped bar chart + table. Without a reference, an explanation and an inline box to
  score the summary with `POST /api/evaluate` (no re-run).
- **How it was produced**: LangGraph path, time per step, model/device/tokens, chunk summaries with page ranges.
- **Why these sentences?** (TF-IDF, TextRank, Hybrid): method explanation (top keywords / graph statistics / input
  reduction), a per-sentence importance chart, and the document with selected sentences highlighted, plus scores
  and page numbers.

The results view and the charting library (Recharts) are lazy-loaded, so the initial page stays light.

### Chart colors

Charts use a palette validated for color-vision deficiency and contrast (`src/utils/chartTheme.js`):
categorical slots for precision / recall / F1, two steps of one blue ramp for "selected vs other", a legend
whenever there are two or more series, tooltips on hover, and a table view for ROUGE (needed because one series
color is below 3:1 contrast).

## Structure

```
src/
├── components/
│   ├── input/      SummarizeForm, FileDropzone, OptionGroup (accessible radio cards / segmented control)
│   ├── results/    ResultView and its panels, charts.jsx (Recharts)
│   ├── ui/         Icon (inline SVG), Card, Button, Banner, Badge, Stat, Collapsible
│   ├── BackendStatus.jsx
│   └── ProgressPanel.jsx
├── hooks/          useSummarize (job lifecycle), useConfig
├── pages/          SummarizePage, HistoryPage, AboutPage
├── services/       api.js (Axios client, job polling, readable errors)
└── utils/          format, history (localStorage), download, validation, chartTheme
```
