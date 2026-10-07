# IntelliSum Backend

FastAPI service that runs the IntelliSum summarization workflow (TF-IDF, TextRank, BART, Hybrid).

## Requirements

- **Python 3.11 or newer** (developed and tested on 3.14, 64-bit).
  On Windows, check which interpreters you have with `py -0`. A plain `python` command may point to an older
  version, so create the virtual environment with the launcher (`py -3.14`) as shown below.
- ~2 GB free disk for the BART model (`facebook/bart-large-cnn`, downloaded automatically on first use).
- Optional: an NVIDIA GPU. Everything runs on CPU, just more slowly for BART/Hybrid.

## Setup

All commands are run from the `backend/` directory.

### 1. Create and activate a virtual environment

Windows (PowerShell):

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If PowerShell blocks the activation script, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Windows (Git Bash):

```bash
py -3.14 -m venv .venv
source .venv/Scripts/activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install PyTorch

Install PyTorch **before** the other requirements so you get the right build.

- **NVIDIA GPU (CUDA 12.x driver, e.g. driver 537+):**
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu126
  ```
- **CPU only:**
  ```bash
  pip install torch
  ```

See <https://pytorch.org/get-started/locally/> for other platforms.

> **GPU not used?** Check `GET /api/health`: `gpu_problem` explains why. A common cause is an NVIDIA driver too
> old for the CUDA version PyTorch was built with (`torch.cuda.is_available()` is `True` but every operation
> fails with *"CUDA-capable device(s) is/are busy or unavailable"*). Update the driver (version 560+ for CUDA
> 12.6) and restart. Until then the app automatically runs BART on the CPU.

The BART model (`facebook/bart-large-cnn`, ~1.6 GB) is downloaded automatically into the Hugging Face cache
the first time an abstractive summary is requested.

### 3. Install the remaining dependencies

```bash
pip install -r requirements-dev.txt      # runtime + pytest/httpx
python -m spacy download en_core_web_sm  # English sentence segmentation model
```

(`requirements.txt` alone is enough for running the server without tests.)

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

- Health check: <http://127.0.0.1:8000/api/health>
- Swagger UI: <http://127.0.0.1:8000/docs>

## Test

```bash
pytest                 # all tests
pytest -m "not slow"   # skip tests that load the BART model
```

## Configuration

Settings are defined in `app/config.py` and can be overridden with environment variables prefixed with
`INTELLISUM_` or a `backend/.env` file, e.g.:

```
INTELLISUM_DEVICE=cpu
INTELLISUM_MAX_UPLOAD_MB=5
```

## Package layout

```
app/
├── main.py           FastAPI app factory, CORS, error handling
├── config.py         Settings (env-overridable)
├── api/              HTTP routers
├── core/             Service layer between API and workflow
├── graph/            LangGraph state + workflow
├── summarizers/      TF-IDF, TextRank, BART, Hybrid
├── preprocessing/    Cleaning, sentence splitting, chunking
├── documents/        TXT / PDF / DOCX loaders
├── evaluation/       ROUGE + statistics
└── utils/            Shared helpers
```
