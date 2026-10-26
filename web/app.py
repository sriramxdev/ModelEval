"""Small read-only telemetry dashboard with no ML runtime dependency."""

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Since app.py is in web/, parents[1] points to the main project root (ModelEval/)
ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "data" / "results.json"
app = FastAPI(title="Model Evaluation Telemetry")

# Mount the data folder using the absolute project root path
app.mount("/data", StaticFiles(directory=str(ROOT / "data")), name="data")

# Point Jinja2 directly to the project root directory where index.html resides
templates = Jinja2Templates(directory=str(ROOT))


def read_results() -> dict[str, Any]:
    try:
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"status": "PENDING", "models": [], "errors": [], "generated_at": None}


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"results": read_results()},
    )


@app.get("/api/results")
async def api_results() -> dict[str, Any]:
    return read_results()