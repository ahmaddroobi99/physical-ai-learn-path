"""Local teaching server: one page, one function per lesson."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from physical_ai_lab.studio import lesson_catalog, run_lesson

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Physical AI Lab")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/lessons")
def lessons() -> list[dict]:
    return lesson_catalog()


@app.post("/api/lessons/{lesson_id}/run")
def run(lesson_id: str, body: dict | None = None) -> JSONResponse:
    raw = body or {}
    params = raw.get("params") if isinstance(raw, dict) else None
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return JSONResponse({"error": "params must be an object."}, status_code=400)
    try:
        return JSONResponse(run_lesson(lesson_id, params))
    except KeyError:
        return JSONResponse({"error": "Unknown lesson."}, status_code=404)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")


if __name__ == "__main__":
    main()
