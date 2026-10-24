import base64

from fastapi.testclient import TestClient

from physical_ai_lab.server import app
from physical_ai_lab.studio import lesson_catalog, run_lesson


def _png(payload: dict) -> None:
    assert payload["images"], payload["id"]
    raw = base64.b64decode(payload["images"][0]["png"])
    assert raw.startswith(b"\x89PNG")


def test_catalog_lists_every_lesson():
    titles = [lesson["title"] for lesson in lesson_catalog()]
    assert titles[0] == "Check the tools"
    assert titles[-1] == "One perception pipeline"
    assert len(titles) == 13
    assert all(lesson["idea"] and lesson["watch"] for lesson in lesson_catalog())


def test_each_fast_lesson_returns_a_call_and_output():
    fast_ids = [
        "environment",
        "foundations",
        "scene",
        "profiling",
        "edges",
        "matching",
        "torch",
        "camera",
        "depth",
        "motion",
        "quantize",
        "pipeline",
    ]
    for lesson_id in fast_ids:
        result = run_lesson(lesson_id, {})
        assert result["call"]
        assert result["lines"]
        assert result["notice"]
        assert result["elapsed_ms"] >= 0
    for lesson_id in ("scene", "edges", "matching", "camera", "depth", "motion", "quantize", "pipeline"):
        _png(run_lesson(lesson_id, {}))


def test_controls_are_clamped_and_bad_text_is_rejected():
    huge = run_lesson("torch", {"n": 99999})
    assert huge["params"]["n"] == 2048
    try:
        run_lesson("foundations", {"readings": "hello"})
    except ValueError as exc:
        assert "number" in str(exc)
    else:
        raise AssertionError("bad readings should fail")


def test_motion_recovers_the_requested_slide():
    result = run_lesson("motion", {"shift_x": 15, "shift_y": 0, "width": 320, "height": 200})
    assert "dx=15.00" in result["lines"][0]
    assert "dy=0.00" in result["lines"][0]


def test_http_page_and_lesson_run():
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200
    assert "Run this function" in page.text
    script = client.get("/static/app.js")
    assert script.status_code == 200
    assert "/api/lessons/" in script.text
    catalog = client.get("/api/lessons")
    assert catalog.status_code == 200
    assert catalog.json()[1]["id"] == "foundations"
    ran = client.post("/api/lessons/foundations/run", json={"params": {"speed_mps": 2}})
    assert ran.status_code == 200
    body = ran.json()
    assert "7.2 km/h" in "\n".join(body["lines"])
    missing = client.post("/api/lessons/nope/run", json={"params": {}})
    assert missing.status_code == 404
    bad = client.post("/api/lessons/foundations/run", json={"params": {"readings": "nope"}})
    assert bad.status_code == 400
