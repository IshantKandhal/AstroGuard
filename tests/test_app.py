"""
Integration and end-to-end route tests for app.py using Flask test client.
Verifies all routes on empty/populated states, demo loading, day selection,
bad CSV rejection, stressed check-in upserts, reset protection, PDF streaming,
and complete absence of external http(s) URLs in offline assets.
"""

import os
import io
import re
import tempfile
import gc
from unittest.mock import patch
import pytest

import config
import database
import app as flask_app


@pytest.fixture
def client():
    """Create Flask test client with an isolated temporary database."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database.init_db(db_path)

    # Patch config DATABASE_PATH to use isolated temp db
    with patch("config.DATABASE_PATH", db_path):
        flask_app.app.config["TESTING"] = True
        flask_app.app.config["WTF_CSRF_ENABLED"] = False
        with flask_app.app.test_client() as test_client:
            yield test_client

    gc.collect()
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
    except PermissionError:
        pass


def test_each_page_returns_200_on_empty_db(client):
    """Verify all application routes return 200 on an empty database without crashing."""
    routes = ["/", "/upload", "/checkin", "/alerts", "/reset", "/report"]
    for r in routes:
        response = client.get(r)
        assert response.status_code == 200, f"Route {r} failed with status {response.status_code}"


def test_load_demo_creates_60_rows_and_day_60_is_normal(client):
    """Verify /load-demo loads 60 days of telemetry and dashboard day 60 shows Normal."""
    post_res = client.post("/load-demo", follow_redirects=True)
    assert post_res.status_code == 200
    html = post_res.get_data(as_text=True)

    assert "Loaded Mars demo telemetry successfully" in html
    # Day 60 should be Normal
    assert "Health Status: NORMAL" in html


def test_day_selector_critical_and_watch(client):
    """Verify /?day=46 displays Critical status and /?day=35 displays Watch status."""
    client.post("/load-demo", follow_redirects=True)

    # Check Day 46 (Dust storm peak)
    res_46 = client.get("/?day=46")
    assert res_46.status_code == 200
    html_46 = res_46.get_data(as_text=True)
    assert "Health Status: CRITICAL" in html_46

    # Check Day 35 (Post-EVA exercise HR spike)
    res_35 = client.get("/?day=35")
    assert res_35.status_code == 200
    html_35 = res_35.get_data(as_text=True)
    assert "Health Status: WATCH" in html_35


def test_jump_to_latest_critical_alert(client):
    """
    Verify 'Jump to Latest Critical Alert':
    - With demo data, latest critical alert is Day 49.
    - After a stressed check-in on Day 60, latest critical jumps to Day 60.
    - Button is hidden when there are no Critical alerts.
    """
    # 1. On empty DB, no jump button
    res_empty = client.get("/")
    assert "Jump to Latest Critical Alert" not in res_empty.get_data(as_text=True)

    # 2. Load demo data: latest critical is Day 49
    client.post("/load-demo", follow_redirects=True)
    res_demo = client.get("/?day=60")
    html_demo = res_demo.get_data(as_text=True)
    assert "Jump to Latest Critical Alert (Day 49)" in html_demo
    assert "/?day=49" in html_demo

    # 3. Submit extreme stress checkin on Day 60 (stress=10.0 -> critical)
    client.post(
        "/checkin",
        data={"date": "2026-03-01", "mood": "5", "sleep_quality": "5", "workload": "5", "physical_comfort": "5", "notes": ""},
        follow_redirects=True,
    )
    res_updated = client.get("/?day=31")
    html_updated = res_updated.get_data(as_text=True)
    assert "Jump to Latest Critical Alert (Day 60)" in html_updated
    assert "/?day=60" in html_updated


def test_load_demo_idempotence_and_reset_reload(client):
    """Verify loading demo twice preserves exactly 60 rows, and reset-reload cycles work repeatedly."""
    # First load
    client.post("/load-demo", follow_redirects=True)
    df1 = database.get_all_readings()
    assert len(df1) == 60

    # Second load: must replace, never duplicate
    client.post("/load-demo", follow_redirects=True)
    df2 = database.get_all_readings()
    assert len(df2) == 60

    # Reset
    client.post("/reset", follow_redirects=True)
    assert len(database.get_all_readings()) == 0
    assert len(database.get_alerts()) == 0

    # Reload demo data again
    client.post("/load-demo", follow_redirects=True)
    df3 = database.get_all_readings()
    assert len(df3) == 60
    assert len(database.get_alerts()) > 0


def test_bad_csv_and_non_csv_rejected(client):
    """Verify non-csv files and CSVs missing required columns are rejected with error flash."""
    # 1. Non-CSV file rejection
    data_txt = {"file": (io.BytesIO(b"dummy data"), "telemetry.txt")}
    res_txt = client.post("/upload", data=data_txt, content_type="multipart/form-data", follow_redirects=True)
    html_txt = res_txt.get_data(as_text=True)
    assert "Only .csv files are supported" in html_txt

    # 2. Bad CSV with missing columns
    bad_csv_bytes = b"day,date,resting_hr\n1,2026-01-01,65.0\n"
    data_bad = {"file": (io.BytesIO(bad_csv_bytes), "telemetry.csv")}
    res_bad = client.post("/upload", data=data_bad, content_type="multipart/form-data", follow_redirects=True)
    html_bad = res_bad.get_data(as_text=True)
    assert "CSV missing mandatory columns" in html_bad

    # Ensure no rows were added to the database
    res_dash = client.get("/")
    assert "No Telemetry Loaded Yet" in res_dash.get_data(as_text=True)


def test_stressed_checkin_updates_stress_and_regenerates_alerts(client):
    """Verify a high-stress check-in updates that date's stress_score and recalibrates alerts."""
    client.post("/load-demo", follow_redirects=True)

    # Submit check-in for Day 60 (2026-03-01) with maximal distress (all 5s -> score 10.0)
    checkin_data = {
        "date": "2026-03-01",
        "mood": "5",
        "sleep_quality": "5",
        "workload": "5",
        "physical_comfort": "5",
        "notes": "Emergency workload surge",
    }
    post_res = client.post("/checkin", data=checkin_data, follow_redirects=True)
    assert post_res.status_code == 200
    html = post_res.get_data(as_text=True)

    # Flash message should confirm score 10.0
    assert "Calculated stress score: 10.0/10" in html
    # Status on day 60 should now trip an alert (Watch or Critical due to stress 10.0)
    assert "Health Status: WATCH" in html or "Health Status: CRITICAL" in html


def test_reset_data_post_vs_get(client):
    """Verify POST /reset clears data while GET /reset does not."""
    client.post("/load-demo", follow_redirects=True)

    # GET /reset should render confirmation, keeping data intact
    res_get = client.get("/reset")
    assert res_get.status_code == 200
    assert "Confirm Telemetry Reset" in res_get.get_data(as_text=True)

    # Check dashboard still has data
    res_dash = client.get("/")
    assert "No Telemetry Loaded Yet" not in res_dash.get_data(as_text=True)

    # POST /reset clears all data
    res_post = client.post("/reset", follow_redirects=True)
    assert res_post.status_code == 200
    assert "All mission telemetry, check-ins, alerts, and cached data have been erased" in res_post.get_data(as_text=True)

    # Dashboard should now be empty
    res_dash_empty = client.get("/")
    assert "No Telemetry Loaded Yet" in res_dash_empty.get_data(as_text=True)


def test_report_endpoint_returns_pdf(client):
    """Verify /report returns an application/pdf attachment."""
    client.post("/load-demo", follow_redirects=True)
    res = client.get("/report")
    assert res.status_code == 200
    assert res.headers["Content-Type"] == "application/pdf"
    assert res.data.startswith(b"%PDF")


def test_no_external_urls_in_templates_and_css():
    """Verify templates and static/css/style.css contain zero external http(s) URLs for offline operation."""
    files_to_check = [
        "static/css/style.css",
        "templates/base.html",
        "templates/dashboard.html",
        "templates/checkin.html",
        "templates/alerts.html",
        "templates/upload.html",
        "templates/reset_confirm.html",
    ]

    url_pattern = re.compile(r'https?://', re.IGNORECASE)

    for rel_path in files_to_check:
        full_path = config.BASE_DIR / rel_path
        assert full_path.exists(), f"File {rel_path} must exist"
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        matches = url_pattern.findall(content)
        assert len(matches) == 0, f"External URL found in {rel_path}: {matches}"


def test_api_explanation_endpoint_returns_rule_text_without_key(client):
    """Verify /api/explanation/<date> returns rule-based JSON response when no API key is configured."""
    client.post("/load-demo", follow_redirects=True)

    with patch("config.GEMINI_API_KEY", ""):
        res = client.get("/api/explanation/2026-02-14")  # Day 45
        assert res.status_code == 200
        data = res.get_json()
        assert data["source"] == "rules"
        assert "Rule-based explanation" in data["label"]
        assert len(data["text"]) > 0
        assert len(data["actions"]) > 0

