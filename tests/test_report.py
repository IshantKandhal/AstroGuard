"""
Unit tests for report.py.
Verifies PDF generation, page count constraints (<= 2 pages), safety banners,
absence of raw z-scores, edge cases (empty DB, missing data), and buffer streaming.
"""

import io
import os
import gc
import tempfile
import pytest

import database
import data_generator
import engine
import report


@pytest.fixture
def temp_populated_db():
    """Create a temporary database loaded with Mars demo data and evaluated alerts."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database.init_db(path)

    df = data_generator.generate_mars_data(seed=42, save_csv=False)
    database.insert_readings_from_df(df, path)
    alerts = engine.evaluate_all_alerts(df)
    database.save_alerts_batch(alerts, path)

    # Add a sample checkin
    database.save_checkin(
        date_str="2026-03-01",
        sleep_quality=4,
        mood=4,
        workload=3,
        physical_comfort=5,
        notes="Mission nominal post-recovery",
        calculated_stress=2.5,
        db_path=path,
    )

    yield path
    gc.collect()
    try:
        if os.path.exists(path):
            os.remove(path)
    except PermissionError:
        pass


@pytest.fixture
def temp_empty_db():
    """Create an empty database with schema initialized."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database.init_db(path)
    yield path
    gc.collect()
    try:
        if os.path.exists(path):
            os.remove(path)
    except PermissionError:
        pass


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    """Calculate total pages in generated PDF."""
    return pdf_bytes.count(b"/Type /Page") - pdf_bytes.count(b"/Type /Pages")


def test_report_starts_with_pdf_header(temp_populated_db):
    """Verify generated report starts with standard PDF magic bytes."""
    buf = io.BytesIO()
    report.build_report(buf, db_path=temp_populated_db)
    assert buf.getvalue().startswith(b"%PDF")


def test_report_at_most_two_pages(temp_populated_db):
    """Verify generated report fits within 1 or 2 pages."""
    buf = io.BytesIO()
    report.build_report(buf, db_path=temp_populated_db)
    pages = get_pdf_page_count(buf.getvalue())
    assert 1 <= pages <= 2, f"Expected at most 2 pages, got {pages}"


def test_report_empty_db_does_not_crash(temp_empty_db):
    """Verify empty database returns a friendly 1-page PDF without crashing."""
    buf = io.BytesIO()
    report.build_report(buf, db_path=temp_empty_db)
    pdf_bytes = buf.getvalue()
    assert pdf_bytes.startswith(b"%PDF")
    pages = get_pdf_page_count(pdf_bytes)
    assert pages == 1, f"Expected 1 page for empty DB, got {pages}"


def test_report_text_contains_simulated_data_banner(temp_populated_db):
    """Verify required 'SIMULATED DATA' safety banner is present in the text."""
    buf = io.BytesIO()
    report.build_report(buf, db_path=temp_populated_db)
    assert b"SIMULATED DATA" in buf.getvalue()


def test_report_text_never_contains_raw_z_scores(temp_populated_db):
    """Verify generated report never displays the term 'z-score' or raw z statistics."""
    buf = io.BytesIO()
    report.build_report(buf, db_path=temp_populated_db)
    lower_bytes = buf.getvalue().lower()
    assert b"z-score" not in lower_bytes
    assert b"z score" not in lower_bytes


def test_build_sample_report_file(temp_populated_db):
    """Generate sample_report.pdf on disk for inspection."""
    output_path = "sample_report.pdf"
    report.build_report(output_path, db_path=temp_populated_db)

    assert os.path.exists(output_path), "sample_report.pdf should have been created on disk"
    with open(output_path, "rb") as f:
        content = f.read()
    assert content.startswith(b"%PDF")
    assert get_pdf_page_count(content) <= 2

