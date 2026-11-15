"""
Unit tests for database.py.
Verifies schema initialization, CRUD operations, stress upserts, and alert regeneration support.
"""

import os
import gc
import tempfile
import pandas as pd
import pytest

import database


@pytest.fixture
def temp_db():
    """Create and tear down a temporary SQLite database for testing."""
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


def test_init_db(temp_db):
    """Verify tables exist after init_db."""
    with database.get_db_connection(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row["name"] for row in cursor.fetchall()]
        assert "health_readings" in tables
        assert "checkins" in tables
        assert "alerts" in tables


def test_insert_readings_from_df(temp_db):
    """Test bulk insertion of health readings from a DataFrame."""
    df = pd.DataFrame(
        [
            {
                "day": 1,
                "date": "2026-01-01",
                "resting_hr": 62.0,
                "spo2": 98.0,
                "sleep_hours": 7.5,
                "stress_score": 3.0,
            },
            {
                "day": 2,
                "date": "2026-01-02",
                "resting_hr": 63.0,
                "spo2": 97.5,
                "sleep_hours": 7.0,
                "stress_score": 3.2,
            },
        ]
    )
    count = database.insert_readings_from_df(df, temp_db)
    assert count == 2

    readings = database.get_all_readings(temp_db)
    assert len(readings) == 2
    assert readings.iloc[0]["date"] == "2026-01-01"
    assert readings.iloc[1]["resting_hr"] == 63.0


def test_upsert_reading_stress_existing_date(temp_db):
    """Test updating stress_score on an existing date row."""
    df = pd.DataFrame(
        [
            {
                "day": 1,
                "date": "2026-01-01",
                "resting_hr": 62.0,
                "spo2": 98.0,
                "sleep_hours": 7.5,
                "stress_score": 2.5,
            }
        ]
    )
    database.insert_readings_from_df(df, temp_db)

    # Upsert updated stress score
    database.upsert_reading_stress("2026-01-01", 6.8, temp_db)

    readings = database.get_all_readings(temp_db)
    assert len(readings) == 1
    assert readings.iloc[0]["stress_score"] == 6.8
    assert readings.iloc[0]["resting_hr"] == 62.0


def test_upsert_reading_stress_new_date(temp_db):
    """Test inserting a new date row when check-in date doesn't exist yet."""
    database.upsert_reading_stress("2026-02-15", 5.5, temp_db)

    readings = database.get_all_readings(temp_db)
    assert len(readings) == 1
    row = readings.iloc[0]
    assert row["date"] == "2026-02-15"
    assert row["stress_score"] == 5.5
    assert pd.isna(row["resting_hr"])
    assert pd.isna(row["spo2"])


def test_save_checkin(temp_db):
    """Test logging raw answers to checkins table."""
    checkin_id = database.save_checkin(
        date_str="2026-01-05",
        sleep_quality=4,
        mood=3,
        workload=5,
        physical_comfort=4,
        notes="Moderate fatigue post-EVA",
        calculated_stress=4.2,
        db_path=temp_db,
    )
    assert checkin_id > 0

    logs = database.get_checkins(temp_db)
    assert len(logs) == 1
    assert logs[0]["notes"] == "Moderate fatigue post-EVA"
    assert logs[0]["calculated_stress"] == 4.2


def test_alerts_batch_and_clear(temp_db):
    """Test saving and clearing alerts batch."""
    alerts = [
        {
            "date": "2026-02-05",
            "level": "Watch",
            "metrics_triggered": ["resting_hr"],
            "z_scores": {"resting_hr": 2.4},
            "explanation": "Elevated HR detected.",
            "action": "Hydrate and rest.",
        },
        {
            "date": "2026-02-15",
            "level": "Critical",
            "metrics_triggered": ["resting_hr", "spo2", "sleep_hours"],
            "z_scores": {"resting_hr": 4.1, "spo2": -3.2, "sleep_hours": -2.8},
            "explanation": "Compound vital decline.",
            "action": "Immediate medical review.",
        },
    ]

    saved_count = database.save_alerts_batch(alerts, temp_db)
    assert saved_count == 2

    retrieved = database.get_alerts(temp_db)
    assert len(retrieved) == 2
    assert retrieved[0]["level"] == "Critical"  # Ordered by date DESC
    assert "resting_hr" in retrieved[0]["metrics_triggered"]

    # Clear alerts
    database.clear_alerts(temp_db)
    assert len(database.get_alerts(temp_db)) == 0


def test_clear_all_data(temp_db):
    """Test clearing all tables."""
    df = pd.DataFrame([{"day": 1, "date": "2026-01-01", "resting_hr": 60, "spo2": 99, "sleep_hours": 8, "stress_score": 2}])
    database.insert_readings_from_df(df, temp_db)
    database.save_checkin("2026-01-01", 5, 5, 1, 5, "Good", 1.0, temp_db)
    database.save_alerts_batch([{"date": "2026-01-01", "level": "Normal"}], temp_db)

    database.clear_all_data(temp_db)
    assert len(database.get_all_readings(temp_db)) == 0
    assert len(database.get_checkins(temp_db)) == 0
    assert len(database.get_alerts(temp_db)) == 0
