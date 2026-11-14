"""
Database access and query operations for AstroGuard.
Handles SQLite connection, schema initialization, and CRUD operations.
"""

import json
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional

import config


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Create a new SQLite connection with row access by column name."""
    path = db_path or config.DATABASE_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database tables with appropriate schemas and indices."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # Table: health_readings (daily astronaut vital metrics)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS health_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                day INTEGER,
                date TEXT UNIQUE NOT NULL,
                resting_hr REAL,
                spo2 REAL,
                sleep_hours REAL,
                stress_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Table: checkins (daily subjective wellness questionnaire answers)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS checkins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                sleep_quality INTEGER,
                mood INTEGER,
                workload INTEGER,
                physical_comfort INTEGER,
                notes TEXT,
                calculated_stress REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Table: alerts (detected anomaly records, explanations, and actions)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                level TEXT NOT NULL,
                metrics_triggered TEXT,
                z_scores TEXT,
                explanation TEXT,
                action TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Table: llm_cache (persistent cache for AI/rule explanations keyed by date and facts hash)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                facts_hash TEXT NOT NULL,
                text TEXT NOT NULL,
                actions TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, facts_hash)
            )
            """
        )

        # Indices for frequent query lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_readings_date ON health_readings(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_date ON alerts(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_checkins_date ON checkins(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_llm_cache_lookup ON llm_cache(date, facts_hash)")
        conn.commit()


def insert_readings_from_df(df: pd.DataFrame, db_path: Optional[str] = None) -> int:
    """
    Replace all health readings with data from a pandas DataFrame.
    Returns the number of rows inserted.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM health_readings")

        rows = []
        for index, row in df.iterrows():
            day_val = int(row["day"]) if "day" in row and pd.notnull(row["day"]) else (index + 1)
            date_val = str(row["date"])
            hr_val = float(row["resting_hr"]) if pd.notnull(row.get("resting_hr")) else None
            spo2_val = float(row["spo2"]) if pd.notnull(row.get("spo2")) else None
            sleep_val = float(row["sleep_hours"]) if pd.notnull(row.get("sleep_hours")) else None
            stress_val = float(row["stress_score"]) if pd.notnull(row.get("stress_score")) else None

            rows.append((day_val, date_val, hr_val, spo2_val, sleep_val, stress_val))

        cursor.executemany(
            """
            INSERT INTO health_readings (day, date, resting_hr, spo2, sleep_hours, stress_score)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return len(rows)


def upsert_reading_stress(date_str: str, stress_score: float, db_path: Optional[str] = None) -> None:
    """
    Upsert the stress score for a given date in health_readings.
    If the date exists, updates the stress_score.
    If no row exists for that date, creates a new row with other vitals as NULL.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, day FROM health_readings WHERE date = ?", (date_str,))
        existing = cursor.fetchone()

        if existing:
            cursor.execute(
                "UPDATE health_readings SET stress_score = ? WHERE date = ?",
                (stress_score, date_str),
            )
        else:
            # Determine next day index based on highest existing day
            cursor.execute("SELECT MAX(day) AS max_day FROM health_readings")
            max_row = cursor.fetchone()
            next_day = (max_row["max_day"] + 1) if max_row and max_row["max_day"] else 1

            cursor.execute(
                """
                INSERT INTO health_readings (day, date, resting_hr, spo2, sleep_hours, stress_score)
                VALUES (?, ?, NULL, NULL, NULL, ?)
                """,
                (next_day, date_str, stress_score),
            )
        conn.commit()


def save_checkin(
    date_str: str,
    sleep_quality: int,
    mood: int,
    workload: int,
    physical_comfort: int,
    notes: str,
    calculated_stress: float,
    db_path: Optional[str] = None,
) -> int:
    """Record raw questionnaire answers into the checkins audit log."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO checkins (date, sleep_quality, mood, workload, physical_comfort, notes, calculated_stress)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                date_str,
                sleep_quality,
                mood,
                workload,
                physical_comfort,
                notes,
                calculated_stress,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def clear_alerts(db_path: Optional[str] = None) -> None:
    """Remove all records from the alerts table."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM alerts")
        conn.commit()


def save_alerts_batch(alerts_list: List[Dict[str, Any]], db_path: Optional[str] = None) -> int:
    """Bulk insert evaluated alerts into the alerts table."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        rows = [
            (
                alert["date"],
                alert["level"],
                json.dumps(alert.get("metrics_triggered", [])),
                json.dumps(alert.get("z_scores", {})),
                alert.get("explanation", ""),
                alert.get("action", ""),
            )
            for alert in alerts_list
        ]
        cursor.executemany(
            """
            INSERT INTO alerts (date, level, metrics_triggered, z_scores, explanation, action)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return len(rows)


def get_all_readings(db_path: Optional[str] = None) -> pd.DataFrame:
    """Retrieve all health readings as a pandas DataFrame sorted by date ascending."""
    conn = get_db_connection(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT id, day, date, resting_hr, spo2, sleep_hours, stress_score FROM health_readings ORDER BY date ASC",
            conn,
        )
        return df
    finally:
        conn.close()


def get_latest_reading_date(db_path: Optional[str] = None) -> Optional[str]:
    """Retrieve the latest available reading date string."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT date FROM health_readings ORDER BY date DESC LIMIT 1")
        row = cursor.fetchone()
        return row["date"] if row else None


def get_alerts(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all alerts ordered by date descending."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, date, level, metrics_triggered, z_scores, explanation, action, created_at
            FROM alerts
            ORDER BY date DESC, id DESC
            """
        )
        rows = cursor.fetchall()
        results = []
        for r in rows:
            results.append(
                {
                    "id": r["id"],
                    "date": r["date"],
                    "level": r["level"],
                    "metrics_triggered": json.loads(r["metrics_triggered"] or "[]"),
                    "z_scores": json.loads(r["z_scores"] or "{}"),
                    "explanation": r["explanation"],
                    "action": r["action"],
                    "created_at": r["created_at"],
                }
            )
        return results


def get_alert_for_date(date_str: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve the alert for a specific date if one exists."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, date, level, metrics_triggered, z_scores, explanation, action, created_at
            FROM alerts
            WHERE date = ?
            ORDER BY id DESC LIMIT 1
            """,
            (date_str,),
        )
        r = cursor.fetchone()
        if not r:
            return None
        return {
            "id": r["id"],
            "date": r["date"],
            "level": r["level"],
            "metrics_triggered": json.loads(r["metrics_triggered"] or "[]"),
            "z_scores": json.loads(r["z_scores"] or "{}"),
            "explanation": r["explanation"],
            "action": r["action"],
            "created_at": r["created_at"],
        }


def get_checkins(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all recorded check-ins ordered by date descending."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, date, sleep_quality, mood, workload, physical_comfort, notes, calculated_stress, created_at
            FROM checkins
            ORDER BY date DESC, id DESC
            """
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_cached_explanation(
    date_str: str,
    facts_hash: str,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve a cached explanation by date and facts_hash if present."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT text, actions, source FROM llm_cache WHERE date = ? AND facts_hash = ?",
            (date_str, facts_hash),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "text": row["text"],
            "actions": json.loads(row["actions"] or "[]"),
            "source": row["source"],
        }


def save_cached_explanation(
    date_str: str,
    facts_hash: str,
    text: str,
    actions: List[str],
    source: str,
    db_path: Optional[str] = None,
) -> None:
    """Persist or update an explanation in llm_cache."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO llm_cache (date, facts_hash, text, actions, source)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(date, facts_hash) DO UPDATE SET
                text = excluded.text,
                actions = excluded.actions,
                source = excluded.source
            """,
            (date_str, facts_hash, text, json.dumps(actions), source),
        )
        conn.commit()


def clear_all_data(db_path: Optional[str] = None) -> None:
    """Clear all data across health_readings, checkins, alerts, and llm_cache."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM health_readings")
        cursor.execute("DELETE FROM checkins")
        cursor.execute("DELETE FROM alerts")
        cursor.execute("DELETE FROM llm_cache")
        conn.commit()

