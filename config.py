"""
Central configuration for AstroGuard.
All thresholds, model parameters, database paths, and metric definitions live here.
No magic numbers should appear elsewhere in the application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Database configuration
DATABASE_PATH = os.getenv("ASTROGUARD_DB", str(BASE_DIR / "astroguard.db"))

# Baseline and Evaluation Windows
BASELINE_DAYS = 30
EVALUATION_START_DAY = 31

# Z-Score Anomaly Thresholds
WATCH_Z_THRESHOLD = 2.0
CRITICAL_Z_THRESHOLD = 3.0
CRITICAL_COMPOUND_COUNT = 3

# Alert Level Constants
LEVEL_NORMAL = "Normal"
LEVEL_WATCH = "Watch"
LEVEL_CRITICAL = "Critical"

# Tracked Health Indicators
# Risk direction defines which drift poses medical danger:
# - 'up': elevated values increase risk (Heart Rate, Stress)
# - 'down': depressed values increase risk (SpO2, Sleep Duration)
METRICS = {
    "resting_hr": {
        "name": "resting_hr",
        "label": "Resting Heart Rate",
        "unit": "bpm",
        "risk_direction": "up",
        "typical_min": 50.0,
        "typical_max": 100.0,
        "default_mean": 62.0,
        "default_sd": 2.5,
    },
    "spo2": {
        "name": "spo2",
        "label": "Blood Oxygen (SpO2)",
        "unit": "%",
        "risk_direction": "down",
        "typical_min": 90.0,
        "typical_max": 100.0,
        "default_mean": 98.0,
        "default_sd": 0.6,
    },
    "sleep_hours": {
        "name": "sleep_hours",
        "label": "Sleep Duration",
        "unit": "hrs",
        "risk_direction": "down",
        "typical_min": 0.0,
        "typical_max": 12.0,
        "default_mean": 7.2,
        "default_sd": 0.5,
    },
    "stress_score": {
        "name": "stress_score",
        "label": "Self-Reported Stress",
        "unit": "1-10",
        "risk_direction": "up",
        "typical_min": 1.0,
        "typical_max": 10.0,
        "default_mean": 2.8,
        "default_sd": 0.6,
    },
}

# Mars Simulation Storyline Settings
SIMULATION_TOTAL_DAYS = 60
SIMULATION_START_DATE = "2026-01-01"
# Clip normal-day synthetic Gaussian noise to +/-1.5 SD to prevent accidental Watch triggers (|z| <= 1.5)
SIMULATION_CLIP_STD_DEV = 1.5
# Day 35 single-metric post-EVA exercise blip (target z = 2.5 in Watch zone [2.0, 3.0])
SIMULATION_HR_BLIP_DAY = 35
SIMULATION_HR_BLIP_Z = 2.5

# Days 45-48 dust storm event parameters (multi-metric compound disruption)
SIMULATION_STORM_START_DAY = 45
SIMULATION_STORM_END_DAY = 48
SIMULATION_STORM_HR_MIN = 10.0
SIMULATION_STORM_HR_MAX = 14.0
SIMULATION_STORM_SPO2_MIN = 94.0
SIMULATION_STORM_SPO2_MAX = 95.0
SIMULATION_STORM_SLEEP_MIN = 4.5
SIMULATION_STORM_SLEEP_MAX = 5.0
SIMULATION_STORM_STRESS_MIN = 6.0
SIMULATION_STORM_STRESS_MAX = 8.0

# Days 49-51 gradual recovery de-escalation factors of dust-storm deviation
SIMULATION_RECOVERY_END_DAY = 51
SIMULATION_RECOVERY_FACTORS = {49: 0.50, 50: 0.20, 51: 0.05}

# AI Explainer / Gemini Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TIMEOUT_SECONDS = 8
AI_EXPLANATIONS_ENABLED = os.getenv("AI_EXPLANATIONS_ENABLED", "true").lower() in ("true", "1", "yes")

# Web Server Settings
SECRET_KEY = os.getenv("SECRET_KEY", "astroguard-mars-mission-secret-key-2026")
MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2 MB file upload limit
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
FLASK_DEBUG = False

