# AstroGuard

AI-assisted astronaut health monitoring system for long-duration space missions.

AstroGuard learns each astronaut's **personal baseline**, spots unusual changes in their health, and explains every alert in plain language with a suggested action. It needs no wearable hardware and no paid services, and runs on any laptop.

> **Disclaimer:** AstroGuard is a student prototype built with simulated data. It is not a medical device and must not be used for real medical decisions.

---

## Problem

On a Mars-class mission, messages to Earth take ~20 minutes each way. Astronauts cannot wait for ground doctors to notice a slow health decline. They need a tool that tells them, early and clearly, that something has changed for *them*.

## Solution

AstroGuard reads daily health data, compares it with the astronaut's own normal, and raises a **Watch** or **Critical** alert when something drifts. Each alert says why it fired and what to do.

## Features

- Personal baseline built from the first 30 days of data
- Four tracked indicators: heart rate, SpO2, sleep, stress
- Z-score anomaly detection plus a compound rule (several metrics off together)
- Three alert levels: Normal / Watch / Critical
- Explainable alerts with suggested actions (rules decide, Gemini explains)
- Daily mental wellness check-in that feeds the stress score (upsert by date)
- CSV upload and a one-click "Load Mars Demo Data" button
- Trend dashboard and health history (100% offline via local Chart.js)
- Downloadable PDF summary report for the ground team

## How the baseline works

```
z = (today's value - baseline mean) / baseline standard deviation
```

Example: baseline HR is 62 bpm with SD 2.5. On Day 45 HR is 76 bpm, so z = (76 - 62) / 2.5 = 5.6, which is far outside normal.

| Level | Rule |
|---|---|
| Normal | Every metric is within the baseline range |
| Watch | Any one metric has |z| > 2 |
| Critical | 3 or more metrics deviate together, or any one has |z| > 3 |

Sleep and SpO2 are risky when they go **down**; heart rate and stress when they go **up**. The engine treats direction accordingly.

## Demo storyline (Mars mission)

1. Day 1-30: normal data, baseline is built (clean data, random noise clipped to +/-1.5 SD).
2. Day 35: a single-metric HR blip (about +5 bpm after strenuous EVA exercise) triggers a **Watch** alert, showing the system spots single-vital anomalies while avoiding false alarms during baseline.
3. Day 45-48: a dust storm disrupts sleep, raises HR and stress, lowers SpO2. A **Critical** alert fires with an explanation and action.
4. Day 52+: values recover; the trend chart shows it.
5. Export the PDF report.

## Tech stack

Python 3.12/3.14(tested), Flask, SQLite, pandas, numpy, Chart.js (vendored offline), ReportLab, Google Gemini API (free tier, optional).

## Project structure

All files reside in the repository root:

```
.
├── app.py               # Flask routes and application entry point
├── config.py            # Thresholds, baseline window, model name, DB path
├── database.py          # SQLite schema, connections, and query operations
├── data_generator.py    # Simulated Mars mission 60-day dataset generator
├── engine.py            # Baseline calculation, directional z-scores, alert rules
├── llm.py               # Gemini explanations with offline rule-based fallback
├── report.py            # ReportLab PDF report builder
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template for GEMINI_API_KEY
├── data/
│   └── mars_demo.csv    # Generated Mars demo dataset
├── static/
│   ├── css/
│   │   └── style.css    # Clean mission-control styling
│   └── js/
│       ├── chart.min.js # Vendored local Chart.js for 100% offline support
│       └── main.js      # Dashboard charts and interactivity
└── templates/
    ├── base.html        # Shared layout and navigation
    ├── dashboard.html   # Main dashboard with vitals, trends, and current alert
    ├── checkin.html     # Mental wellness daily check-in form
    └── alerts.html      # Comprehensive alert log with explanations
```

## Setup

```bash
git clone <your-repo-url>
cd <repo-dir>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add your GEMINI_API_KEY (optional)
python app.py
```

Open `http://127.0.0.1:5000`.

If no API key is set or no internet connection is present, AstroGuard uses built-in rule-based explanations, so the demo always works.

## Usage

1. Click **Load Mars Demo Data** (or upload your own CSV).
2. Open the **Dashboard** to see trends and the current alert.
3. Fill the daily **Check-in** to update the stress score for that date.
4. Open **Alerts** to read the explanation and suggested action.
5. Click **Download Report** for the PDF summary.

## CSV format

```
day,date,resting_hr,spo2,sleep_hours,stress_score
1,2026-01-01,61.8,97.6,6.7,3.1
```

| Column | Meaning | Typical range |
|---|---|---|
| resting_hr | Resting heart rate (bpm) | 50-100 |
| spo2 | Blood oxygen saturation (%) | 90-100 |
| sleep_hours | Hours slept | 0-12 |
| stress_score | Self-reported stress (1-10) | 1-10 |

## Limitations

- Uses simulated data, not real astronaut data
- One astronaut at a time
- Optional internet for Gemini explanations (rule-based fallback works offline)
- Thresholds are illustrative, not clinically validated

## Use of Super Intelligence (SI)

- Code: written largely with Google Antigravity (Gemini) from step-by-step prompts written by our team. We reviewed, ran and tested it (42 automated tests).
- Planning, documentation and checklists: drafted with Claude (Anthropic).
- In the app: optional alert explanations by the Google Gemini API. Alert levels are decided by rules, not by the SI.
- Data: simulated by a Python script (data_generator.py). It is not real astronaut data.

## Project Demo

- Demo slides: https://drive.google.com/file/d/17GOvjrqdrgcLFHYW-0mHpgE-FHQRB8JP/view?usp=sharing
- Demo video: https://drive.google.com/file/d/1D_ovF0AOhQW2Dxu19sq5WntRxE2MgX06/view?usp=sharing
