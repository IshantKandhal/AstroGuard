"""
Flask Web Application for AstroGuard.
Serves mission control telemetry dashboards, daily check-ins, alert histories,
CSV ingestion, and ground medical PDF reports.
"""

import io
import os
from typing import Dict, Any, Optional
import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    send_file,
)

import config
import database
import engine
import data_generator
import llm
import report

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH


def regenerate_alerts(db_path: Optional[str] = None) -> None:
    """
    Recalculate baseline and re-evaluate anomaly alerts from scratch.
    Ensures stored alert records never go stale after data changes.
    """
    df = database.get_all_readings(db_path)
    if df.empty:
        database.clear_alerts(db_path)
        return

    complete_df = df[df.apply(engine.is_row_complete, axis=1)]
    if len(complete_df) < config.BASELINE_DAYS:
        database.clear_alerts(db_path)
        return

    baseline = engine.compute_baseline(df)
    evaluated = engine.evaluate_all_alerts(df)

    # Attach initial rule-based explanations for immediate display
    for a in evaluated:
        row_match = df[df["date"] == a["date"]]
        if not row_match.empty:
            r = row_match.iloc[0]
            vitals = {k: float(r[k]) for k in config.METRICS.keys() if pd.notnull(r.get(k))}
        else:
            vitals = {}

        rule_exp = llm.generate_rule_explanation(a, vitals, baseline)
        a["explanation"] = rule_exp["text"]
        a["action"] = "; ".join(rule_exp["actions"])

    database.clear_alerts(db_path)
    database.save_alerts_batch(evaluated, db_path)


@app.route("/")
def dashboard():
    """Main Telemetry Dashboard view with Day Selector, Vital Cards, and Timeline Charts."""
    df = database.get_all_readings()
    if df.empty:
        return render_template("dashboard.html", is_empty=True)

    # Calculate baseline if sufficient complete data exists
    complete_df = df[df.apply(engine.is_row_complete, axis=1)]
    baseline = engine.compute_baseline(df) if len(complete_df) >= config.BASELINE_DAYS else None

    # Load evaluated alerts
    alerts_list = database.get_alerts()
    alert_map = {a["date"]: a for a in alerts_list}

    # Prepare timeline summary list for day selector and Chart.js
    all_days = []
    date_to_day = {}

    for _, row in df.iterrows():
        d_num = int(row["day"]) if pd.notnull(row.get("day")) else 1
        d_date = str(row["date"])
        date_to_day[d_date] = d_num
        a_entry = alert_map.get(d_date)
        lvl = a_entry["level"] if a_entry else config.LEVEL_NORMAL
        all_days.append(
            {
                "day": d_num,
                "date": d_date,
                "level": lvl,
                "resting_hr": float(row["resting_hr"]) if pd.notnull(row.get("resting_hr")) else None,
                "spo2": float(row["spo2"]) if pd.notnull(row.get("spo2")) else None,
                "sleep_hours": float(row["sleep_hours"]) if pd.notnull(row.get("sleep_hours")) else None,
                "stress_score": float(row["stress_score"]) if pd.notnull(row.get("stress_score")) else None,
            }
        )

    # Compute latest Critical day strictly from the evaluated alerts
    crit_days = [date_to_day[a["date"]] for a in alerts_list if a["level"] == config.LEVEL_CRITICAL and a["date"] in date_to_day]
    latest_critical_day = max(crit_days) if crit_days else None

    # Determine selected day (default: latest recorded day)
    latest_day = all_days[-1]["day"]
    day_param = request.args.get("day")
    selected_day = int(day_param) if day_param and day_param.isdigit() else latest_day

    # Retrieve selected row
    selected_row_candidates = [d for d in all_days if d["day"] == selected_day]
    selected_row = selected_row_candidates[0] if selected_row_candidates else all_days[-1]
    selected_day = selected_row["day"]
    selected_date = selected_row["date"]

    # Current level and vitals
    selected_alert = alert_map.get(selected_date)
    current_level = selected_alert["level"] if selected_alert else selected_row["level"]

    current_vitals = {
        "resting_hr": selected_row["resting_hr"],
        "spo2": selected_row["spo2"],
        "sleep_hours": selected_row["sleep_hours"],
        "stress_score": selected_row["stress_score"],
    }

    # Fast initial explanation (cached or rule-based)
    facts_hash = llm.compute_facts_hash(selected_alert or {"level": current_level}, current_vitals)
    cached_exp = database.get_cached_explanation(selected_date, facts_hash)

    if cached_exp:
        explanation_text = cached_exp["text"]
        action_bullets = cached_exp["actions"]
        explanation_source = cached_exp["source"]
    else:
        rule_exp = llm.generate_rule_explanation(selected_alert or {"level": current_level}, current_vitals, baseline)
        explanation_text = rule_exp["text"]
        action_bullets = rule_exp["actions"]
        explanation_source = "rules"

    explanation_label = llm.get_explanation_label(explanation_source)

    # Percentage shifts vs baseline for cards
    def _pct(curr, key):
        if not baseline or curr is None or key not in baseline or baseline[key]["mean"] == 0:
            return "—", "neutral"
        mean_v = baseline[key]["mean"]
        diff_pct = ((curr - mean_v) / mean_v) * 100.0
        r_dir = config.METRICS[key]["risk_direction"]

        is_bad = (r_dir == "up" and diff_pct > 10.0) or (r_dir == "down" and diff_pct < -3.0)
        cls_name = "up-bad" if is_bad else "neutral"
        return f"{diff_pct:+.1f}% vs base", cls_name

    hr_pct_shift, hr_delta_class = _pct(current_vitals["resting_hr"], "resting_hr")
    spo2_pct_shift, spo2_delta_class = _pct(current_vitals["spo2"], "spo2")
    sleep_pct_shift, sleep_delta_class = _pct(current_vitals["sleep_hours"], "sleep_hours")
    stress_pct_shift, stress_delta_class = _pct(current_vitals["stress_score"], "stress_score")

    return render_template(
        "dashboard.html",
        is_empty=False,
        all_days=all_days,
        selected_day=selected_day,
        selected_date=selected_date,
        latest_day=latest_day,
        latest_critical_day=latest_critical_day,
        current_level=current_level,
        current_vitals=current_vitals,
        baseline=baseline,
        baseline_days=config.BASELINE_DAYS,
        hr_pct_shift=hr_pct_shift,
        hr_delta_class=hr_delta_class,
        spo2_pct_shift=spo2_pct_shift,
        spo2_delta_class=spo2_delta_class,
        sleep_pct_shift=sleep_pct_shift,
        sleep_delta_class=sleep_delta_class,
        stress_pct_shift=stress_pct_shift,
        stress_delta_class=stress_delta_class,
        explanation_text=explanation_text,
        action_bullets=action_bullets,
        explanation_source=explanation_source,
        explanation_label=explanation_label,
        ai_enabled=config.AI_EXPLANATIONS_ENABLED and bool(config.GEMINI_API_KEY),
        timeline_json=all_days,
        baseline_json=baseline,
    )


@app.route("/api/explanation/<date_str>")
def api_explanation(date_str: str):
    """
    Non-blocking endpoint to asynchronously upgrade rule explanation to Gemini text.
    Returns immediately with cached or rule explanation if Gemini is unavailable or unconfigured.
    """
    df = database.get_all_readings()
    baseline = engine.compute_baseline(df) if len(df) >= config.BASELINE_DAYS else None
    alert_info = database.get_alert_for_date(date_str) or {"date": date_str, "level": config.LEVEL_NORMAL, "metrics_triggered": []}

    # Query vitals
    row = df[df["date"] == date_str]
    vitals = {}
    if not row.empty:
        r = row.iloc[0]
        vitals = {k: float(r[k]) for k in config.METRICS.keys() if pd.notnull(r.get(k))}

    # Explain alert
    if config.AI_EXPLANATIONS_ENABLED and config.GEMINI_API_KEY:
        exp_result = llm.explain_alert(alert_info, vitals, baseline)
    else:
        exp_result = llm.generate_rule_explanation(alert_info, vitals, baseline)

    return jsonify(
        {
            "text": exp_result["text"],
            "actions": exp_result["actions"],
            "source": exp_result["source"],
            "label": llm.get_explanation_label(exp_result["source"]),
        }
    )


@app.route("/load-demo", methods=["POST"])
def load_demo():
    """Load the pre-generated 60-day Mars mission dataset into the database."""
    csv_file = config.BASE_DIR / "data" / "mars_demo.csv"
    if not csv_file.exists():
        data_generator.generate_mars_data(seed=42, save_csv=True)

    df = pd.read_csv(csv_file)
    row_count = database.insert_readings_from_df(df)
    regenerate_alerts()

    flash(f"Loaded Mars demo telemetry successfully ({row_count} mission days established).", "success")
    return redirect(url_for("dashboard"))


@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    """Telemetry CSV upload form and in-memory validation handler."""
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("No file was selected for upload.", "error")
            return redirect(url_for("upload_page"))

        if not file.filename.lower().endswith(".csv"):
            flash("Invalid file type. Only .csv files are supported.", "error")
            return redirect(url_for("upload_page"))

        try:
            content = file.stream.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(content))

            # Validate required columns
            required_cols = ["date", "resting_hr", "spo2", "sleep_hours", "stress_score"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                flash(f"CSV missing mandatory columns: {', '.join(missing)}", "error")
                return redirect(url_for("upload_page"))

            if df.empty:
                flash("Uploaded CSV file contains no data rows.", "error")
                return redirect(url_for("upload_page"))

            # Populate database and regenerate alerts
            count = database.insert_readings_from_df(df)
            regenerate_alerts()

            flash(f"Successfully ingested {count} telemetry records from {file.filename}.", "success")
            return redirect(url_for("dashboard"))

        except Exception as e:
            flash(f"Error parsing CSV file: {str(e)}", "error")
            return redirect(url_for("upload_page"))

    return render_template("upload.html")


@app.route("/checkin", methods=["GET", "POST"])
def checkin():
    """Subjective mental wellness questionnaire."""
    if request.method == "POST":
        date_str = request.form.get("date", "").strip()
        mood = int(request.form.get("mood", 3))
        sleep_quality = int(request.form.get("sleep_quality", 3))
        workload = int(request.form.get("workload", 3))
        physical_comfort = int(request.form.get("physical_comfort", 3))
        notes = request.form.get("notes", "").strip()[:300]

        if not date_str:
            flash("Observation date is required.", "error")
            return redirect(url_for("checkin"))

        answers = {
            "mood": mood,
            "sleep_quality": sleep_quality,
            "workload": workload,
            "physical_comfort": physical_comfort,
        }

        # Calculate stress score using pure engine function
        calculated_stress = engine.compute_stress(answers)

        # Record in checkins audit log and upsert stress into health_readings
        database.save_checkin(
            date_str=date_str,
            sleep_quality=sleep_quality,
            mood=mood,
            workload=workload,
            physical_comfort=physical_comfort,
            notes=notes,
            calculated_stress=calculated_stress,
        )
        database.upsert_reading_stress(date_str, calculated_stress)
        regenerate_alerts()

        # Find day index for redirect
        df = database.get_all_readings()
        day_match = df[df["date"] == date_str]
        target_day = int(day_match.iloc[0]["day"]) if not day_match.empty and pd.notnull(day_match.iloc[0].get("day")) else None

        flash(
            f"Daily wellness check-in logged for {date_str}. Calculated stress score: {calculated_stress:.1f}/10.",
            "success",
        )
        if target_day:
            return redirect(url_for("dashboard", day=target_day))
        return redirect(url_for("dashboard"))

    default_date = database.get_latest_reading_date() or "2026-01-01"
    return render_template("checkin.html", default_date=default_date)


@app.route("/alerts")
def alerts_page():
    """Historical Anomaly Alerts log with level filtering."""
    all_alerts = database.get_alerts()
    df = database.get_all_readings()
    date_to_day = {}
    if not df.empty:
        for _, r in df.iterrows():
            date_to_day[str(r["date"])] = int(r["day"]) if pd.notnull(r.get("day")) else 1

    # Augment day index
    for a in all_alerts:
        a["day"] = date_to_day.get(a["date"], 1)

    filter_level = request.args.get("level", "All")
    if filter_level in [config.LEVEL_WATCH, config.LEVEL_CRITICAL]:
        filtered_alerts = [a for a in all_alerts if a["level"] == filter_level]
    else:
        filtered_alerts = [a for a in all_alerts if a["level"] in [config.LEVEL_WATCH, config.LEVEL_CRITICAL]]

    return render_template(
        "alerts.html",
        alerts=filtered_alerts,
        filter_level=filter_level,
        evaluation_start_day=config.EVALUATION_START_DAY,
    )


@app.route("/reset", methods=["GET", "POST"])
def reset_confirm():
    """Reset confirmation page for GET requests; action is POST /reset."""
    if request.method == "POST":
        return reset_data()
    return render_template("reset_confirm.html")


@app.route("/reset-action", methods=["POST"])
def reset_data():
    """Clear all database tables."""
    database.clear_all_data()
    flash("All mission telemetry, check-ins, alerts, and cached data have been erased.", "warning")
    return redirect(url_for("dashboard"))


@app.route("/report")
def download_report():
    """Stream ground medical PDF summary report."""
    buf = io.BytesIO()
    report.build_report(buf)
    buf.seek(0)
    return send_file(
        buf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="astroguard_ground_medical_summary.pdf",
    )


if __name__ == "__main__":
    database.init_db()
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )

