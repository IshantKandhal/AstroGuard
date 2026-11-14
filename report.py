"""
PDF Medical Report Builder for AstroGuard.
Generates an executive, publication-grade clinical summary report for ground control doctors.
Built exclusively with ReportLab (platypus and graphics.charts.lineplots.LinePlot).
Strictly adheres to clinical safety: zero exposure of raw z-scores, prominent simulation banner,
and strictly offline operation (never calls Gemini; uses cache or deterministic rule explanations).
"""

import io
from datetime import datetime
from typing import Union, Optional, List, Tuple, Dict, Any
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, String, Line, Rect
from reportlab.graphics.charts.lineplots import LinePlot

import config
import database
import engine
import llm


class NumberedCanvas(canvas.Canvas):
    """Custom canvas that tracks total pages to draw accurate running footers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_footer(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_footer(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4B5563"))
        # Left footer: Medical simulation disclaimer
        self.drawString(36, 20, "SIMULATED DATA - STUDENT PROTOTYPE - NOT FOR MEDICAL USE")
        # Right footer: Page count
        self.drawRightString(612 - 36, 20, f"Page {self._pageNumber} of {page_count} | Ground Medical Team Handoff")
        self.restoreState()


def create_trend_chart(
    data_points: List[Tuple[float, float]],
    baseline_mean: float,
    title: str,
    unit: str,
    chart_color: colors.Color,
) -> Drawing:
    """
    Render a compact LinePlot trend chart showing readings and a dashed baseline average.
    Uses built-in ReportLab graphics LinePlot (no external plotting libraries).
    """
    drawing = Drawing(240, 80)

    # Chart Title
    title_str = String(10, 68, f"{title} ({unit})", fontName="Helvetica-Bold", fontSize=8, fillColor=colors.HexColor("#1F2937"))
    drawing.add(title_str)

    if not data_points or len(data_points) < 2:
        no_data = String(30, 30, "Insufficient trend telemetry", fontName="Helvetica-Oblique", fontSize=8, fillColor=colors.HexColor("#9CA3AF"))
        drawing.add(no_data)
        return drawing

    lp = LinePlot()
    lp.x = 22
    lp.y = 12
    lp.width = 205
    lp.height = 50

    # Series 1: Actual readings, Series 2: Horizontal baseline average
    x_vals = [p[0] for p in data_points]
    baseline_points = [(min(x_vals), baseline_mean), (max(x_vals), baseline_mean)]

    lp.data = [data_points, baseline_points]

    # Line 0: Actual vitals trend
    lp.lines[0].strokeColor = chart_color
    lp.lines[0].strokeWidth = 1.3

    # Line 1: Baseline average line (dashed red/gray)
    lp.lines[1].strokeColor = colors.HexColor("#EF4444")
    lp.lines[1].strokeWidth = 1.0
    lp.lines[1].strokeDashArray = [3, 2]

    # Clean axes styling
    lp.xValueAxis.strokeColor = colors.HexColor("#D1D5DB")
    lp.yValueAxis.strokeColor = colors.HexColor("#D1D5DB")
    lp.xValueAxis.labels.fontName = "Helvetica"
    lp.xValueAxis.labels.fontSize = 6
    lp.xValueAxis.labels.fillColor = colors.HexColor("#6B7280")
    lp.yValueAxis.labels.fontName = "Helvetica"
    lp.yValueAxis.labels.fontSize = 6
    lp.yValueAxis.labels.fillColor = colors.HexColor("#6B7280")

    drawing.add(lp)
    return drawing


def build_report(
    path_or_buffer: Union[str, Path, io.BytesIO],
    db_path: Optional[str] = None,
) -> None:
    """
    Generate an executive PDF summary of astronaut health history and active alerts.
    Writes directly to file or BytesIO buffer.
    """
    doc = SimpleDocTemplate(
        path_or_buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=36,
        pageCompression=0,
    )

    styles = getSampleStyleSheet()

    # Custom Clean Typography Styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=2,
    )
    meta_style = ParagraphStyle(
        "ReportMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#475569"),
        spaceAfter=6,
    )
    banner_style = ParagraphStyle(
        "SimBanner",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        alignment=1,  # Centered
        textColor=colors.HexColor("#991B1B"),
    )
    section_h2 = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=6,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#334155"),
    )
    badge_style = ParagraphStyle(
        "BadgeText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=13,
        alignment=1,
        textColor=colors.white,
    )

    story = []

    # 1. Header Block
    story.append(Paragraph("AstroGuard Mission Telemetry Summary", title_style))
    current_date_str = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    meta_text = f"Subject: Astronaut ID-01 | Mission: Ares-V (Mars Transfer) | Report Date: {current_date_str}"
    story.append(Paragraph(meta_text, meta_style))

    # 2. Prominent Disclaimer Banner Box
    banner_table = Table(
        [[Paragraph("SIMULATED DATA - NOT FOR MEDICAL USE - ASTRONAUT HEALTH MONITORING PROTOTYPE", banner_style)]],
        colWidths=[540],
    )
    banner_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEE2E2")),
                ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#DC2626")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(banner_table)
    story.append(Spacer(1, 6))

    # Retrieve Readings from Database
    df = database.get_all_readings(db_path)

    # Edge Case 1: Empty Database
    if df.empty:
        empty_box = Table(
            [[Paragraph("<b>No telemetry data loaded yet.</b><br/>Please load mission CSV readings or load the Mars Demo Data to establish baseline and evaluate health anomalies.", body_style)]],
            colWidths=[540],
        )
        empty_box.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("PADDING", (0, 0), (-1, -1), 16),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]
            )
        )
        story.append(Spacer(1, 40))
        story.append(empty_box)
        doc.build(story, canvasmaker=NumberedCanvas)
        return

    # Check for baseline feasibility
    complete_df = df[df.apply(engine.is_row_complete, axis=1)].copy()
    if len(complete_df) < config.BASELINE_DAYS:
        # Edge Case 2: Incomplete Baseline Window
        calib_text = (
            f"<b>Baseline Calibration in Progress:</b> Currently recorded {len(complete_df)} complete daily readings. "
            f"A minimum of {config.BASELINE_DAYS} complete days is strictly required to establish individual baseline averages."
        )
        story.append(Paragraph(calib_text, body_style))
        story.append(Spacer(1, 10))
        doc.build(story, canvasmaker=NumberedCanvas)
        return

    # Compute baseline
    baseline = engine.compute_baseline(df)
    latest_row = complete_df.iloc[-1]
    latest_date = str(latest_row["date"])
    latest_day = int(latest_row["day"]) if "day" in latest_row and pd.notnull(latest_row["day"]) else len(complete_df)

    # Evaluate latest day alert
    latest_eval = engine.evaluate_day_alert(latest_row, baseline)
    current_level = latest_eval["level"] if latest_eval else config.LEVEL_NORMAL

    # Retrieve explanation: Strictly use cached explanation if available, or generate rule explanation.
    # NEVER call Gemini in report builder!
    latest_vitals = {k: float(latest_row[k]) for k in config.METRICS.keys()}
    facts_hash = llm.compute_facts_hash(latest_eval or {"level": current_level, "metrics_triggered": []}, latest_vitals)
    cached_exp = database.get_cached_explanation(latest_date, facts_hash, db_path)

    if cached_exp:
        explanation_text = cached_exp["text"]
        action_bullets = cached_exp["actions"]
    else:
        rule_exp = llm.generate_rule_explanation(latest_eval or {"level": current_level}, latest_vitals, baseline)
        explanation_text = rule_exp["text"]
        action_bullets = rule_exp["actions"]

    # 3. Current Health Status Banner & Actions
    if current_level == config.LEVEL_CRITICAL:
        badge_bg = colors.HexColor("#DC2626")
        status_label = "CRITICAL ALERT - MULTI-METRIC OR SEVERE DEVIATION"
    elif current_level == config.LEVEL_WATCH:
        badge_bg = colors.HexColor("#D97706")
        status_label = "WATCH ALERT - SINGLE-METRIC ANOMALY DETECTED"
    else:
        badge_bg = colors.HexColor("#16A34A")
        status_label = "HEALTH STATUS: NORMAL (WITHIN PERSONAL BASELINE)"

    status_table = Table(
        [[Paragraph(status_label, badge_style)]],
        colWidths=[540],
    )
    status_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), badge_bg),
                ("BOX", (0, 0), (-1, -1), 1.0, badge_bg),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(status_table)
    story.append(Spacer(1, 4))

    # Explanation and Suggested Actions
    actions_formatted = "<br/>".join([f"• {act}" for act in action_bullets[:2]])
    exp_table_content = [
        [
            Paragraph(f"<b>Summary Evaluation (Day {latest_day}, {latest_date}):</b><br/>{explanation_text}", body_style),
            Paragraph(f"<b>Recommended Action Protocol:</b><br/>{actions_formatted}", body_style),
        ]
    ]
    exp_table = Table(exp_table_content, colWidths=[310, 230])
    exp_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(exp_table)
    story.append(Spacer(1, 6))

    # 4. Baseline vs Current Vitals Table
    story.append(Paragraph("Physiological Baseline Comparison", section_h2))

    vitals_table_data = [
        [
            Paragraph("<b>Tracked Indicator</b>", body_style),
            Paragraph("<b>Baseline Average</b>", body_style),
            Paragraph("<b>Baseline Variation</b>", body_style),
            Paragraph("<b>Latest Value</b>", body_style),
            Paragraph("<b>Percent Shift</b>", body_style),
            Paragraph("<b>Condition</b>", body_style),
        ]
    ]

    for metric_name, m_info in config.METRICS.items():
        b_mean = baseline[metric_name]["mean"]
        b_std = baseline[metric_name]["std"]
        curr_val = float(latest_row[metric_name])
        pct_shift = ((curr_val - b_mean) / b_mean * 100.0) if b_mean else 0.0

        # Assess condition text in plain words (no z-score)
        direction = m_info["risk_direction"]
        is_elevated_risk = (direction == "up" and pct_shift > 15.0) or (direction == "down" and pct_shift < -5.0)

        condition_text = "Elevated Risk" if is_elevated_risk else "Nominal"

        vitals_table_data.append(
            [
                Paragraph(f"{m_info['label']} ({m_info['unit']})", body_style),
                Paragraph(f"{b_mean:.2f}", body_style),
                Paragraph(f"{b_std:.2f}", body_style),
                Paragraph(f"{curr_val:.2f}", body_style),
                Paragraph(f"{pct_shift:+.1f}%", body_style),
                Paragraph(condition_text, body_style),
            ]
        )

    v_table = Table(vitals_table_data, colWidths=[160, 80, 80, 75, 75, 70])
    v_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story.append(v_table)
    story.append(Spacer(1, 6))

    # 5. Four Small Trend Charts (2x2 Grid)
    story.append(Paragraph("30-Day Physiological Telemetry Trends (Dashed Line = Baseline Average)", section_h2))

    recent_30 = complete_df.tail(30).copy()
    day_indices = list(range(1, len(recent_30) + 1))

    c1 = create_trend_chart(
        list(zip(day_indices, recent_30["resting_hr"].astype(float))),
        baseline["resting_hr"]["mean"],
        "Resting Heart Rate",
        "bpm",
        colors.HexColor("#2563EB"),
    )
    c2 = create_trend_chart(
        list(zip(day_indices, recent_30["spo2"].astype(float))),
        baseline["spo2"]["mean"],
        "Blood Oxygen (SpO2)",
        "%",
        colors.HexColor("#0D9488"),
    )
    c3 = create_trend_chart(
        list(zip(day_indices, recent_30["sleep_hours"].astype(float))),
        baseline["sleep_hours"]["mean"],
        "Sleep Duration",
        "hrs",
        colors.HexColor("#6366F1"),
    )
    c4 = create_trend_chart(
        list(zip(day_indices, recent_30["stress_score"].astype(float))),
        baseline["stress_score"]["mean"],
        "Self-Reported Stress",
        "1-10",
        colors.HexColor("#D97706"),
    )

    charts_table = Table(
        [
            [c1, c2],
            [c3, c4],
        ],
        colWidths=[270, 270],
    )
    charts_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(charts_table)
    story.append(Spacer(1, 6))

    # 6. Recent Anomaly Alerts Log (Newest First, Capped at 10)
    story.append(Paragraph("Recent Anomaly Alerts History (Watch and Critical)", section_h2))

    all_alerts = database.get_alerts(db_path)
    # Filter only Watch and Critical alerts
    filtered_alerts = [a for a in all_alerts if a.get("level") in [config.LEVEL_WATCH, config.LEVEL_CRITICAL]]
    top_10_alerts = filtered_alerts[:10]

    alerts_table_data = [
        [
            Paragraph("<b>Date</b>", body_style),
            Paragraph("<b>Level</b>", body_style),
            Paragraph("<b>Shifted Indicators</b>", body_style),
            Paragraph("<b>Clinical Summary & Protocol</b>", body_style),
        ]
    ]

    if not top_10_alerts:
        alerts_table_data.append(
            [
                Paragraph("Nominal", body_style),
                Paragraph("None", body_style),
                Paragraph("All vitals stable", body_style),
                Paragraph("No Watch or Critical alerts recorded in evaluated mission timeline.", body_style),
            ]
        )
    else:
        for alert in top_10_alerts:
            m_list = alert.get("metrics_triggered", [])
            m_str = ", ".join([config.METRICS.get(m, {}).get("label", m) for m in m_list]) if m_list else "None"
            exp_snippet = alert.get("explanation", "")
            if len(exp_snippet) > 85:
                exp_snippet = exp_snippet[:82] + "..."

            alerts_table_data.append(
                [
                    Paragraph(alert["date"], body_style),
                    Paragraph(alert["level"], body_style),
                    Paragraph(m_str, body_style),
                    Paragraph(exp_snippet, body_style),
                ]
            )

    alerts_table = Table(alerts_table_data, colWidths=[65, 55, 160, 260])
    alerts_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story.append(alerts_table)

    if len(filtered_alerts) > 10:
        story.append(Spacer(1, 2))
        note_text = f"Showing the 10 most recent alerts. {len(filtered_alerts)} total Watch/Critical events recorded."
        story.append(Paragraph(note_text, meta_style))

    # 7. Mental Wellness Check-In Summary (Last 14 Days)
    story.append(Spacer(1, 4))
    story.append(Paragraph("Mental Wellness & Stress Trajectory", section_h2))

    checkins = database.get_checkins(db_path)
    last_14_stress = complete_df.tail(14)["stress_score"].astype(float).mean() if len(complete_df) >= 14 else complete_df["stress_score"].astype(float).mean()

    if checkins:
        latest_c = checkins[0]
        checkin_summary = (
            f"14-Day Stress Average: {last_14_stress:.1f} / 10 | Latest Questionnaire ({latest_c['date']}): "
            f"Mood: {latest_c['mood']}/5, Sleep Quality: {latest_c['sleep_quality']}/5, Workload Pacing: {latest_c['workload']}/5, "
            f"Physical Comfort: {latest_c['physical_comfort']}/5 (Calculated Stress Score: {latest_c['calculated_stress']:.1f})"
        )
    else:
        checkin_summary = (
            f"14-Day Stress Average: {last_14_stress:.1f} / 10 | No questionnaire check-in logs submitted yet."
        )

    wellness_table = Table(
        [[Paragraph(checkin_summary, body_style)]],
        colWidths=[540],
    )
    wellness_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(wellness_table)

    # Build the document using the 2-page numbered canvas
    doc.build(story, canvasmaker=NumberedCanvas)
