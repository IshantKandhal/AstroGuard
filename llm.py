"""
LLM Explainer module for AstroGuard.
Provides natural-language explanations and calm, non-diagnostic action items
for health anomaly alerts using Google Gemini (with an offline rule-based fallback).
"""

import os
import json
import hashlib
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import config
import database

# System instruction strictly enforcing clinical safety, tone, and brevity
SYSTEM_INSTRUCTION = (
    "You are AstroGuard, an AI assistant for astronaut health monitoring on long-duration space missions. "
    "Given numerical vital changes and alert level, provide: "
    "1. An explanation of at most 2 concise sentences describing what shifted compared to the astronaut's baseline. "
    "2. Exactly 1 or 2 bullet points with calm, non-diagnostic suggested actions. "
    "Rules: Never mention 'z-score' or raw statistical scores. Never diagnose medical conditions. "
    "Never prescribe medications. Keep the tone calm, mission-focused, and supportive."
)


def compute_facts_hash(alert: Dict[str, Any], vitals: Dict[str, float]) -> str:
    """
    Generate a deterministic SHA-256 hash of the alert's numerical facts.
    Includes level, triggered metrics, and rounded vital values.
    """
    facts = {
        "level": alert.get("level", config.LEVEL_NORMAL),
        "metrics_triggered": sorted(alert.get("metrics_triggered", [])),
        "vitals": {k: round(float(v), 2) for k, v in sorted(vitals.items()) if v is not None},
    }
    encoded = json.dumps(facts, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def get_explanation_label(source: str) -> str:
    """Return human-readable UI label for the explanation source."""
    return "AI explanation" if source == "gemini" else "Rule-based explanation"


def generate_rule_explanation(
    alert: Dict[str, Any],
    vitals: Dict[str, float],
    baseline: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    Deterministic rule-based explainer.
    Constructs natural language summaries using actual values and percentage changes.
    Never exposes raw z-scores.
    """
    level = alert.get("level", config.LEVEL_NORMAL)
    triggered = alert.get("metrics_triggered", [])

    if level == config.LEVEL_NORMAL:
        return {
            "text": "All physiological vitals are well within your personal mission baseline.",
            "actions": [
                "Continue standard mission schedule and scheduled hydration.",
                "Maintain regular sleep hygiene and exercise protocols.",
            ],
            "source": "rules",
        }

    # Build description of triggered metrics
    phrases = []
    actions: List[str] = []

    for metric_name in triggered:
        metric_info = config.METRICS.get(metric_name, {})
        label = metric_info.get("label", metric_name)
        val = vitals.get(metric_name)

        if val is None:
            continue

        base_mean = baseline[metric_name]["mean"] if baseline and metric_name in baseline else metric_info.get("default_mean", val)
        unit = metric_info.get("unit", "")

        # Compute percentage change
        if base_mean > 0:
            pct_change = ((val - base_mean) / base_mean) * 100.0
        else:
            pct_change = 0.0

        if metric_name == "resting_hr":
            direction_str = "above" if pct_change >= 0 else "below"
            phrases.append(f"Resting heart rate is {abs(pct_change):.0f}% {direction_str} your baseline at {val:.0f} {unit}")
            actions.append("Hydrate, perform a 10-minute seated rest period, and postpone strenuous EVA workouts.")
        elif metric_name == "spo2":
            phrases.append(f"Blood oxygen dropped to {val:.1f}{unit} ({abs(pct_change):.1f}% below baseline)")
            actions.append("Inspect cabin environmental oxygen levels and verify pulse oximeter sensor fit.")
        elif metric_name == "sleep_hours":
            phrases.append(f"Sleep fell to {val:.1f} {unit}")
            actions.append("Target a full rest cycle tonight; restrict cabin blue-spectrum light 1 hour before sleep.")
        elif metric_name == "stress_score":
            phrases.append(f"Self-reported stress rose to {val:.1f} on the 10-point scale")
            actions.append("Complete a guided breathing session and review upcoming workload pacing.")

    if not phrases:
        explanation_text = f"Anomalous telemetry pattern identified for {', '.join(triggered)}."
    elif len(phrases) == 1:
        explanation_text = f"{phrases[0]}."
    else:
        explanation_text = f"{'; '.join(phrases[:-1])}, and {phrases[-1]}."

    # Finalize actions based on severity
    if level == config.LEVEL_CRITICAL:
        # Critical alerts must ALWAYS prioritize contacting ground medical team
        critical_actions = ["Contact the ground medical team immediately via primary telemetry link."]
        for act in actions:
            if act not in critical_actions and len(critical_actions) < 3:
                critical_actions.append(act)
        final_actions = critical_actions
    else:
        # Watch level actions (max 2)
        final_actions = actions[:2] if actions else ["Monitor vitals and review telemetry on the next watch cycle."]

    return {
        "text": sanitize_text(explanation_text),
        "actions": [sanitize_text(a) for a in final_actions],
        "source": "rules",
    }


def sanitize_text(text: str) -> str:
    """Ensure the text never contains the word 'z-score' or raw z numbers."""
    cleaned = text.replace("z-score", "variation").replace("Z-score", "Variation")
    cleaned = cleaned.replace("z score", "variation").replace("Z score", "Variation")
    return cleaned


def call_gemini_api(
    alert: Dict[str, Any],
    vitals: Dict[str, float],
    baseline: Optional[Dict[str, Dict[str, float]]],
) -> Dict[str, Any]:
    """
    Call Gemini directly with an 8-second timeout.
    Only numeric telemetry and baseline means are passed (no check-in notes).
    """
    api_key = config.GEMINI_API_KEY
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY.")

    from google import genai
    from google.genai import types

    # Prepare structured numerical facts only
    level = alert.get("level", config.LEVEL_NORMAL)
    telemetry_summary = []
    for m in alert.get("metrics_triggered", []):
        val = vitals.get(m)
        b_mean = baseline[m]["mean"] if baseline and m in baseline else config.METRICS[m]["default_mean"]
        unit = config.METRICS[m]["unit"]
        pct = ((val - b_mean) / b_mean * 100.0) if b_mean else 0.0
        telemetry_summary.append(
            f"- {config.METRICS[m]['label']}: current={val} {unit}, baseline_mean={b_mean} {unit}, delta={pct:+.1f}%"
        )

    user_prompt = (
        f"Alert Level: {level}\n"
        f"Telemetry Anomalies:\n" + "\n".join(telemetry_summary) + "\n\n"
        "Provide: 1-2 sentence explanation and 1-2 bullet actions."
    )

    # Configure client with HTTP timeout if supported
    http_opts = None
    try:
        http_opts = types.HttpOptions(timeout=config.GEMINI_TIMEOUT_SECONDS)
        client = genai.Client(api_key=api_key, http_options=http_opts)
    except Exception:
        client = genai.Client(api_key=api_key)

    def _execute():
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
                max_output_tokens=300,
            ),
        )
        return response.text

    # Daemonized execution: never use 'with' context manager which waits for worker on exit
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_execute)
        raw_output = future.result(timeout=config.GEMINI_TIMEOUT_SECONDS)
    finally:
        # Non-blocking shutdown unblocks immediately upon TimeoutError
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=False)

    if not raw_output:
        raise ValueError("Empty response received from Gemini.")

    # Parse output into explanation text and action bullets
    lines = [line.strip() for line in raw_output.split("\n") if line.strip()]
    explanation_parts = []
    action_bullets = []

    for line in lines:
        if line.startswith(("-", "*", "•")) or (len(line) > 2 and line[0].isdigit() and line[1] in [".", ")"]):
            cleaned_bullet = line.lstrip("-*•0123456789.) ").strip()
            if cleaned_bullet:
                action_bullets.append(sanitize_text(cleaned_bullet))
        else:
            explanation_parts.append(line)

    explanation_text = " ".join(explanation_parts) if explanation_parts else "Telemetry shifts detected across tracked vitals."
    explanation_text = sanitize_text(explanation_text)

    # Ensure Critical alerts retain mandatory ground contact protocol
    if level == config.LEVEL_CRITICAL:
        mandatory = "Contact the ground medical team immediately via primary telemetry link."
        if not any("ground medical" in b.lower() for b in action_bullets):
            action_bullets.insert(0, mandatory)

    if not action_bullets:
        action_bullets = ["Monitor physiological vitals and rest in cabin."]

    return {
        "text": explanation_text,
        "actions": action_bullets[:2],
        "source": "gemini",
    }


def explain_alert(
    alert: Dict[str, Any],
    vitals: Optional[Dict[str, float]] = None,
    baseline: Optional[Dict[str, Dict[str, float]]] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main entry point for alert explanation.
    Checks database cache first. If not cached, tries Gemini with timeout,
    falling back to deterministic rule-based output on any exception.
    The alert level is strictly preserved and never modified.
    """
    date_str = str(alert.get("date", ""))
    level = alert.get("level", config.LEVEL_NORMAL)

    # If vitals dictionary not supplied, attempt to query from database
    if vitals is None:
        vitals = {}
        with database.get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT resting_hr, spo2, sleep_hours, stress_score FROM health_readings WHERE date = ?", (date_str,))
            row = cursor.fetchone()
            if row:
                for k in config.METRICS.keys():
                    if row[k] is not None:
                        vitals[k] = float(row[k])

    facts_hash = compute_facts_hash(alert, vitals)

    # 1. Check persistent cache
    cached = database.get_cached_explanation(date_str, facts_hash, db_path)
    if cached:
        return cached

    # 2. If Normal, return rules directly (no need to consume LLM quota)
    if level == config.LEVEL_NORMAL:
        explanation = generate_rule_explanation(alert, vitals, baseline)
        database.save_cached_explanation(date_str, facts_hash, explanation["text"], explanation["actions"], "rules", db_path)
        return explanation

    # 3. Try Gemini with timeout and error fallback
    explanation: Dict[str, Any]
    try:
        explanation = call_gemini_api(alert, vitals, baseline)
    except Exception:
        # Fall back gracefully to deterministic rule-based generator on any failure
        explanation = generate_rule_explanation(alert, vitals, baseline)

    # 4. Cache explanation
    database.save_cached_explanation(
        date_str,
        facts_hash,
        explanation["text"],
        explanation["actions"],
        explanation["source"],
        db_path,
    )

    return explanation

