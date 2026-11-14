"""
Analytical Engine for AstroGuard.
Pure functions to calculate personal baseline statistics, directional risk z-scores,
and anomaly alert levels (Normal, Watch, Critical).
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

import config


def compute_stress(answers: Dict[str, int]) -> float:
    """
    Map 4 subjective wellness questionnaire answers (scored 1 to 5) to a continuous 1-10 stress score.

    Dimensions (1 = not at all, 5 = extremely; higher score always means MORE stress/worse):
    - mood: How tense or anxious do you feel?
    - sleep_quality: How poorly did you sleep?
    - workload: How overwhelming is your workload?
    - physical_comfort: How much physical discomfort do you have?

    Formula:
    average_raw = sum(answers.values()) / 4.0  # Range [1.0, 5.0]
    stress_score = 1.0 + (average_raw - 1.0) / 4.0 * 9.0  # Range [1.0, 10.0]

    Extremes and midpoint:
    - All 1s (minimal stress / not at all): stress = 1.0
    - All 5s (maximal stress / extremely): stress = 10.0
    - All 3s (neutral midpoint / moderate): stress = 5.5
    """
    required_keys = ["mood", "sleep_quality", "workload", "physical_comfort"]
    vals = []
    for k in required_keys:
        v = int(answers.get(k, 3))
        v = max(1, min(5, v))  # Clamp strictly to [1, 5]
        vals.append(v)

    raw_avg = sum(vals) / float(len(vals))
    scaled_stress = 1.0 + ((raw_avg - 1.0) / 4.0) * 9.0
    return round(float(scaled_stress), 1)


def get_required_vital_columns() -> List[str]:
    """Return list of required vital indicator column names."""
    return list(config.METRICS.keys())


def is_row_complete(row: pd.Series) -> bool:
    """Check whether all required vital metrics are present and non-null."""
    for col in get_required_vital_columns():
        val = row.get(col)
        if pd.isna(val) or val is None:
            return False
    return True


def compute_baseline(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate baseline mean and sample standard deviation (ddof=1)
    from the first BASELINE_DAYS complete records.
    Skips incomplete rows.
    """
    if df.empty:
        raise ValueError("Cannot compute baseline from empty dataset.")

    # Filter complete rows only
    complete_df = df[df.apply(is_row_complete, axis=1)].copy()

    # Sort by day or date if available
    if "day" in complete_df.columns:
        complete_df = complete_df.sort_values(by="day", ascending=True)
    elif "date" in complete_df.columns:
        complete_df = complete_df.sort_values(by="date", ascending=True)

    # Use records strictly within the baseline window (days 1 to BASELINE_DAYS)
    if "day" in complete_df.columns:
        baseline_slice = complete_df[complete_df["day"] <= config.BASELINE_DAYS]
    else:
        baseline_slice = complete_df.iloc[: config.BASELINE_DAYS]

    if len(baseline_slice) < 2:
        raise ValueError(
            f"Insufficient complete baseline readings ({len(baseline_slice)}). "
            f"Need at least 2 complete days to calculate sample standard deviation."
        )

    baseline: Dict[str, Dict[str, float]] = {}
    for col in get_required_vital_columns():
        values = baseline_slice[col].astype(float)
        mean_val = float(values.mean())
        # Sample standard deviation (ddof=1)
        std_val = float(values.std(ddof=1))
        # Guard against zero standard deviation
        if std_val == 0.0 or np.isnan(std_val):
            std_val = config.METRICS[col]["default_sd"]

        baseline[col] = {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "count": int(len(values)),
        }

    return baseline


def compute_metric_z_scores(
    value: float,
    mean: float,
    std: float,
    risk_direction: str,
) -> Tuple[float, float]:
    """
    Compute standard signed z-score and directional risk z-score.
    - standard_z = (value - mean) / std
    - risk_z:
        - for risk 'up': (value - mean) / std (higher is worse)
        - for risk 'down': (mean - value) / std (lower is worse)
    """
    if std <= 0.0:
        return 0.0, 0.0

    standard_z = (value - mean) / std
    if risk_direction == "up":
        risk_z = standard_z
    elif risk_direction == "down":
        risk_z = -standard_z
    else:
        risk_z = abs(standard_z)

    return round(float(standard_z), 3), round(float(risk_z), 3)


def evaluate_day_alert(
    row: pd.Series,
    baseline: Dict[str, Dict[str, float]],
) -> Optional[Dict[str, Any]]:
    """
    Evaluate anomaly alert level for a single day based on baseline stats.
    Returns None if row is incomplete or day < EVALUATION_START_DAY.
    """
    # Verify row completeness
    if not is_row_complete(row):
        return None

    # Check day index constraint
    day_val = row.get("day")
    if day_val is not None and not pd.isna(day_val):
        day_num = int(day_val)
        if day_num < config.EVALUATION_START_DAY:
            return None
    else:
        day_num = None

    date_str = str(row["date"])
    standard_z_map: Dict[str, float] = {}
    risk_z_map: Dict[str, float] = {}
    metrics_triggered: List[str] = []

    for metric_name, metric_info in config.METRICS.items():
        val = float(row[metric_name])
        b_mean = baseline[metric_name]["mean"]
        b_std = baseline[metric_name]["std"]
        r_dir = metric_info["risk_direction"]

        std_z, risk_z = compute_metric_z_scores(val, b_mean, b_std, r_dir)
        standard_z_map[metric_name] = std_z
        risk_z_map[metric_name] = risk_z

        # A metric is flagged if its risk z-score exceeds the Watch threshold
        if risk_z > config.WATCH_Z_THRESHOLD:
            metrics_triggered.append(metric_name)

    # Determine Alert Level
    # Rule 1 (Critical): any single metric has risk_z > 3.0 OR 3+ metrics have risk_z > 2.0
    has_severe_metric = any(rz > config.CRITICAL_Z_THRESHOLD for rz in risk_z_map.values())
    compound_triggered = len(metrics_triggered) >= config.CRITICAL_COMPOUND_COUNT

    if has_severe_metric or compound_triggered:
        level = config.LEVEL_CRITICAL
    elif len(metrics_triggered) > 0:
        level = config.LEVEL_WATCH
    else:
        level = config.LEVEL_NORMAL

    max_risk_z = max(risk_z_map.values()) if risk_z_map else 0.0

    return {
        "day": day_num,
        "date": date_str,
        "level": level,
        "metrics_triggered": metrics_triggered,
        "z_scores": standard_z_map,
        "risk_z_scores": risk_z_map,
        "max_risk_z": max_risk_z,
    }


def evaluate_all_alerts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Compute baseline from first 30 complete days and evaluate all days
    starting from EVALUATION_START_DAY onward.
    Skips days 1-30 and any incomplete rows.
    """
    if df.empty:
        return []

    baseline = compute_baseline(df)
    alerts: List[Dict[str, Any]] = []

    # Sort rows chronologically
    sort_col = "day" if "day" in df.columns else "date"
    sorted_df = df.sort_values(by=sort_col, ascending=True)

    for _, row in sorted_df.iterrows():
        eval_result = evaluate_day_alert(row, baseline)
        if eval_result is not None:
            alerts.append(eval_result)

    return alerts

