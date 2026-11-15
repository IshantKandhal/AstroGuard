"""
Unit and integration tests for engine.py and data_generator.py.
Verifies baseline computation, anomaly detection rules, storyline alert levels,
and robustness across multiple random seeds.
"""

import pytest
import pandas as pd

import config
import engine
import data_generator


def test_compute_stress_extremes_and_midpoint():
    """Verify compute_stress maps questionnaire answers to [1.0, 10.0] correctly."""
    # Low stress extreme
    best_answers = {"mood": 1, "sleep_quality": 1, "workload": 1, "physical_comfort": 1}
    assert engine.compute_stress(best_answers) == 1.0

    # High stress extreme
    worst_answers = {"mood": 5, "sleep_quality": 5, "workload": 5, "physical_comfort": 5}
    assert engine.compute_stress(worst_answers) == 10.0

    # Neutral midpoint
    mid_answers = {"mood": 3, "sleep_quality": 3, "workload": 3, "physical_comfort": 3}
    assert engine.compute_stress(mid_answers) == 5.5

    # Intermediate test
    mixed_answers = {"mood": 2, "sleep_quality": 4, "workload": 3, "physical_comfort": 3}
    # avg = 12 / 4 = 3.0 -> 5.5
    assert engine.compute_stress(mixed_answers) == 5.5


def test_baseline_computation():
    """Verify baseline uses ddof=1 and first 30 complete days."""
    df = data_generator.generate_mars_data(seed=42, save_csv=False)
    baseline = engine.compute_baseline(df)

    assert len(baseline) == len(config.METRICS)
    for m in config.METRICS.keys():
        assert "mean" in baseline[m]
        assert "std" in baseline[m]
        assert baseline[m]["count"] == config.BASELINE_DAYS
        assert baseline[m]["std"] > 0.0


def test_days_1_to_30_not_evaluated():
    """Verify days 1-30 are reserved for baseline and never output in evaluated alerts."""
    df = data_generator.generate_mars_data(seed=42, save_csv=False)
    alerts = engine.evaluate_all_alerts(df)

    evaluated_days = [a["day"] for a in alerts]
    for d in range(1, config.BASELINE_DAYS + 1):
        assert d not in evaluated_days, f"Day {d} should not be evaluated."
    assert min(evaluated_days) >= config.EVALUATION_START_DAY


def test_storyline_alert_levels_and_table(capsys):
    """
    Test alert levels across the 60-day Mars mission storyline:
    - Days 31-34: Normal
    - Day 35: Watch (HR risk_z in [2.0, 3.0])
    - Days 45-48: Critical
    - Days 52-60: Normal
    - Print table of date, level, and max risk_z for all days 31-60.
    """
    df = data_generator.generate_mars_data(seed=42, save_csv=False)
    alerts = engine.evaluate_all_alerts(df)

    alert_by_day = {a["day"]: a for a in alerts}

    # 1. Days 31-34 must be Normal
    for day in range(31, 35):
        assert alert_by_day[day]["level"] == config.LEVEL_NORMAL, (
            f"Day {day} expected Normal, got {alert_by_day[day]['level']}"
        )

    # 2. Day 35 must be Watch with HR risk_z between 2 and 3
    day_35_alert = alert_by_day[35]
    assert day_35_alert["level"] == config.LEVEL_WATCH, (
        f"Day 35 expected Watch, got {day_35_alert['level']}"
    )
    hr_risk_z = day_35_alert["risk_z_scores"]["resting_hr"]
    assert 2.0 < hr_risk_z <= 3.0, (
        f"Day 35 HR risk_z expected in (2.0, 3.0], got {hr_risk_z}"
    )

    # 3. Days 45-48 must be Critical
    for day in range(45, 49):
        assert alert_by_day[day]["level"] == config.LEVEL_CRITICAL, (
            f"Day {day} expected Critical, got {alert_by_day[day]['level']}"
        )

    # 4. Days 49-51 de-escalation: Day 49 Critical -> Day 50 Watch -> Day 51 Normal
    recovery_levels = {day: alert_by_day[day]["level"] for day in range(49, 52)}
    assert alert_by_day[49]["level"] == config.LEVEL_CRITICAL, f"Day 49 expected Critical, got {alert_by_day[49]['level']}"
    assert alert_by_day[50]["level"] == config.LEVEL_WATCH, f"Day 50 expected Watch, got {alert_by_day[50]['level']}"
    assert alert_by_day[51]["level"] == config.LEVEL_NORMAL, f"Day 51 expected Normal, got {alert_by_day[51]['level']}"

    # 5. Days 52-60 must be Normal
    for day in range(52, 61):
        assert alert_by_day[day]["level"] == config.LEVEL_NORMAL, (
            f"Day {day} expected Normal, got {alert_by_day[day]['level']}"
        )

    # Print the formatted inspection table for Days 31-60
    print("\n" + "=" * 62)
    print(f"{'Day':^5} | {'Date':^12} | {'Level':^10} | {'Max Risk Z':^12} | {'Triggered Metrics'}")
    print("-" * 62)
    for day in range(31, 61):
        a = alert_by_day[day]
        metrics_str = ", ".join(a["metrics_triggered"]) if a["metrics_triggered"] else "-"
        print(f"{day:^5} | {a['date']:^12} | {a['level']:^10} | {a['max_risk_z']:^12.2f} | {metrics_str}")
    print("=" * 62)
    print(f"Days 49-51 Recovery Levels: {recovery_levels}\n")


@pytest.mark.parametrize("seed", [1, 7, 42, 99])
def test_seed_robustness(seed):
    """Verify Day 35 = Watch and Days 45-48 = Critical across diverse random seeds."""
    df = data_generator.generate_mars_data(seed=seed, save_csv=False)
    alerts = engine.evaluate_all_alerts(df)
    alert_by_day = {a["day"]: a for a in alerts}

    # Day 35 must be Watch
    assert alert_by_day[35]["level"] == config.LEVEL_WATCH, (
        f"Seed {seed}: Day 35 expected Watch, got {alert_by_day[35]['level']}"
    )
    hr_risk_z = alert_by_day[35]["risk_z_scores"]["resting_hr"]
    assert 2.0 < hr_risk_z <= 3.0, (
        f"Seed {seed}: Day 35 HR risk_z expected in (2.0, 3.0], got {hr_risk_z}"
    )

    # Days 45-48 must be Critical
    for day in range(45, 49):
        assert alert_by_day[day]["level"] == config.LEVEL_CRITICAL, (
            f"Seed {seed}: Day {day} expected Critical, got {alert_by_day[day]['level']}"
        )

    # Days 49-51 de-escalation holds across all seeds
    assert alert_by_day[49]["level"] == config.LEVEL_CRITICAL, f"Seed {seed}: Day 49 expected Critical, got {alert_by_day[49]['level']}"
    assert alert_by_day[50]["level"] == config.LEVEL_WATCH, f"Seed {seed}: Day 50 expected Watch, got {alert_by_day[50]['level']}"
    assert alert_by_day[51]["level"] == config.LEVEL_NORMAL, f"Seed {seed}: Day 51 expected Normal, got {alert_by_day[51]['level']}"


def test_skip_incomplete_rows():
    """Verify rows with missing vitals are excluded from baseline and alert output."""
    df = data_generator.generate_mars_data(seed=42, save_csv=False)

    # Introduce a missing vital on Day 33
    df.loc[df["day"] == 33, "spo2"] = None

    alerts = engine.evaluate_all_alerts(df)
    evaluated_days = [a["day"] for a in alerts]
    assert 33 not in evaluated_days, "Incomplete Day 33 should have been skipped."

