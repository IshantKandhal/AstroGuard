"""
Simulated Mars Mission Health Data Generator for AstroGuard.
Generates a realistic 60-day telemetry timeline including:
- Days 1-30: Clean personal baseline calibration
- Days 31-34: Stable normal operations
- Day 35: Isolated post-EVA exercise HR blip (triggers Watch alert, target risk_z ~ 2.5)
- Days 36-44: Return to normal
- Days 45-48: Severe Martian dust storm disruption (triggers Critical alert)
- Days 49-51: Medical intervention and gradual recovery
- Days 52-60: Restored normal baseline health
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Optional
import numpy as np
import pandas as pd

import config


def generate_clipped_normal(
    mean: float,
    std: float,
    clip_multiplier: float,
    rng: np.random.Generator,
) -> float:
    """
    Generate a random normal sample bounded by +/- clip_multiplier * std.

    Simulation Design Note:
    In unclipped Gaussian random distributions, values exceed |z| > 2.0 approximately
    4.5% of the time purely by statistical chance. Across a 30-day baseline or normal mission days,
    pure random noise would generate accidental Watch alerts on healthy days.
    Clipping the noise to [-1.5 * std, +1.5 * std] preserves natural physiological variability
    while ensuring that baseline days and healthy periods strictly reflect Normal status.
    """
    raw_noise = rng.normal(0.0, std)
    max_bound = clip_multiplier * std
    clipped_noise = float(np.clip(raw_noise, -max_bound, max_bound))
    return float(mean + clipped_noise)


def generate_mars_data(
    seed: int = 42,
    total_days: int = config.SIMULATION_TOTAL_DAYS,
    save_csv: bool = True,
    csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate 60-day Mars mission dataset according to the mission storyline.
    Uses sample standard deviation (ddof=1) to calibrate Day 35 blip relative
    to the generated baseline.
    """
    rng = np.random.default_rng(seed)
    start_dt = datetime.strptime(config.SIMULATION_START_DATE, "%Y-%m-%d")

    records = []

    # Phase 1: Generate Days 1-30 (Clean personal baseline calibration)
    for day in range(1, config.BASELINE_DAYS + 1):
        date_str = (start_dt + timedelta(days=day - 1)).strftime("%Y-%m-%d")
        record = {"day": day, "date": date_str}

        for metric_name, m_info in config.METRICS.items():
            val = generate_clipped_normal(
                mean=m_info["default_mean"],
                std=m_info["default_sd"],
                clip_multiplier=config.SIMULATION_CLIP_STD_DEV,
                rng=rng,
            )
            record[metric_name] = round(val, 2)

        records.append(record)

    # Compute baseline sample statistics (ddof=1) from days 1-30
    baseline_df = pd.DataFrame(records)
    baseline_stats: Dict[str, Dict[str, float]] = {}
    for metric_name in config.METRICS.keys():
        s_series = baseline_df[metric_name]
        baseline_stats[metric_name] = {
            "mean": float(s_series.mean()),
            "std": float(s_series.std(ddof=1)),
        }

    # Phase 2: Generate Days 31 to total_days
    # Save Day 48 values for gradual recovery interpolation
    last_storm_vals: Dict[str, float] = {}

    for day in range(config.BASELINE_DAYS + 1, total_days + 1):
        date_str = (start_dt + timedelta(days=day - 1)).strftime("%Y-%m-%d")
        record = {"day": day, "date": date_str}

        if day == config.SIMULATION_HR_BLIP_DAY:
            # Day 35: Target HR risk_z = 2.5 (Watch zone: 2.0 < z <= 3.0)
            hr_mean = baseline_stats["resting_hr"]["mean"]
            hr_std = baseline_stats["resting_hr"]["std"]
            record["resting_hr"] = round(hr_mean + config.SIMULATION_HR_BLIP_Z * hr_std, 2)

            # All other metrics stay normal with clipped noise calibrated to baseline
            for metric_name in ["spo2", "sleep_hours", "stress_score"]:
                val = generate_clipped_normal(
                    mean=baseline_stats[metric_name]["mean"],
                    std=baseline_stats[metric_name]["std"],
                    clip_multiplier=config.SIMULATION_CLIP_STD_DEV,
                    rng=rng,
                )
                record[metric_name] = round(val, 2)

        elif config.SIMULATION_STORM_START_DAY <= day <= config.SIMULATION_STORM_END_DAY:
            # Days 45-48: Martian Dust Storm event (Compound Critical Alert)
            hr_delta = rng.uniform(config.SIMULATION_STORM_HR_MIN, config.SIMULATION_STORM_HR_MAX)
            spo2_val = rng.uniform(config.SIMULATION_STORM_SPO2_MIN, config.SIMULATION_STORM_SPO2_MAX)
            sleep_val = rng.uniform(config.SIMULATION_STORM_SLEEP_MIN, config.SIMULATION_STORM_SLEEP_MAX)
            stress_val = rng.uniform(config.SIMULATION_STORM_STRESS_MIN, config.SIMULATION_STORM_STRESS_MAX)

            record["resting_hr"] = round(baseline_stats["resting_hr"]["mean"] + hr_delta, 2)
            record["spo2"] = round(spo2_val, 2)
            record["sleep_hours"] = round(sleep_val, 2)
            record["stress_score"] = round(stress_val, 2)

            if day == config.SIMULATION_STORM_END_DAY:
                last_storm_vals = {
                    "resting_hr": record["resting_hr"],
                    "spo2": record["spo2"],
                    "sleep_hours": record["sleep_hours"],
                    "stress_score": record["stress_score"],
                }

        elif (config.SIMULATION_STORM_END_DAY + 1) <= day <= config.SIMULATION_RECOVERY_END_DAY:
            # Days 49-51: Gradual de-escalation timeline (Day 49 Critical -> Day 50 Watch -> Day 51 Normal)
            if day == 49:
                # Day 49 (Critical): 50% of peak storm deviation across all vitals
                for metric_name in config.METRICS.keys():
                    b_mean = baseline_stats[metric_name]["mean"]
                    b_std = baseline_stats[metric_name]["std"]
                    storm_v = last_storm_vals.get(metric_name, b_mean)
                    val = b_mean + 0.50 * (storm_v - b_mean)
                    jitter = float(np.clip(rng.normal(0, b_std * 0.05), -0.1 * b_std, 0.1 * b_std))
                    record[metric_name] = round(val + jitter, 2)

            elif day == 50:
                # Day 50 (Watch): Cardiovascular recovery lags (HR at 2.4 * baseline_sd in Watch zone),
                # while SpO2, sleep, and stress have improved towards normal (1.2 * baseline_sd)
                b_hr_mean = baseline_stats["resting_hr"]["mean"]
                b_hr_std = baseline_stats["resting_hr"]["std"]
                record["resting_hr"] = round(b_hr_mean + 2.40 * b_hr_std, 2)

                b_spo2_mean = baseline_stats["spo2"]["mean"]
                b_spo2_std = baseline_stats["spo2"]["std"]
                record["spo2"] = round(b_spo2_mean - 1.20 * b_spo2_std, 2)

                b_sleep_mean = baseline_stats["sleep_hours"]["mean"]
                b_sleep_std = baseline_stats["sleep_hours"]["std"]
                record["sleep_hours"] = round(b_sleep_mean - 1.20 * b_sleep_std, 2)

                b_stress_mean = baseline_stats["stress_score"]["mean"]
                b_stress_std = baseline_stats["stress_score"]["std"]
                record["stress_score"] = round(b_stress_mean + 1.20 * b_stress_std, 2)

            elif day == 51:
                # Day 51 (Normal): Almost fully recovered (5% of storm deviation), within normal variation
                for metric_name in config.METRICS.keys():
                    b_mean = baseline_stats[metric_name]["mean"]
                    b_std = baseline_stats[metric_name]["std"]
                    storm_v = last_storm_vals.get(metric_name, b_mean)
                    val = b_mean + 0.05 * (storm_v - b_mean)
                    jitter = float(np.clip(rng.normal(0, b_std * 0.05), -0.1 * b_std, 0.1 * b_std))
                    record[metric_name] = round(val + jitter, 2)

        else:
            # Normal days (Days 31-34, 36-44, and 52-60)
            # Use baseline_stats mean and empirical sample std (ddof=1) with +/-1.5 clip
            # to guarantee that normal day risk_z never exceeds 1.5 (avoiding accidental Watch triggers)
            for metric_name in config.METRICS.keys():
                b_mean = baseline_stats[metric_name]["mean"]
                b_std = baseline_stats[metric_name]["std"]
                val = generate_clipped_normal(
                    mean=b_mean,
                    std=b_std,
                    clip_multiplier=config.SIMULATION_CLIP_STD_DEV,
                    rng=rng,
                )
                record[metric_name] = round(val, 2)

        records.append(record)

    df = pd.DataFrame(records)

    if save_csv:
        output_file = csv_path or (config.BASE_DIR / "data" / "mars_demo.csv")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)

    return df


if __name__ == "__main__":
    generated_df = generate_mars_data()
    print(f"Generated {len(generated_df)} rows of Mars demo data.")
