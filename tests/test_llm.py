"""
Unit tests for llm.py.
Verifies AI explanation generation, rule-based fallback, caching,
timeout resilience, and safety sanitization (zero exposure of raw z-scores).
All tests mock the Gemini client and never touch external network.
"""

import os
import gc
import time
import tempfile
from unittest.mock import patch, MagicMock
import pytest

import config
import database
import llm


@pytest.fixture
def temp_db():
    """Create and tear down a temporary database with llm_cache initialized."""
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


@pytest.fixture
def sample_alert():
    return {
        "date": "2026-02-04",
        "day": 35,
        "level": "Watch",
        "metrics_triggered": ["resting_hr"],
        "z_scores": {"resting_hr": 2.5},
        "risk_z_scores": {"resting_hr": 2.5},
    }


@pytest.fixture
def sample_vitals():
    return {
        "resting_hr": 66.5,
        "spo2": 98.0,
        "sleep_hours": 7.2,
        "stress_score": 2.8,
    }


@pytest.fixture
def sample_baseline():
    return {
        "resting_hr": {"mean": 62.0, "std": 1.8},
        "spo2": {"mean": 98.0, "std": 0.5},
        "sleep_hours": {"mean": 7.2, "std": 0.5},
        "stress_score": {"mean": 2.8, "std": 0.5},
    }


def test_no_api_key_falls_back_to_rules(temp_db, sample_alert, sample_vitals, sample_baseline):
    """When GEMINI_API_KEY is empty, explain_alert falls back to rules without raising errors."""
    with patch("config.GEMINI_API_KEY", ""):
        result = llm.explain_alert(
            sample_alert,
            vitals=sample_vitals,
            baseline=sample_baseline,
            db_path=temp_db,
        )

        assert result["source"] == "rules"
        assert len(result["actions"]) > 0
        assert "resting heart rate" in result["text"].lower()
        assert sample_alert["level"] == "Watch"


def test_simulated_exception_falls_back_to_rules(temp_db, sample_alert, sample_vitals, sample_baseline):
    """Network failure or API exception cleanly triggers rule fallback."""
    with patch("config.GEMINI_API_KEY", "dummy_key"):
        with patch("llm.call_gemini_api", side_effect=RuntimeError("Simulated network outage")):
            result = llm.explain_alert(
                sample_alert,
                vitals=sample_vitals,
                baseline=sample_baseline,
                db_path=temp_db,
            )
            assert result["source"] == "rules"
            assert "resting heart rate" in result["text"].lower()


def test_mocked_successful_gemini_response(temp_db, sample_alert, sample_vitals, sample_baseline):
    """Mocked successful Gemini output returns source 'gemini' and keeps level unchanged."""
    mock_gemini_output = {
        "text": "Your resting heart rate elevated slightly above baseline following the afternoon EVA.",
        "actions": [
            "Hydrate thoroughly with electrolyte mix.",
            "Schedule a 15-minute relaxation break before evening maintenance.",
        ],
        "source": "gemini",
    }

    with patch("config.GEMINI_API_KEY", "dummy_key"):
        with patch("llm.call_gemini_api", return_value=mock_gemini_output):
            result = llm.explain_alert(
                sample_alert,
                vitals=sample_vitals,
                baseline=sample_baseline,
                db_path=temp_db,
            )
            assert result["source"] == "gemini"
            assert result["text"] == mock_gemini_output["text"]
            assert result["actions"] == mock_gemini_output["actions"]
            # Alert level remains intact
            assert sample_alert["level"] == "Watch"


def test_caching_avoids_repeated_api_calls(temp_db, sample_alert, sample_vitals, sample_baseline):
    """Second call with identical facts uses cache and does not call Gemini."""
    mock_gemini_output = {
        "text": "Heart rate slightly elevated.",
        "actions": ["Rest in cabin."],
        "source": "gemini",
    }

    with patch("config.GEMINI_API_KEY", "dummy_key"):
        with patch("llm.call_gemini_api", return_value=mock_gemini_output) as mock_call:
            # First call: hits Gemini
            res1 = llm.explain_alert(sample_alert, sample_vitals, sample_baseline, db_path=temp_db)
            assert res1["source"] == "gemini"
            assert mock_call.call_count == 1

            # Second call: should hit cache
            res2 = llm.explain_alert(sample_alert, sample_vitals, sample_baseline, db_path=temp_db)
            assert res2["source"] == "gemini"
            assert res2["text"] == mock_gemini_output["text"]
            # call_count remains 1 because cache was used
            assert mock_call.call_count == 1


def test_changed_facts_triggers_new_call(temp_db, sample_alert, sample_vitals, sample_baseline):
    """When vitals or alert facts change (e.g. post check-in), a new call is triggered."""
    mock_gemini_output_1 = {"text": "Explanation 1", "actions": ["Action 1"], "source": "gemini"}
    mock_gemini_output_2 = {"text": "Explanation 2", "actions": ["Action 2"], "source": "gemini"}

    with patch("config.GEMINI_API_KEY", "dummy_key"):
        with patch("llm.call_gemini_api", side_effect=[mock_gemini_output_1, mock_gemini_output_2]) as mock_call:
            # First call with initial vitals
            res1 = llm.explain_alert(sample_alert, sample_vitals, sample_baseline, db_path=temp_db)
            assert res1["text"] == "Explanation 1"
            assert mock_call.call_count == 1

            # Change vitals (e.g. stress increased after check-in)
            modified_vitals = dict(sample_vitals)
            modified_vitals["resting_hr"] = 72.0
            modified_vitals["stress_score"] = 6.5

            res2 = llm.explain_alert(sample_alert, modified_vitals, sample_baseline, db_path=temp_db)
            assert res2["text"] == "Explanation 2"
            assert mock_call.call_count == 2


def test_no_z_score_or_raw_statistics_in_output(temp_db, sample_alert, sample_vitals, sample_baseline):
    """Verify generated explanation text never contains the word 'z-score' or raw z numbers."""
    with patch("config.GEMINI_API_KEY", ""):
        rule_result = llm.explain_alert(sample_alert, sample_vitals, sample_baseline, db_path=temp_db)
        combined_text = (rule_result["text"] + " " + " ".join(rule_result["actions"])).lower()

        assert "z-score" not in combined_text
        assert "z score" not in combined_text
        assert "z=" not in combined_text


def test_critical_alert_always_includes_ground_medical_team(temp_db, sample_baseline):
    """Critical alerts must always prioritize contacting ground medical team."""
    critical_alert = {
        "date": "2026-02-14",
        "day": 45,
        "level": "Critical",
        "metrics_triggered": ["resting_hr", "spo2", "sleep_hours", "stress_score"],
    }
    critical_vitals = {
        "resting_hr": 76.0,
        "spo2": 94.2,
        "sleep_hours": 4.6,
        "stress_score": 7.5,
    }

    with patch("config.GEMINI_API_KEY", ""):
        result = llm.explain_alert(critical_alert, critical_vitals, sample_baseline, db_path=temp_db)
        assert result["source"] == "rules"
        assert any("ground medical" in act.lower() for act in result["actions"])


def test_timeout_unblocks_immediately(temp_db, sample_alert, sample_vitals, sample_baseline):
    """
    Verify a hung/sleeping client call unblocks on timeout without freezing the thread.
    Short timeout: 0.5s, sleep: 3.0s. Must return rules in under 1.5 seconds.
    """
    def _slow_generate(*args, **kwargs):
        time.sleep(3.0)
        return MagicMock(text="Finished after sleep")

    with patch("config.GEMINI_API_KEY", "dummy_key"):
        with patch("config.GEMINI_TIMEOUT_SECONDS", 0.5):
            with patch("google.genai.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.models.generate_content.side_effect = _slow_generate
                mock_client_cls.return_value = mock_client

                t_start = time.time()
                result = llm.explain_alert(
                    sample_alert,
                    vitals=sample_vitals,
                    baseline=sample_baseline,
                    db_path=temp_db,
                )
                elapsed = time.time() - t_start

                assert elapsed < 1.5, f"Took too long ({elapsed:.2f}s), should have timed out within ~0.5s"
                assert result["source"] == "rules"
                assert "resting heart rate" in result["text"].lower()


def test_get_explanation_label():
    """Verify UI label helper."""
    assert llm.get_explanation_label("gemini") == "AI explanation"
    assert llm.get_explanation_label("rules") == "Rule-based explanation"

