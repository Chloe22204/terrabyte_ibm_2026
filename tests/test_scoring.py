"""The model as the dashboard and API use it (ml/scoring.py + artifacts_logistic/)."""
import random

import numpy as np
import pytest

from app import dashboard
from conftest import needs_raw

scorer = dashboard.scorer
SAMPLE = random.Random(42).sample(list(dashboard.PATIENTS.values()), 200)


def inputs(p):
    return dashboard.model_inputs(dashboard.RAW[p["id"]])


@needs_raw
def test_live_score_matches_training_output():
    """The scorer must reproduce train.py's test-set results exactly (same features, fills, model)."""
    for p in SAMPLE:
        r = scorer.score(inputs(p))
        assert r["risk_6m"] == pytest.approx(p["risk"], abs=1e-6)
        assert r["risk_6m_interval_80"][0] == pytest.approx(p["lo"], abs=1e-6)
        assert r["risk_6m_interval_80"][1] == pytest.approx(p["hi"], abs=1e-6)
        assert r["decision"] == p["decision"]


@needs_raw
def test_outputs_are_valid_probabilities_and_curves():
    for p in SAMPLE:
        r = scorer.score(inputs(p))
        lo, hi = r["risk_6m_interval_80"]
        assert 0 <= lo <= hi <= 1 and 0 <= r["risk_6m"] <= 1
        s = np.array(r["survival_curve"]["survival"])
        assert s[0] == pytest.approx(1.0) and np.all(np.diff(s) <= 1e-12)        # never rises
        assert s[r["survival_curve"]["days"].index(180)] == pytest.approx(1 - r["risk_6m"], abs=1e-6)
        assert r["survival_6m"] == pytest.approx(1 - r["risk_6m"])
        assert r["decision"] in dashboard.DECISIONS


@needs_raw
def test_factors_are_grouped_per_variable():
    for p in SAMPLE[:50]:
        f = scorer.score(inputs(p))["top_factors"]
        names = [x["feature"] for x in f]
        assert len(f) == 5 and len(set(names)) == 5
        assert not any(n.endswith("_missing") or "_" in n for n in names)        # no raw dummy columns
        assert all(x["label"] and "_" not in x["label"] for x in f)              # plain-English labels
        mags = [abs(x["log_odds_contribution"]) for x in f]
        assert mags == sorted(mags, reverse=True)


def test_all_missing_inputs_still_score():
    r = scorer.score({})
    assert 0 <= r["risk_6m"] <= 1 and r["decision"] in dashboard.DECISIONS


def test_unknown_categories_are_handled():
    r = scorer.score({"age": 70, "sex": "unknown-value", "dzgroup": "Not a diagnosis", "ca": "maybe"})
    assert 0 <= r["risk_6m"] <= 1


def test_non_numeric_lab_value_is_rejected_not_guessed():
    with pytest.raises(ValueError):
        scorer.score({"age": "seventy"})


@pytest.mark.parametrize("age", [0, 18, 50, 80, 101, 120])
def test_extreme_but_numeric_values_stay_in_range(age):
    r = scorer.score({"age": age, "alb": 0.5, "bili": 40, "crea": 15, "meanbp": 0})
    assert 0 <= r["risk_6m"] <= 1 and np.isfinite(r["risk_6m"])


def test_risk_rises_with_physiology_score():
    base = {"age": 65, "dzgroup": "CHF", "sex": "male"}
    low = scorer.score({**base, "aps": 20, "sps": 15})["risk_6m"]
    high = scorer.score({**base, "aps": 90, "sps": 50})["risk_6m"]
    assert high > low


def test_scoring_is_deterministic():
    p = {"age": 77, "sex": "female", "dzgroup": "COPD", "alb": 2.2}
    assert scorer.score(p) == scorer.score(p)


def test_decision_thresholds_are_sane():
    th = dashboard.THRESHOLDS
    assert 0 < th["t_mid"] <= th["t_high"] < 1
    assert 0 < th["max_interval_width"] < 1
