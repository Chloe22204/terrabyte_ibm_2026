"""
Shared scoring logic used by the API (and the Streamlit app if it runs next to the artifacts).
Loads everything from plain JSON files produced by train.py. Needs only numpy (+ xgboost optional).
"""
import glob, json, os
import numpy as np
from coxph import CoxPH
from logit import LogisticModel, anchor_curve

CURVE_DAYS = [0, 15, 30, 60, 90, 120, 180, 270, 365]


class TimeToTalkScorer:
    def __init__(self, artifacts_dir):
        a = artifacts_dir
        self.model = LogisticModel.from_json(f"{a}/logit_model.json")   # main 6-month risk
        self.cox = CoxPH.from_json(f"{a}/cox_model.json")               # survival-curve shape only
        pre = json.load(open(f"{a}/preprocess.json"))
        self.features, self.fills = pre["features"], pre["fill_values"]
        cal = json.load(open(f"{a}/calibrator.json"))
        self.cal_x, self.cal_y = cal["x"], cal["y"]
        self.th = json.load(open(f"{a}/thresholds.json"))
        self.boots = [LogisticModel.from_json(p) for p in sorted(glob.glob(f"{a}/bootstraps/logit_*.json"))]
        self.xgb = None
        if os.path.exists(f"{a}/xgb_6m.json"):
            try:
                import xgboost as xgb
                self.xgb = xgb.XGBClassifier()
                self.xgb.load_model(f"{a}/xgb_6m.json")
            except ImportError:
                pass

    def _calib(self, r):
        return np.interp(r, self.cal_x, self.cal_y)

    def vectorize(self, patient: dict) -> np.ndarray:
        """Turn a raw patient record (SUPPORT2-style keys) into the model's feature vector,
        applying exactly the same fill rules and dummy encoding as training."""
        p = {k.lower().replace("_", "."): v for k, v in patient.items()}
        x = []
        for f in self.features:
            if f.endswith("_missing"):
                base = f[: -len("_missing")]
                x.append(1.0 if p.get(base) in (None, "", "nan") else 0.0)
            elif f in self.fills:
                v = p.get(f)
                x.append(float(v) if v not in (None, "", "nan") else float(self.fills[f]))
            else:  # dummy column like "dzgroup_CHF" or "sex_male"
                col, _, level = f.partition("_")
                x.append(1.0 if str(p.get(col, "unknown")) == level else 0.0)
        return np.array(x, dtype=float)

    def score(self, patient: dict) -> dict:
        x = self.vectorize(patient).reshape(1, -1)
        risk = float(self._calib(self.model.predict_proba(x)[0]))
        # Cox curve shape, rescaled so it passes through the main model's 6-month risk
        S = anchor_curve(self.cox.predict_survival(x, CURVE_DAYS), self.cox.predict_survival(x, [180])[:, 0], [risk])[0]
        if self.boots:
            br = [float(self._calib(m.predict_proba(x)[0])) for m in self.boots]
            lo, hi = float(np.percentile(br, 10)), float(np.percentile(br, 90))
        else:
            lo = hi = risk
        second = float(self.xgb.predict_proba(x)[0, 1]) if self.xgb is not None else None

        th = self.th
        disagree = second is not None and abs(second - risk) > th["max_model_disagreement"]
        if lo >= th["t_high"] and not disagree:
            decision = "Conversation due now"
        elif hi < th["t_mid"] and not disagree:
            decision = "No action"
        elif (hi - lo) > th["max_interval_width"] or disagree:
            decision = "Clinician judgement"
        elif risk >= th["t_mid"]:
            decision = "Review in 2 weeks"
        else:
            decision = "No action"

        contrib = self.model.contributions(x[0])
        top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:5]
        return {
            "risk_6m": risk, "risk_6m_interval_80": [lo, hi],
            "survival_2m": float(S[CURVE_DAYS.index(60)]),
            "survival_6m": 1 - risk,
            "survival_curve": {"days": CURVE_DAYS, "survival": [float(s) for s in S]},
            "second_opinion_risk_6m": second, "models_disagree": bool(disagree),
            "decision": decision,
            "top_factors": [{"feature": k, "effect": "raises risk" if v > 0 else "lowers risk",
                             "log_odds_contribution": float(v)} for k, v in top],
            "thresholds": {k: th[k] for k in ["t_high", "t_mid", "max_interval_width", "max_model_disagreement"]},
        }
