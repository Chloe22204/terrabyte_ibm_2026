"""
TimeToTalk - training & evaluation pipeline on SUPPORT2.

Run:   python ml/train.py --data data/support2.csv --out artifacts
Needs: numpy, pandas, scipy, scikit-learn, matplotlib (xgboost optional)

Produces in --out:
  cox_model.json            model for the API (plain JSON, no pickle)
  calibrator.json           isotonic recalibration map (fit on validation set)
  thresholds.json           decision-layer thresholds + how they were chosen
  metrics.json              every number for the pitch
  patients_scored.csv       precomputed test-set results (demo fallback)
  fairness.csv              performance by age band / sex / race
  fig_*.png                 charts for the video
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from coxph import CoxPH, concordance_index, kaplan_meier

SEED = 42
HORIZONS = {"2m": 60, "6m": 180}

# ---------------------------------------------------------------------------
# 1. Columns
# ---------------------------------------------------------------------------
# Leaky / after-the-fact columns: NEVER model inputs.
#   outcome & relatives, physician estimates, DNR fields, original study scores,
#   length of stay, costs, TISS averaged over days 3-25, 2-month functional outcome.
LEAKY = ["death", "d.time", "hospdead", "prg2m", "prg6m", "dnr", "dnrday",
         "surv2m", "surv6m", "slos", "charges", "totcst", "totmcst", "avtisst", "sfdm2"]
# Not used as inputs for fairness / ethics reasons (kept for the fairness audit only)
EXCLUDED_SENSITIVE = ["race", "income", "edu"]

NUMERIC = ["age", "num.co", "scoma", "hday", "meanbp", "wblc", "hrt", "resp", "temp",
           "pafi", "alb", "bili", "crea", "sod", "ph", "glucose", "bun", "urine",
           "adlsc", "sps", "aps", "diabetes", "dementia"]
CATEGORICAL = ["sex", "dzgroup", "ca"]

# Missing-value rule (documented, defensible):
# For lab values the SUPPORT investigators recommend imputing a "normal" value, because a
# missing lab usually means the doctor thought it unnecessary (patient probably normal).
# Everything else: training-set median. Every imputed column also gets a "was missing" flag.
NORMAL_FILL = {"alb": 3.5, "pafi": 333.3, "bili": 1.01, "crea": 1.01,
               "bun": 6.51, "wblc": 9.0, "urine": 2502.0}


def load(path):
    df = pd.read_csv(path)
    # UCI sometimes uses underscores instead of dots
    df.columns = [c.strip().lower().replace("_", ".") for c in df.columns]
    if "id" not in df.columns:
        df.insert(0, "id", np.arange(1, len(df) + 1))
    missing = [c for c in ["death", "d.time"] + NUMERIC + CATEGORICAL if c not in df.columns]
    if missing:
        print("WARNING - columns not found, will be skipped:", missing)
    return df


def build_features(df, train_idx, fill_values=None):
    num = [c for c in NUMERIC if c in df.columns]
    cat = [c for c in CATEGORICAL if c in df.columns]
    X = df[num].astype(float).copy()
    if fill_values is None:
        fill_values = {}
        for c in num:
            fill_values[c] = NORMAL_FILL.get(c, float(X.loc[train_idx, c].median()))
    for c in num:
        if X[c].isna().any():
            X[c + "_missing"] = X[c].isna().astype(float)
        X[c] = X[c].fillna(fill_values[c])
    dummies = pd.get_dummies(df[cat].fillna("unknown").astype(str), drop_first=True, dtype=float)
    X = pd.concat([X, dummies], axis=1)
    return X, fill_values


def six_month_status(T, E, horizon=180):
    """1 = died within horizon, 0 = known alive at horizon, NaN = censored before horizon."""
    y = np.full(len(T), np.nan)
    y[(E == 1) & (T <= horizon)] = 1
    y[T > horizon] = 0
    return y


def calibration_table(pred_risk, T, E, horizon=180, bins=10):
    """Predicted vs Kaplan-Meier observed risk by decile (handles censoring)."""
    q = pd.qcut(pred_risk, bins, labels=False, duplicates="drop")
    rows = []
    for b in np.unique(q):
        m = q == b
        rows.append({"pred": float(np.mean(pred_risk[m])),
                     "obs": float(1 - kaplan_meier(T[m], E[m], horizon)), "n": int(m.sum())})
    return pd.DataFrame(rows)


def main(args):
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = load(args.data).reset_index(drop=True)
    T = df["d.time"].to_numpy(float)
    E = df["death"].to_numpy(int)

    # ---------------------------------------------------------------- split 60/20/20
    perm = rng.permutation(len(df))
    n_tr, n_va = int(0.6 * len(df)), int(0.2 * len(df))
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    X, fills = build_features(df, tr)
    feats = list(X.columns)
    Xv = X.to_numpy(float)
    json.dump({"fill_values": fills, "features": feats}, open(f"{args.out}/preprocess.json", "w"), indent=1)

    # ---------------------------------------------------------------- Cox model
    cox = CoxPH(penalizer=args.penalizer).fit(Xv[tr], T[tr], E[tr], feats)
    cox.to_json(f"{args.out}/cox_model.json")

    def risk6(model, idx):
        return 1 - model.predict_survival(Xv[idx], [180])[:, 0]

    r_va_raw, r_te_raw = risk6(cox, va), risk6(cox, te)
    lp_te = cox.linear_predictor(Xv[te])

    metrics = {"n_total": int(len(df)), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
               "n_features": len(feats)}
    metrics["cox_c_index_test"] = concordance_index(T[te], lp_te, E[te])

    y_va, y_te = six_month_status(T[va], E[va]), six_month_status(T[te], E[te])
    k_va, k_te = ~np.isnan(y_va), ~np.isnan(y_te)
    metrics["cox_auc_6m_test"] = roc_auc_score(y_te[k_te], r_te_raw[k_te])

    # ---------------------------------------------------------------- recalibration (fit on VALIDATION)
    # Only keep recalibration if it helps: fit on one half of validation, check on the other half.
    va_idx = np.where(k_va)[0]
    half = rng.permutation(va_idx)
    a, b = half[: len(half) // 2], half[len(half) // 2:]
    iso_a = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(r_va_raw[a], y_va[a])
    err_raw = calibration_table(r_va_raw[b], T[va][b], E[va][b])
    err_iso = calibration_table(iso_a.predict(r_va_raw[b]), T[va][b], E[va][b])
    mae_raw = float((err_raw.pred - err_raw.obs).abs().mean())
    mae_iso = float((err_iso.pred - err_iso.obs).abs().mean())
    use_recal = mae_iso < mae_raw
    metrics["recalibration_check_on_validation"] = {"mae_raw": mae_raw, "mae_isotonic": mae_iso,
                                                    "used": bool(use_recal)}
    if use_recal:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(r_va_raw[k_va], y_va[k_va])
        xs, ys = iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist()
    else:
        xs, ys = [0.0, 1.0], [0.0, 1.0]          # identity: raw Cox is already well calibrated
    json.dump({"x": xs, "y": ys}, open(f"{args.out}/calibrator.json", "w"))
    calib = lambda r: np.interp(r, xs, ys)
    r_va, r_te = calib(r_va_raw), calib(r_te_raw)
    metrics["auc_6m_test_after_recalibration"] = roc_auc_score(y_te[k_te], r_te[k_te])
    cal_before = calibration_table(r_te_raw, T[te], E[te])
    cal_after = calibration_table(r_te, T[te], E[te])
    metrics["calibration_mean_abs_error_before"] = float((cal_before.pred - cal_before.obs).abs().mean())
    metrics["calibration_mean_abs_error_after"] = float((cal_after.pred - cal_after.obs).abs().mean())

    # ---------------------------------------------------------------- bootstrap uncertainty (80% interval)
    boots_va, boots_te = [], []
    os.makedirs(f"{args.out}/bootstraps", exist_ok=True)
    for b in range(args.bootstraps):
        bi = rng.choice(tr, size=len(tr), replace=True)
        m = CoxPH(penalizer=args.penalizer).fit(Xv[bi], T[bi], E[bi], feats)
        m.to_json(f"{args.out}/bootstraps/cox_{b:02d}.json")   # used by the API for live intervals
        boots_va.append(calib(risk6(m, va)))
        boots_te.append(calib(risk6(m, te)))
    lo_va, hi_va = np.percentile(boots_va, [10, 90], axis=0)
    lo_te, hi_te = np.percentile(boots_te, [10, 90], axis=0)

    # ---------------------------------------------------------------- optional second opinion (XGBoost)
    second_va = second_te = None
    try:
        import xgboost as xgb
        ktr_y = six_month_status(T[tr], E[tr])
        ktr = ~np.isnan(ktr_y)
        clf = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, random_state=SEED)
        clf.fit(Xv[tr][ktr], ktr_y[ktr])
        second_va, second_te = clf.predict_proba(Xv[va])[:, 1], clf.predict_proba(Xv[te])[:, 1]
        metrics["xgb_auc_6m_test"] = roc_auc_score(y_te[k_te], second_te[k_te])
        clf.save_model(f"{args.out}/xgb_6m.json")
    except ImportError:
        print("xgboost not available - skipping second opinion")

    # ---------------------------------------------------------------- decision layer (thresholds from VALIDATION)
    yv, rv = y_va[k_va], r_va[k_va]
    grid = np.linspace(0.05, 0.95, 181)
    ppv = [(yv[rv >= t].mean() if (rv >= t).sum() >= 20 else np.nan) for t in grid]
    ok = [t for t, p in zip(grid, ppv) if p >= args.target_ppv]
    t_high = float(ok[0]) if ok else 0.6
    sens = [((rv >= t) & (yv == 1)).sum() / max((yv == 1).sum(), 1) for t in grid]
    t_mid = float(max([t for t, s in zip(grid, sens) if s >= args.target_sensitivity], default=0.3))
    t_mid = min(t_mid, t_high)
    width_va = hi_va - lo_va
    w_max = float(np.percentile(width_va, 90))
    thresholds = {
        "t_high": t_high, "t_mid": t_mid, "max_interval_width": w_max, "max_model_disagreement": 0.25,
        "how_chosen": {
            "t_high": f"lowest 6-month risk where >= {args.target_ppv:.0%} of flagged validation patients died within 6 months",
            "t_mid": f"highest threshold that still catches >= {args.target_sensitivity:.0%} of validation deaths within 6 months",
            "max_interval_width": "90th percentile of bootstrap 80%-interval width on validation (about 1 in 10 patients is 'too uncertain')",
            "max_model_disagreement": "Cox and XGBoost 6-month risks differ by more than 25 percentage points",
            "note": "These are design choices. A hospital can tune them to its own capacity and values.",
        }}
    json.dump(thresholds, open(f"{args.out}/thresholds.json", "w"), indent=1)

    def decide(r, lo, hi, second):
        out = []
        for i in range(len(r)):
            disagree = second is not None and abs(second[i] - r[i]) > thresholds["max_model_disagreement"]
            if lo[i] >= t_high and not disagree:
                out.append("Conversation due now")
            elif hi[i] < t_mid and not disagree:
                out.append("No action")
            elif (hi[i] - lo[i]) > w_max or disagree:
                out.append("Clinician judgement")
            elif r[i] >= t_mid:
                out.append("Review in 2 weeks")
            else:
                out.append("No action")
        return np.array(out)

    dec_te = decide(r_te, lo_te, hi_te, second_te)
    metrics["decision_counts_test"] = pd.Series(dec_te).value_counts().to_dict()
    for d in ["Conversation due now", "Review in 2 weeks", "No action", "Clinician judgement"]:
        m = (dec_te == d) & k_te
        metrics[f"died_6m_rate_[{d}]"] = float(y_te[m].mean()) if m.any() else None

    # ---------------------------------------------------------------- physicians vs model (test set)
    if "prg6m" in df.columns:
        phys = df["prg6m"].to_numpy(float)[te]
        mk = k_te & ~np.isnan(phys)
        phys_risk = 1 - phys
        metrics["physician_n"] = int(mk.sum())
        metrics["physician_auc_6m"] = roc_auc_score(y_te[mk], phys_risk[mk])
        metrics["model_auc_6m_same_patients"] = roc_auc_score(y_te[mk], r_te[mk])
        metrics["physician_mean_predicted_survival_6m"] = float(np.mean(phys[mk]))
        metrics["actual_survival_6m_same_patients"] = float(1 - y_te[mk].mean())
        metrics["model_mean_predicted_survival_6m"] = float(1 - np.mean(r_te[mk]))
        died = mk & (y_te == 1)
        metrics["pct_of_6m_deaths_physician_gave_>=50pct_survival"] = float(np.mean(phys[died] >= 0.5))
        metrics["pct_of_6m_deaths_model_flagged_due_or_review"] = float(
            np.mean(np.isin(dec_te[died], ["Conversation due now", "Review in 2 weeks"])))
        cal_phys = calibration_table(phys_risk[mk], T[te][mk], E[te][mk])
    else:
        cal_phys = None

    # ---------------------------------------------------------------- the problem in the data (whole dataset)
    if "dnr" in df.columns and "dnrday" in df.columns:
        died6 = (E == 1) & (T <= 180)
        dnr = df["dnr"].astype(str).str.lower()
        no_dnr = died6 & dnr.str.startswith("no").to_numpy()
        after = died6 & dnr.str.contains("after").to_numpy()
        days_before_death = (T - df["dnrday"].to_numpy(float))
        metrics["problem_deaths_within_6m"] = int(died6.sum())
        metrics["problem_pct_died_6m_with_no_dnr"] = float(no_dnr.sum() / died6.sum())
        metrics["problem_pct_dnr_written_in_last_7_days_of_life"] = float(
            np.sum(after & (days_before_death <= 7)) / died6.sum())
        metrics["problem_median_days_dnr_to_death"] = float(np.nanmedian(days_before_death[after]))

        # "days earlier" (imperfect proxy): our flag is available on study day 3
        dnr_te = dnr.to_numpy()[te]
        dnrday_te = df["dnrday"].to_numpy(float)[te]
        cand = (y_te == 1) & (dec_te == "Conversation due now") & (np.char.find(dnr_te.astype(str), "after") >= 0)
        earlier = dnrday_te[cand] - 3
        metrics["days_earlier_n"] = int(cand.sum())
        metrics["days_earlier_median"] = float(np.median(earlier)) if cand.any() else None
        metrics["days_earlier_pct_positive"] = float(np.mean(earlier > 0)) if cand.any() else None
        never = (y_te == 1) & (dec_te == "Conversation due now") & np.char.startswith(dnr_te.astype(str), "no")
        metrics["flagged_due_now_died_without_any_dnr"] = int(never.sum())

    # ---------------------------------------------------------------- fairness audit (test set)
    rows = []
    groups = {"age band": pd.cut(df["age"].to_numpy()[te], [0, 50, 65, 80, 200],
                                 labels=["<50", "50-64", "65-79", "80+"]).astype(str)}
    for c in ["sex", "race"]:
        if c in df.columns:
            groups[c] = df[c].fillna("unknown").astype(str).to_numpy()[te]
    for gname, g in groups.items():
        for level in np.unique(g):
            m = (g == level) & k_te
            if m.sum() < 30 or len(np.unique(y_te[m])) < 2:
                continue
            rows.append({"group": gname, "level": level, "n": int(m.sum()),
                         "auc": roc_auc_score(y_te[m], r_te[m]),
                         "mean_predicted_risk": float(r_te[m].mean()),
                         "observed_6m_death_rate": float(y_te[m].mean())})
    fair = pd.DataFrame(rows)
    fair.to_csv(f"{args.out}/fairness.csv", index=False)

    # ---------------------------------------------------------------- precomputed demo file
    top = []
    for i in te:
        c = cox.contributions(Xv[i])
        best = sorted(c.items(), key=lambda kv: -abs(kv[1]))[:3]
        top.append("; ".join(f"{k} {'+' if v > 0 else '-'}" for k, v in best))
    curve_times = [30, 60, 90, 180, 365]
    S = cox.predict_survival(Xv[te], curve_times)
    out = pd.DataFrame({"id": df["id"].to_numpy()[te], "age": df["age"].to_numpy()[te],
                        "sex": df["sex"].to_numpy()[te] if "sex" in df else None,
                        "dzgroup": df["dzgroup"].to_numpy()[te] if "dzgroup" in df else None,
                        "risk_6m": r_te, "risk_6m_low": lo_te, "risk_6m_high": hi_te,
                        "surv_2m": 1 - calib(1 - S[:, 1]), "surv_6m": 1 - r_te,
                        "second_opinion_6m": second_te, "decision": dec_te, "top_factors": top})
    for j, t in enumerate(curve_times):
        out[f"S_{t}d_raw"] = S[:, j]
    out.sort_values("risk_6m", ascending=False).to_csv(f"{args.out}/patients_scored.csv", index=False)

    # ---------------------------------------------------------------- charts
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    ax.plot(cal_before.pred, cal_before.obs, "o-", label="Cox (raw)")
    ax.plot(cal_after.pred, cal_after.obs, "o-", label="Cox (recalibrated)")
    if cal_phys is not None:
        ax.plot(cal_phys.pred, cal_phys.obs, "s-", label="Physicians (prg6m)")
    ax.set(xlabel="Predicted 6-month mortality", ylabel="Observed 6-month mortality (Kaplan-Meier)",
           title="Calibration on held-out test patients")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{args.out}/fig_calibration.png", dpi=160)

    if cal_phys is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        vals = [metrics["physician_mean_predicted_survival_6m"], metrics["model_mean_predicted_survival_6m"],
                metrics["actual_survival_6m_same_patients"]]
        ax.bar(["Doctors predicted", "TimeToTalk predicted", "Actually survived"], vals,
               color=["#c0504d", "#4f81bd", "#555555"])
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.0%}", ha="center")
        ax.set(ylim=(0, 1), ylabel="6-month survival", title="Average 6-month survival: predicted vs actual")
        fig.tight_layout(); fig.savefig(f"{args.out}/fig_doctors_vs_model.png", dpi=160)

    coefs = pd.Series(cox.coef_, index=feats).sort_values()
    coefs = pd.concat([coefs.head(8), coefs.tail(8)])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(coefs.index, coefs.values, color=["#4f81bd" if v < 0 else "#c0504d" for v in coefs.values])
    ax.set(title="Strongest risk factors (Cox, per 1 SD)", xlabel="log hazard ratio")
    fig.tight_layout(); fig.savefig(f"{args.out}/fig_risk_factors.png", dpi=160)

    json.dump(metrics, open(f"{args.out}/metrics.json", "w"), indent=1, default=float)
    print(json.dumps(metrics, indent=1, default=float))
    print("\nThresholds:", json.dumps(thresholds, indent=1))
    print("\nFairness:\n", fair.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/support2.csv")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--penalizer", type=float, default=0.1)
    p.add_argument("--bootstraps", type=int, default=30)
    p.add_argument("--target_ppv", type=float, default=0.75)
    p.add_argument("--target_sensitivity", type=float, default=0.80)
    main(p.parse_args())
