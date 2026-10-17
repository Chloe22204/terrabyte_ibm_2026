"""
Timely: train the "time to talk" model on SUPPORT2.

Run from ~/timely:  python3 train.py
Inputs : data/support2.csv
Outputs: model/timely.joblib        (models + threshold + metadata, for the API)
         data/test_patients.csv     (held-out patients, for the dashboard replay)
         model/metrics.txt          (all results, for README / video)
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

SEED = 42
TARGET_SENSITIVITY = 0.80   # catch at least 80% of patients who need the talk
os.makedirs("model", exist_ok=True)
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


# ---------------------------------------------------------------- 1. Load
df = pd.read_csv("data/support2.csv")
log("Loaded:", df.shape)

# ---------------------------------------------------------------- 2. Label
# 1 = died within 6 months (the conversation was needed)
# 0 = survived past 6 months
# drop = alive but followed < 180 days (outcome unknown)
died_6m = (df["death"] == 1) & (df["d.time"] <= 180)
known = died_6m | (df["d.time"] > 180)
df = df[known].copy()
df["label"] = died_6m[known].astype(int)
log("After dropping unknown outcomes:", df.shape, "| positive rate:", round(df["label"].mean(), 3))

# ---------------------------------------------------------------- 3. Features
# Leaky (outcome / after-the-fact): death, hospdead, d.time, slos, sfdm2, dnr, dnrday,
#   charges, totcst, totmcst, avtisst, surv2m, surv6m
# Benchmark only, never a feature: prg2m, prg6m (doctor's survival estimate)
# Fairness audit only, not a feature: race, income, edu
NUM = ["age", "num.co", "scoma", "sps", "aps", "hday", "diabetes", "dementia",
       "meanbp", "wblc", "hrt", "resp", "temp", "pafi", "alb", "bili", "crea",
       "sod", "ph", "glucose", "bun", "urine", "adlsc"]
CAT = ["sex", "dzgroup", "dzclass", "ca"]
FEATURES = NUM + CAT

# SUPPORT's documented "normal" fill values for missing labs
FILL = {"alb": 3.5, "pafi": 333.3, "bili": 1.01, "crea": 1.01,
        "bun": 6.51, "wblc": 9.0, "urine": 2502.0}


def prepare(X):
    X = X[FEATURES].copy()
    for c, v in FILL.items():
        X[c] = X[c].fillna(v)
    for c in CAT:
        X[c] = X[c].astype(str)
    return X


X = prepare(df)
y = df["label"].values

# ---------------------------------------------------------------- 4. Split 60/20/20
idx = np.arange(len(df))
i_train, i_tmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=SEED)
i_val, i_test = train_test_split(i_tmp, test_size=0.5, stratify=y[i_tmp], random_state=SEED)
log(f"Train {len(i_train)} | Val {len(i_val)} | Test {len(i_test)}")


def preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])


models = {
    "logistic": Pipeline([("prep", preprocessor()),
                          ("clf", LogisticRegression(max_iter=3000, C=0.5))]),
    "gradient_boosting": Pipeline([("prep", preprocessor()),
                                   ("clf", HistGradientBoostingClassifier(
                                       max_iter=300, learning_rate=0.05,
                                       max_leaf_nodes=15, random_state=SEED))]),
}

# ---------------------------------------------------------------- 5. Train + evaluate on validation
log("\n=== Validation ===")
val_auc = {}
for name, m in models.items():
    m.fit(X.iloc[i_train], y[i_train])
    p = m.predict_proba(X.iloc[i_val])[:, 1]
    val_auc[name] = roc_auc_score(y[i_val], p)
    log(f"{name:18s} AUROC {val_auc[name]:.3f} | AUPRC {average_precision_score(y[i_val], p):.3f}")

# prefer the simpler, explainable model unless boosting is clearly better
best = "gradient_boosting" if val_auc["gradient_boosting"] - val_auc["logistic"] > 0.02 else "logistic"
log("Chosen model:", best)

# ---------------------------------------------------------------- 6. Threshold (on validation, never test)
p_val = models[best].predict_proba(X.iloc[i_val])[:, 1]
pos = np.sort(p_val[y[i_val] == 1])
threshold = float(pos[int((1 - TARGET_SENSITIVITY) * len(pos))])
log(f"Threshold for >= {TARGET_SENSITIVITY:.0%} sensitivity: {threshold:.3f}")

# ---------------------------------------------------------------- 7. Test set (final, honest numbers)
log("\n=== Test (held out) ===")
p_test = models[best].predict_proba(X.iloc[i_test])[:, 1]
yt = y[i_test]
flag = p_test >= threshold
tp = (flag & (yt == 1)).sum(); fn = (~flag & (yt == 1)).sum()
fp = (flag & (yt == 0)).sum(); tn = (~flag & (yt == 0)).sum()
log(f"AUROC {roc_auc_score(yt, p_test):.3f} | AUPRC {average_precision_score(yt, p_test):.3f} | Brier {brier_score_loss(yt, p_test):.3f}")
log(f"Sensitivity {tp/(tp+fn):.3f} | Specificity {tn/(tn+fp):.3f} | PPV {tp/max(tp+fp,1):.3f} | Flagged {flag.mean():.1%}")

# ---------------------------------------------------------------- 8. AI vs doctors (prg6m)
log("\n=== AI vs doctor estimate (test patients with a doctor estimate) ===")
test_df = df.iloc[i_test]
has = test_df["prg6m"].notna().values
doc_risk = 1 - test_df["prg6m"].values[has]          # prg6m = doctor's P(survive 6 months)
log(f"n = {has.sum()}")
log(f"Doctor AUROC {roc_auc_score(yt[has], doc_risk):.3f} | AI AUROC {roc_auc_score(yt[has], p_test[has]):.3f}")
# at the same number of flags, how many needed talks does each catch?
k = int(flag[has].sum())
doc_flag = np.zeros(has.sum(), bool); doc_flag[np.argsort(-doc_risk)[:k]] = True
need = yt[has] == 1
log(f"Same flag budget ({k} patients): AI catches {(flag[has] & need).sum()}/{need.sum()}, "
    f"doctor catches {(doc_flag & need).sum()}/{need.sum()}")
# doctor overoptimism: needed the talk but doctor said >50% chance of surviving 6 months
over = need & (test_df["prg6m"].values[has] > 0.5)
log(f"Patients who died within 6m but doctor estimated >50% survival: {over.sum()}/{need.sum()} "
    f"| AI flagged {(over & flag[has]).sum()} of them")

# ---------------------------------------------------------------- 9. Fairness audit
log("\n=== Fairness (test set) ===")
audit = pd.DataFrame({"y": yt, "p": p_test, "flag": flag,
                      "sex": test_df["sex"].values, "race": test_df["race"].fillna("missing").values,
                      "age_group": pd.cut(test_df["age"], [0, 50, 65, 80, 120],
                                          labels=["<50", "50-64", "65-79", "80+"]).astype(str).values})
for col in ["sex", "race", "age_group"]:
    for g, d in audit.groupby(col):
        if len(d) < 30 or d["y"].nunique() < 2:
            continue
        sens = (d["flag"] & (d["y"] == 1)).sum() / max((d["y"] == 1).sum(), 1)
        log(f"{col:9s} {str(g):10s} n={len(d):4d} AUROC {roc_auc_score(d['y'], d['p']):.3f} "
            f"sensitivity {sens:.3f} flagged {d['flag'].mean():.1%}")

# ---------------------------------------------------------------- 10. Save for API + dashboard
lr = models["logistic"]
bundle = {
    "model": models[best], "model_name": best,
    "explainer": lr,                      # logistic coefficients give the reasons
    "explainer_features": list(lr.named_steps["prep"].get_feature_names_out()),
    "threshold": threshold, "features": FEATURES, "num": NUM, "cat": CAT, "fill": FILL,
}
joblib.dump(bundle, "model/timely.joblib")

out = test_df[["id"] + FEATURES].copy()
out["label"] = yt                         # for evaluation in the demo only, never sent to the API
out.to_csv("data/test_patients.csv", index=False)

with open("model/metrics.txt", "w") as f:
    f.write("\n".join(LOG))
log("\nSaved model/timely.joblib, data/test_patients.csv, model/metrics.txt")
