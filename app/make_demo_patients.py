"""
Build app/demo_patients.json for the clinician demo screen: 8 held-out TEST patients, two per decision tier.

  python app/make_demo_patients.py                                   # defaults below
  python app/make_demo_patients.py --artifacts artifacts --out app/demo_patients.json

- Patients come only from artifacts/patients_scored.csv (the test split), joined to data/support2.csv for raw values.
- The same leaky / sensitive columns as api/try_api.py are dropped from the model input.
- The doctor's own 6-month survival estimate (prg6m) is stored separately as doctor_survival_6m, never inside "patient".
- If the scorer can be loaded, each pick is re-scored to check its tier; without artifacts/bootstraps/ the
  80% range collapses, so "Clinician judgement" may not reproduce (see docs/SETUP.md step 2).
"""
import argparse, json, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LEAKY = ["death", "d.time", "hospdead", "prg2m", "prg6m", "dnr", "dnrday", "surv2m", "surv6m",
         "slos", "charges", "totcst", "totmcst", "avtisst", "sfdm2", "race", "income", "edu"]
# (decision, target 6-month risk) - two patients per tier, picked near these risks with different diagnoses
TIERS = [("Conversation due now", [0.85, 0.72]), ("Review in 2 weeks", [0.50, 0.42]),
         ("No action", [0.20, 0.10]), ("Clinician judgement", [0.55, 0.45])]

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/support2.csv")
p.add_argument("--scored", default="artifacts/patients_scored.csv")
p.add_argument("--artifacts", default=os.environ.get("TTT_ARTIFACTS", "artifacts"), help="for the re-score check")
p.add_argument("--out", default=os.path.join(HERE, "demo_patients.json"))
a = p.parse_args()

d = pd.read_csv(a.data)
d.columns = [c.strip().lower().replace("_", ".") for c in d.columns]
if "id" not in d.columns:
    d.insert(0, "id", range(1, len(d) + 1))
d = d.set_index("id")
s = pd.read_csv(a.scored)
s = s[s.id.isin(d.index)].copy()
s["prg6m"] = d.loc[s.id, "prg6m"].values
s["width"] = s.risk_6m_high - s.risk_6m_low
s["agree"] = (s.second_opinion_6m - s.risk_6m).abs() <= 0.25

scorer = None
try:
    sys.path.insert(0, os.path.join(HERE, "..", "ml"))
    from scoring import TimeToTalkScorer
    scorer = TimeToTalkScorer(a.artifacts)
    if not scorer.boots:
        print(f"warning: no bootstrap models in {a.artifacts}/bootstraps; tiers are checked without the 80% range")
except Exception as e:  # the JSON can still be built without the check
    print(f"warning: scorer not loaded ({e}); tiers not re-checked")


def to_py(v):
    if pd.isna(v):
        return None
    return v.item() if hasattr(v, "item") else v


def model_input(pid):
    row = d.loc[pid].drop(labels=[c for c in LEAKY if c in d.columns])
    return {k: to_py(v) for k, v in row.items()}


picked, used_dz = [], set()
for decision, targets in TIERS:
    g = s[(s.decision == decision) & s.prg6m.notna() & s.age.between(40, 88)]
    if decision == "Clinician judgement":
        # prefer "unsure because the range is wide" over "unsure because the models disagree",
        # so the tier reproduces even where the second-opinion model isn't installed
        g = g[g.agree & (g.width > 0.14)] if len(g[g.agree & (g.width > 0.14)]) >= 2 else g
    else:
        g = g[g.agree]
    for t in targets:
        cand = g[~g.id.isin([x["source_id"] for x in picked])].assign(dist=(g.risk_6m - t).abs()).sort_values("dist")
        fresh = cand[~cand.dzgroup.isin(used_dz)]
        for _, r in (fresh if len(fresh) else cand).iterrows():
            patient = model_input(int(r.id))
            if scorer is not None and scorer.score(patient)["decision"] != decision:
                continue      # keep only patients the live scorer puts in the same tier
            used_dz.add(r.dzgroup)
            prg = to_py(r.prg6m)
            picked.append({
                "source_id": int(r.id), "patient_id": f"SUPPORT2-{int(r.id)}",
                "age": int(r.age), "sex": r.sex, "diagnosis": r.dzgroup,
                "cancer": patient.get("ca"), "tier": decision,
                "doctor_survival_6m": None if prg is None else round(float(prg), 2),
                "patient": patient,
            })
            break
        else:
            print(f"warning: no patient found for {decision} near {t:.0%}")

out = {"source": "SUPPORT2 held-out test patients (artifacts/patients_scored.csv); de-identified research data",
       "dropped_columns": LEAKY, "patients": picked}
with open(a.out, "w") as f:
    json.dump(out, f, indent=1)
print(f"wrote {len(picked)} patients to {a.out}")
for x in picked:
    print(f"  {x['tier']:22s} {x['patient_id']:14s} {x['age']} {x['sex']:6s} {x['diagnosis']:18s} "
          f"doctor survival est {x['doctor_survival_6m']}")
