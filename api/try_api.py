"""
Send real SUPPORT2 patients to the running TimeToTalk API and print the answers.
Uses only the Python standard library + pandas.

  python api/try_api.py --token <TTT_API_TOKEN>                      # 4 demo patients (one per decision)
  python api/try_api.py --token <TOKEN> --ids 123 456 --doctor 0.4   # specific patients, with a doctor estimate
  python api/try_api.py --token wrong                                # shows the 401 (security screenshot)
"""
import argparse, json, urllib.request, urllib.error
import pandas as pd

LEAKY = ["death", "d.time", "hospdead", "prg2m", "prg6m", "dnr", "dnrday", "surv2m", "surv6m",
         "slos", "charges", "totcst", "totmcst", "avtisst", "sfdm2", "race", "income", "edu"]

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://localhost:8000")
p.add_argument("--token", required=True)
p.add_argument("--ids", nargs="*", type=int)
p.add_argument("--doctor", type=float, default=None, help="doctor's estimate of 6-month survival, 0-1")
p.add_argument("--data", default="data/support2.csv")
p.add_argument("--scored", default="artifacts/patients_scored.csv")
a = p.parse_args()

d = pd.read_csv(a.data)
d.columns = [c.strip().lower().replace("_", ".") for c in d.columns]
if "id" not in d.columns:
    d.insert(0, "id", range(1, len(d) + 1))

ids = a.ids
if not ids:   # one demo patient per decision, from the held-out test set
    s = pd.read_csv(a.scored)
    ids = [int(s[s.decision == dec].iloc[0]["id"]) for dec in
           ["Conversation due now", "Review in 2 weeks", "No action", "Clinician judgement"] if (s.decision == dec).any()]

for pid in ids:
    row = d[d["id"] == pid].iloc[0].drop(labels=[c for c in LEAKY if c in d.columns])
    patient = {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items() if k != "id"}
    body = json.dumps({"patient_id": f"SUPPORT2-{pid}", "clinician_id": "demo-doctor",
                       "patient": patient, "doctor_survival_6m": a.doctor}).encode()
    req = urllib.request.Request(f"{a.url}/score", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {a.token}"})
    try:
        r = json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e:
        print(f"patient {pid}: HTTP {e.code} {e.read().decode()}")
        continue
    lo, hi = r["risk_6m_interval_80"]
    print(f"\npatient {pid}  (ref {r['patient_ref']}, {r['latency_ms']:.1f} ms)")
    print(f"  decision : {r['decision']}")
    print(f"  6-month risk : {r['risk_6m']:.0%}  (80% range {lo:.0%}-{hi:.0%}),  second opinion {r['second_opinion_risk_6m']:.0%}"
          if r['second_opinion_risk_6m'] is not None else f"  6-month risk : {r['risk_6m']:.0%}  (80% range {lo:.0%}-{hi:.0%})")
    if r.get("combined_risk_6m") is not None:
        print(f"  doctor risk {r['doctor_risk_6m']:.0%}  ->  doctor + TimeToTalk combined : {r['combined_risk_6m']:.0%}")
    print("  top factors : " + ", ".join(f"{f['feature']} ({f['effect']})" for f in r["top_factors"][:3]))
