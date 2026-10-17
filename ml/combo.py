import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

s = pd.read_csv("artifacts/patients_scored.csv")
d = pd.read_csv("data/support2.csv")
d.columns = [c.strip().lower().replace("_", ".") for c in d.columns]
if "id" not in d.columns:
    d.insert(0, "id", np.arange(1, len(d) + 1))
m = s.merge(d[["id", "prg6m", "death", "d.time"]], on="id")

y = np.where((m["death"] == 1) & (m["d.time"] <= 180), 1.0,
             np.where(m["d.time"] > 180, 0.0, np.nan))
k = ~np.isnan(y) & m["prg6m"].notna().to_numpy()

model = m["risk_6m"].to_numpy()[k]
doctor = 1 - m["prg6m"].to_numpy()[k]
combo = (model + doctor) / 2
yk = y[k]

print("test patients with a doctor estimate:", int(k.sum()))
print("model AUC   :", round(roc_auc_score(yk, model), 3))
print("doctor AUC  :", round(roc_auc_score(yk, doctor), 3))
print("combined AUC:", round(roc_auc_score(yk, combo), 3))