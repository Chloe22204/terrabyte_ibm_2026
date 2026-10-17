# TimeToTalk — terrabyters_IBM Z Datathon 2026 Project

**AI that tells clinicians when a seriously ill patient is ready for the care-planning conversation, how sure it is, and how to begin.**
Everything, including the LLM, runs inside one isolated IBM LinuxONE (z17) server, so patient data never leaves it.

## Architecture

```
SUPPORT2 data ─► ml/train.py ─► artifacts/ (JSON models, thresholds, charts)
                                     │
                     ┌───────────────┴──────────────┐
                     ▼                              ▼
         api/ (FastAPI in Podman on LinuxONE)   ml/accelerator_model.py (MLP → ONNX → z17 AI accelerator)
          ├ logistic 6-month risk model + 20 bootstrap models (uncertainty)
          ├ Cox model for the survival-curve shape
          ├ XGBoost second opinion
          ├ decision layer: Due now / Review in 2 weeks / No action / Clinician judgement
          └ token auth, hashed patient IDs, audit log, non-root container
                     │
                     ▼
         genai/ IBM Granite (llama.cpp, local) + RAG over care-planning guidelines
                     │  → conversation prep sheet (EN / 中文 / Melayu / தமிழ்) + "ask why"
                     ▼
         app/ Streamlit clinician dashboard
```

## Run it

```bash
# 1. data: download SUPPORT2 from https://archive.ics.uci.edu/dataset/880 → data/support2.csv
# 2. train + evaluate (≈1 min on 3 vCPUs)
python ml/train.py --data data/support2.csv --out artifacts --bootstraps 20
# 3. optional: accelerator second-opinion model
python ml/accelerator_model.py --data data/support2.csv --out artifacts
# 4. API in a container
docker build -t timetotalk-api -f api/Containerfile .
docker run -d --name ttt -p 8000:8000 -e TTT_API_TOKEN=<token> -e TTT_ID_SALT=<salt> timetotalk-api
# or without a container (e.g. inside the Jupyter terminal):
TTT_API_TOKEN=<token> TTT_ARTIFACTS=artifacts nohup python -m uvicorn app:app --app-dir api --port 8000 &
curl localhost:8000/health
```

## Scoring API (runs on LinuxONE)

A FastAPI service (`api/app.py`) serves the model on the z17. It scores one patient in **about 1–2 ms**.

| Endpoint | What it returns |
|---|---|
| `GET /health` | Status, machine type (`s390x` on LinuxONE), number of loaded models |
| `POST /score` | Decision, 6-month risk with 80% range, second opinion, top factors in plain English, survival curve. If `doctor_survival_6m` is sent, also the **combined doctor + model risk** |
| `POST /benchmark?n=1000` | Scores one patient *n* times and reports patients per second |

**Example request**

```json
POST /score   (header: Authorization: Bearer <token>)
{ "patient_id": "3739", "patient": { "age": 80.1, "sex": "male", "dzgroup": "Coma", "scoma": 44, "...": "..." },
  "doctor_survival_6m": 0.4 }
```

**Example reply** (shortened)

```json
{ "decision": "Conversation due now", "risk_6m": 0.85, "risk_6m_interval_80": [0.82, 0.89],
  "combined_risk_6m": 0.72,
  "top_factors": [ { "label": "Difficulty with daily activities (0-7) 7.0", "effect": "raises risk" },
                   { "label": "Diagnosis: Coma", "effect": "raises risk" } ] }
```

**Security**
- **Token required:** every scoring call needs the bearer token. A missing or wrong token gets `401`.
- **Pseudonymised IDs:** patient IDs are SHA-256 hashed with a secret salt before logging. Raw IDs are never stored.
- **Audit log:** every decision is logged with time, clinician and decision. No clinical values are logged.
- **No outside AI:** inference (risk models and Granite) stays on the LinuxONE server.
- **Container:** the container runs as a non-root user (`api/Containerfile`).

**Try it**
- `python api/try_api.py --token <token>` scores 4 demo patients, one per decision.
- `api/API_demo.ipynb` is a notebook walkthrough of calling the API from Python.

## Proof it runs on IBM Z (LinuxONE z17)

All of these were captured on our LinuxONE Community Cloud server.

**1. The hardware:** architecture `s390x`, machine type **9175 (IBM z17)**, and our API reporting it runs there.

![IBM Z proof](screenshots/01_ibmz_s390x_z17.png)

**2. Four patients, four decisions,** each with a risk, an uncertainty range, a second opinion and plain-English reasons.

![Four demo patients](screenshots/02_four_patients.png)

**3. AI + clinician:** the doctor's own estimate (60% risk) blended with TimeToTalk (85%) gives a combined risk of 73%.

![Doctor plus AI](screenshots/03_doctor_plus_ai.png)

**4. Security:** a wrong token is rejected (`401`), and the audit log records only hashed patient IDs, the clinician and the decision. No clinical values are logged.

![Security](screenshots/04_security_401_audit.png)

**5. Speed:** 1,000 patients scored in 0.67 s, which is **about 1,490 patients per second (0.67 ms each)** on 3 virtual CPUs.

![Speed benchmark](screenshots/05_speed_benchmark.png)

**6. The on-chip AI accelerator:** `lscpu` lists the `nnpa` flag (Neural Network Processing Assist, the Telum II AI accelerator), so our server can use it.

![AI accelerator flag](screenshots/06_ai_accelerator_nnpa.png)

**7. Training and evaluation on the mainframe:** the full pipeline (main model, 20 bootstrap models, XGBoost, fairness audit) runs in **2.2 seconds** on the z17, and the doctor + AI result (**0.831** vs 0.786 doctor alone) is reproduced there.

![Training on z17](screenshots/07_training_and_doctor_plus_ai_on_z17.png)

## Decisions log (what we chose, what we rejected, and why)

| Decision | Why |
|---|---|
| **Logistic regression** for 6-month risk as the main model | SUPPORT2 follows every patient to death or past 6 months, so a model aimed directly at "died within 6 months" loses nothing to censoring. On the same test split it beat our Cox model (AUC 0.815 vs 0.803). Every factor is still explainable, so a clinician can check it. |
| Cox model kept only for the **survival-curve shape** | The dashboard shows a curve over a year. The Cox curve is rescaled to pass through the logistic 6-month risk, so the two never disagree. |
| Wrote our own numpy/scipy Cox model instead of `lifelines` | `lifelines` needs `autograd-gamma`, which failed to build in testing. Ours has no compiled dependencies, so it runs on s390x. |
| All models saved as plain JSON | No pickle files in the API (a pickle can run code when loaded), and the container needs only numpy. |
| **Rejected** deep learning as the main model | Little gain on about 9,000 patients and much harder to explain. Kept a small neural net only as a second opinion on the AI accelerator. |
| **Leaky columns excluded** (death, d.time, hospdead, prg2m/6m, dnr/dnrday, surv2m/6m, slos, costs, avtisst, sfdm2) | The test for each column: would a doctor know it on day 3? If not, it's out. |
| **Race, income and education are not model inputs** | Fairness. They are used only to audit performance across groups (`fairness.csv`). |
| Missing labs → SUPPORT investigators' "normal" values, others → training median, plus a "was missing" flag | A missing lab usually means the doctor didn't think it was needed. This is documented and reproducible. |
| 60/20/20 split stratified on 6-month status, seed 42; thresholds and recalibration use **validation only** | The test set is touched once, at the end. |
| Recalibration (Platt scaling) only kept if it **clearly** helps on a held-out half of validation (error down ≥ 20% and ≥ 0.005) | An earlier isotonic version passed a weaker check on validation but made test calibration worse. |
| **AI + clinician experiment**: combine model risk with the doctor's own estimate (`prg6m`), weights fit on validation | Tests whether the tool adds to clinical judgement rather than replacing it. The doctor estimate is never a model input otherwise. |
| **"Clinician judgement"** output when the uncertainty is wide or the two models disagree | The model should say when it doesn't know. |
| **Rejected** a "risk is rising fast" alert | SUPPORT2 has one snapshot per patient, so we can't validate trends. It's listed under next steps. |
| Granite runs locally, not a cloud LLM | Patient data never leaves LinuxONE. Granite only explains the model's decision and never changes the numbers. |

## Results

All numbers are on **1,821 held-out test patients** the model never saw during training. They're reproduced on the IBM z17 (LinuxONE), where the full training run takes about 2 seconds. Full output is in `artifacts/metrics.json`, `artifacts/fairness.csv` and the `fig_*.png` charts.

### The problem, in the data
Of **4,265** seriously ill patients who died within 6 months:
- **37%** never had a do-not-resuscitate (DNR) decision recorded
- **35%** had it recorded only in their **final 7 days** of life
- the median gap between the DNR decision and death was **5 days**

So **72%** of these patients either never had the decision recorded or had it only in their last week. A DNR order is our stand-in for a care-planning conversation, not the same thing.

### Model performance

| Measure | Result | In plain words |
|---|---|---|
| AUC, death within 6 months | **0.813** | 81% of the time, it ranks a patient who died above one who survived |
| C-index (over time) | 0.736 | Ranking quality across the whole follow-up |
| Calibration error | **2.4 percentage points** | When it says 60% risk, the real rate is close to 60% |
| XGBoost second opinion, AUC | 0.813 | An independent model agrees |

### Decision tiers: share who actually died within 6 months

| Recommendation | Patients | Died within 6 months |
|---|---|---|
| Conversation due now | 571 | **80%** |
| Clinician judgement (model unsure) | 141 (8%) | 50% |
| Review in 2 weeks | 371 | 47% |
| No action | 738 | **20%** |

"Due now", "review" and "clinician judgement" together reach **83%** of patients who died within 6 months.

### AI + clinician: better together
Tested on the 1,497 test patients whose doctor gave a survival estimate:

| | AUC | Deaths caught when each flags the same 763 patients |
|---|---|---|
| Doctor's estimate alone | 0.786 | 524 of 710 |
| TimeToTalk alone | 0.804 | 538 of 710 |
| **Doctor + TimeToTalk** | **0.833** | **560 of 710** |

TimeToTalk matches experienced doctors from routine day-3 data, and **combining the two beats either alone**. With the same workload, the combination catches **36 more** of the patients who died than doctors alone. The two make different mistakes, so the tool sharpens clinical judgement rather than replacing it. (The combination weights are fit on validation data. A simple 50/50 average gives 0.831, from `ml/combo.py`.)

### Fairness audit (test set)

| Group | AUC | Share of deaths flagged |
|---|---|---|
| Age <50 / 50–64 / 65–79 / 80+ | 0.84 / 0.80 / 0.78 / 0.85 | 66% / 71% / 76% / 85% |
| Female / male | 0.82 / 0.81 | 72% / 77% |
| Black / Hispanic / white | 0.82 / 0.81 / 0.81 | 66% / 52%* / 76% |

Ranking accuracy is similar across groups, and predicted risk matches observed risk within each group. The model **flags a smaller share of deaths among younger patients and Black and Hispanic patients**. (*There are only 57 Hispanic test patients, so that figure is unreliable.) A hospital would need to monitor this before using the tool. It's listed under next steps.

### What did *not* hold up (reported honestly)
- **"Doctors are too optimistic":** not true in this data. Doctors' average 6-month survival estimate (50%) was close to reality (53%).
- **"Flags patients days earlier":** compared with when the DNR was written, the median is only 1 day earlier, so we don't claim it.

## Limitations & next steps
- SUPPORT2 is 1989–94 US data. This is a proof of concept. Next step: retrain and validate on Singapore hospital records.
- Repeated measurements per patient → a real "risk is rising" trigger
- Close the fairness gap in how many deaths are flagged for younger and minority patients, e.g. group-aware thresholds, checked on local data
- Clinical pilot with the palliative care team, and link with Singapore's national advance care planning programme
- Fine-tune Granite on local care-planning guidance in the four official languages

## Ethics
Clinician-facing only. The model prompts and never decides. It is never shown to patients. Thresholds can be adjusted by each hospital.

## Open-source libraries & data
- SUPPORT2 dataset — https://archive.ics.uci.edu/dataset/880
- NumPy — https://numpy.org · SciPy — https://scipy.org · pandas — https://pandas.pydata.org
- scikit-learn — https://scikit-learn.org · XGBoost — https://xgboost.ai · matplotlib — https://matplotlib.org
- PyTorch — https://pytorch.org · ONNX / ONNX Runtime — https://onnx.ai, https://onnxruntime.ai
- FastAPI — https://fastapi.tiangolo.com · Uvicorn — https://www.uvicorn.org · Streamlit — https://streamlit.io
- llama.cpp — https://github.com/ggml-org/llama.cpp · IBM Granite — https://www.ibm.com/granite
- IBM LinuxONE Community Cloud Jupyter Lab ML image — https://github.com/linuxone-community-cloud/jupyter-lab-ml
