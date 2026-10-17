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
                     │  → conversation prep sheet (EN / 中文 / Melayu / தமிழ், tested languages only) + "ask why"
                     ▼
         app/ Streamlit clinician dashboard
```

## Run it

Full step-by-step guide, with checks and troubleshooting: **[docs/SETUP.md](docs/SETUP.md)**. Quick version, on the LinuxONE VM:

```bash
python3 ml/train.py --data data/support2.csv --out artifacts --bootstraps 20      # 1. train (data from UCI, see guide)
llama-server -m ~/models/granite-be.gguf -t 3 -c 2048 --host 127.0.0.1 --port 8080 &  # 2. Granite (build steps in guide)
python3 genai/prep_sheet.py --language English                                    # 3. prep sheet from the command line
podman build -t timetotalk-api -f api/Containerfile .                              # 4. API in a container
podman run -d --name ttt --network host -e TTT_API_TOKEN=$(openssl rand -hex 24) -e TTT_ID_SALT=$(openssl rand -hex 16) timetotalk-api
curl localhost:8000/health
```

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

Care-planning guidance used for retrieval (`genai/guidelines/`, summarised in our own words, not copied):
- Serious Illness Conversation Guide, Ariadne Labs — https://www.ariadnelabs.org/serious-illness-care/
- VitalTalk — https://www.vitaltalk.org · SPICT — https://www.spict.org.uk
- SPIKES protocol, Baile et al., The Oncologist 2000 — https://doi.org/10.1634/theoncologist.5-4-302
- NICE NG142 End of life care for adults — https://www.nice.org.uk/guidance/ng142
- Living Matters, Agency for Integrated Care Singapore — https://www.livingmatters.sg
