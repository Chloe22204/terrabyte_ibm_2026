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
          ├ Cox survival model + 20 bootstrap models (uncertainty)
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
podman build -t timetotalk-api -f api/Containerfile .
podman run -d --name ttt -p 8000:8000 -e TTT_API_TOKEN=$(openssl rand -hex 24) -e TTT_ID_SALT=$(openssl rand -hex 16) timetotalk-api
curl localhost:8000/health
```

## Decisions log (what we chose, what we rejected, and why)

| Decision | Why |
|---|---|
| **Cox proportional hazards** as the main model | Handles patients still alive at follow-up (censoring), gives a full survival curve, and every factor is explainable. A clinician can check it. |
| Wrote our own numpy/scipy Cox model instead of `lifelines` | `lifelines` needs `autograd-gamma`, which failed to build in testing. Ours has no compiled dependencies, so it runs on s390x. |
| **Rejected** deep learning as the main model | Little gain on about 9,000 patients and much harder to explain. Kept a small neural net only as a second opinion on the AI accelerator. |
| **Leaky columns excluded** (death, d.time, hospdead, prg2m/6m, dnr/dnrday, surv2m/6m, slos, costs, avtisst, sfdm2) | The test for each column: would a doctor know it on day 3? If not, it's out. |
| **Race, income and education are not model inputs** | Fairness. They are used only to audit performance across groups (`fairness.csv`). |
| Missing labs → SUPPORT investigators' "normal" values, others → training median, plus a "was missing" flag | A missing lab usually means the doctor didn't think it was needed. This is documented and reproducible. |
| 60/20/20 split, seed 42; thresholds and recalibration use **validation only** | The test set is touched once, at the end. |
| Recalibration only kept if it helps on a held-out half of validation | Avoids "fixing" calibration that was already fine. |
| **"Clinician judgement"** output when the uncertainty is wide or the two models disagree | The model should say when it doesn't know. |
| **Rejected** a "risk is rising fast" alert | SUPPORT2 has one snapshot per patient, so we can't validate trends. It's listed under next steps. |
| Granite runs locally, not a cloud LLM | Patient data never leaves LinuxONE. Granite only explains the model's decision and never changes the numbers. |

## Results
See `artifacts/metrics.json` and the `fig_*.png` charts. Includes C-index, AUC, calibration, doctors vs model, "days earlier" (an imperfect proxy) and the fairness audit.

## Limitations & next steps
- SUPPORT2 is 1989–94 US data. This is a proof of concept. Next step: retrain and validate on Singapore hospital records.
- Repeated measurements per patient → a real "risk is rising" trigger
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
