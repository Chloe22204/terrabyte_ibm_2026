"""
TimeToTalk scoring API - runs on LinuxONE inside a container.

  export TTT_API_TOKEN=<long random string>      # required
  export TTT_ARTIFACTS=/app/artifacts
  uvicorn app:app --host 0.0.0.0 --port 8000

Security measures (screenshot these for the rubric):
  - every request needs the bearer token (no token -> 401)
  - patient identifiers are SHA-256 hashed with a secret salt before logging; raw IDs never stored
  - audit log of every scoring request (who/when/decision), no clinical values written to logs
  - all inference (risk models + Granite) happens on this LinuxONE VM; no patient data leaves it
"""
import hashlib, json, logging, os, sys, time
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

for sub in ("ml", "genai"):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", sub))
    sys.path.insert(0, f"/app/{sub}")
from scoring import TimeToTalkScorer  # noqa: E402
from prep_sheet import LANGUAGES, ask_why, make_prep_sheet  # noqa: E402
import granite  # noqa: E402

TOKEN = os.environ.get("TTT_API_TOKEN")
SALT = os.environ.get("TTT_ID_SALT", "change-me")
ARTIFACTS = os.environ.get("TTT_ARTIFACTS", "artifacts")
if not TOKEN:
    raise SystemExit("Set TTT_API_TOKEN before starting the API")

logging.basicConfig(filename=os.environ.get("TTT_AUDIT_LOG", "audit.log"), level=logging.INFO,
                    format="%(asctime)s %(message)s")
scorer = TimeToTalkScorer(ARTIFACTS)
app = FastAPI(title="TimeToTalk API", version="1.0")


def auth(authorization: str = Header(default="")):
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="invalid or missing token")


def pseudonym(patient_id: str) -> str:
    return hashlib.sha256((SALT + str(patient_id)).encode()).hexdigest()[:16]


class ScoreRequest(BaseModel):
    patient_id: str
    clinician_id: str = "unknown"
    patient: dict           # SUPPORT2-style fields, e.g. {"age": 78, "sex": "female", "alb": 2.4, ...}


class PrepRequest(ScoreRequest):
    language: str = "English"


class WhyRequest(ScoreRequest):
    question: str


@app.get("/health")
def health():
    import platform
    return {"status": "ok", "machine": platform.machine(), "features": len(scorer.features),
            "bootstrap_models": len(scorer.boots), "second_opinion": scorer.xgb is not None,
            "granite": granite.health()}


@app.post("/score", dependencies=[Depends(auth)])
def score(req: ScoreRequest):
    t0 = time.perf_counter()
    result = scorer.score(req.patient)
    ms = (time.perf_counter() - t0) * 1000
    pid = pseudonym(req.patient_id)
    logging.info(json.dumps({"event": "score", "patient": pid, "clinician": req.clinician_id,
                             "decision": result["decision"], "latency_ms": round(ms, 2)}))
    return {"patient_ref": pid, "latency_ms": ms, **result}


@app.post("/prep_sheet", dependencies=[Depends(auth)])
def prep_sheet(req: PrepRequest):
    """Score + Granite conversation prep sheet. Only metadata is logged, never the generated text."""
    if req.language not in LANGUAGES:
        raise HTTPException(status_code=422, detail=f"language must be one of {list(LANGUAGES)}")
    result = scorer.score(req.patient)
    sheet = make_prep_sheet(result, req.patient, req.language)
    pid = pseudonym(req.patient_id)
    logging.info(json.dumps({"event": "prep_sheet", "patient": pid, "clinician": req.clinician_id,
                             "decision": result["decision"], "language": req.language,
                             "used_llm": sheet["used_llm"], "latency_s": sheet["latency_s"],
                             "n_warnings": len(sheet["warnings"])}))
    return {"patient_ref": pid, "score": result, **sheet}


@app.post("/ask_why", dependencies=[Depends(auth)])
def why(req: WhyRequest):
    if not req.question.strip() or len(req.question) > 500:
        raise HTTPException(status_code=422, detail="question must be 1-500 characters")
    result = scorer.score(req.patient)
    out = ask_why(result, req.question, req.patient)
    logging.info(json.dumps({"event": "ask_why", "patient": pseudonym(req.patient_id),
                             "clinician": req.clinician_id, "used_llm": out["used_llm"]}))
    return out


@app.post("/benchmark", dependencies=[Depends(auth)])
def benchmark(req: ScoreRequest, n: int = 1000):
    """Scores the same patient n times; use for the 'compute' screenshot."""
    t0 = time.perf_counter()
    for _ in range(n):
        scorer.score(req.patient)
    total = time.perf_counter() - t0
    return {"n": n, "total_s": total, "per_patient_ms": total / n * 1000, "patients_per_s": n / total}
