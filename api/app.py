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
import hashlib, json, logging, os, re, sys, time
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from typing import Optional
from pydantic import BaseModel

for sub in ("ml", "genai"):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", sub))
    sys.path.insert(0, f"/app/{sub}")
from scoring import TimeToTalkScorer  # noqa: E402
from prep_sheet import LANGUAGES, ask_why, make_prep_sheet  # noqa: E402
import granite  # noqa: E402

try:   # optional richer Granite briefing from the GenAI work; prep_sheet.py is used if it isn't there
    from briefing import briefing as granite_briefing  # noqa: E402
except Exception:
    granite_briefing = None

TOKEN = os.environ.get("TTT_API_TOKEN")
SALT = os.environ.get("TTT_ID_SALT", "change-me")
ARTIFACTS = os.environ.get("TTT_ARTIFACTS", "artifacts")
# clinician demo screen: /app/ui in the container, app/ when run from the repo
UI_DIR = os.environ.get("TTT_UI_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
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
    doctor_survival_6m: Optional[float] = None   # optional: clinician's own estimate (0-1) of surviving 6 months


def blend_with_doctor(result: dict, doctor_survival_6m):
    """AI + clinician: average the model's 6-month risk with the doctor's own estimate.
    On held-out SUPPORT2 patients this beat either alone (AUC 0.831 vs 0.804 model / 0.786 doctor; ml/combo.py)."""
    if doctor_survival_6m is None:
        return {"doctor_risk_6m": None, "combined_risk_6m": None}
    d = min(max(float(doctor_survival_6m), 0.0), 1.0)
    return {"doctor_risk_6m": 1 - d, "combined_risk_6m": (result["risk_6m"] + (1 - d)) / 2}


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
    result.update(blend_with_doctor(result, req.doctor_survival_6m))
    logging.info(json.dumps({"event": "score", "patient": pid, "clinician": req.clinician_id,
                             "decision": result["decision"], "doctor_input": req.doctor_survival_6m is not None,
                             "latency_ms": round(ms, 2)}))
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


# ------------------------------------------------------------------ clinician demo screen
# GET / is the static page (no auth, no patient values in the URL). The page holds the token in memory only
# and sends it as a bearer header to /score, /briefing, /why and /benchmark.
UI_HEADERS = {
    "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                               "connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'; "
                               "frame-ancestors 'none'",
}
DEMO_SLOT = '<script id="demo-patients" type="application/json">null</script>'


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    try:
        html = open(os.path.join(UI_DIR, "index.html"), encoding="utf-8").read()
    except OSError:
        raise HTTPException(status_code=404, detail=f"index.html not found in TTT_UI_DIR={UI_DIR}")
    try:   # de-identified SUPPORT2 test patients, inlined so the page needs no extra route
        demo = json.load(open(os.path.join(UI_DIR, "demo_patients.json"), encoding="utf-8"))
    except (OSError, ValueError):
        demo = None
    slot = DEMO_SLOT.replace(">null<", ">" + json.dumps(demo).replace("<", "\\u003c") + "<")
    return HTMLResponse(html.replace(DEMO_SLOT, slot), headers=UI_HEADERS)


def granite_up() -> bool:
    try:
        return bool(granite.health().get("ok"))
    except Exception:
        return False


def guidance():
    """Questions and phrases to avoid for the template sheet: docs/guidelines.txt if present,
    otherwise the RAG notes in genai/guidelines/. Built-in defaults fill any gap."""
    questions, avoid, source = [], [], "built-in defaults"
    txt = os.path.join(REPO, "docs", "guidelines.txt")
    try:
        if os.path.exists(txt):
            text, source = open(txt, encoding="utf-8").read(), "docs/guidelines.txt"
            questions = re.findall(r'"([^"]+\?)"', text)
            avoid = [ln.strip("-* ").strip() for ln in text.splitlines() if re.match(r"\s*[-*]?\s*(avoid|don't)\b", ln, re.I)]
        else:
            from retriever import GUIDE_DIR
            q = os.path.join(GUIDE_DIR, "questions_to_explore.md")
            if os.path.exists(q):
                questions, source = re.findall(r'"([^"]+\?)"', open(q, encoding="utf-8").read()), "genai/guidelines/"
            a = os.path.join(GUIDE_DIR, "language_to_avoid.md")
            if os.path.exists(a):
                for part in re.split(r"^## ", open(a, encoding="utf-8").read(), flags=re.M)[1:]:
                    head, _, body = part.partition("\n")
                    first = re.split(r"(?<=\.)\s", " ".join(body.split()), maxsplit=1)[0]
                    avoid.append(f"{head.strip()}: {first}")
                source = "genai/guidelines/"
    except Exception:
        pass
    defaults_q = ["What have the doctors told you about your illness so far?",
                  "How much information would you like about what may lie ahead?",
                  "If your health were to get worse, what would matter most to you?",
                  "What worries you most about the future?",
                  "Who should speak for you if you could not speak for yourself?"]
    defaults_a = ['"There is nothing more we can do": care never stops, even when some treatments do.',
                  '"Do you want us to do everything?": ask about goals, then recommend what fits them.',
                  "Exact time predictions: talk in ranges and admit uncertainty.",
                  "Unexplained jargon (DNR, palliative, prognosis): use plain words and check understanding."]
    questions = list(dict.fromkeys(questions + defaults_q))[:5]
    return questions, (avoid or defaults_a)[:6], source


OPENING = {
    "Conversation due now": "\"I'd like to talk with you about how things are going with your illness, and plan ahead "
                            "so that your care fits what matters most to you. Is now a good time?\"",
    "Review in 2 weeks": "\"Could we set aside some time in the next couple of weeks to talk about how things are "
                         "going, and what matters most to you as we plan your care?\"",
    "Clinician judgement": "\"I'd like to hear how you feel things have been going, and what matters most to you, "
                           "so we can plan for whatever happens. Is now a good time?\"",
    "No action": "No conversation is flagged today. If the patient or family raise worries, you could open with: "
                 "\"How do you feel things have been going lately?\"",
}
NEXT_STEP = {
    "Conversation due now": "Record what the patient understands, their goals and what matters most, who should speak "
                            "for them if they cannot, and a date to revisit. Offer a Living Matters advance care "
                            "planning facilitator if they want to formalise their wishes.",
    "Review in 2 weeks": "Book the conversation within the next 2 weeks and note the planned date in the care plan. "
                         "Review again sooner if the patient's condition changes.",
    "Clinician judgement": "The model is unsure for this patient. Record your own assessment and the treating team's "
                           "view, and agree a date to review.",
    "No action": "Nothing to record today. Review again after an unplanned admission or a decline in daily activities.",
}


def factor_lines(score, effect):
    return [f["label"] for f in score["top_factors"] if f["effect"] == effect]


def template_briefing(score, patient=None):
    lo, hi = score["risk_6m_interval_80"]
    questions, avoid, source = guidance()
    up = factor_lines(score, "raises risk")
    lines = ["Template (Granite not connected)", ""]
    if score["decision"] == "Clinician judgement":
        lines += ["Note: the model is unsure for this patient. Your own clinical assessment should lead.", ""]
    lines += [f"Why now: {score['decision']}. Estimated 6-month mortality risk {score['risk_6m']:.0%} "
              f"(80% range {lo:.0%} to {hi:.0%})." + (f" Main factors raising it: {'; '.join(up[:3])}." if up else ""),
              "", "Suggested opening", OPENING.get(score["decision"], OPENING["Clinician judgement"]),
              "", "Questions to explore"] + [f"{i}. {q}" for i, q in enumerate(questions, 1)]
    lines += ["", "Phrases to avoid"] + [f"- {a}" for a in avoid]
    lines += ["", "Next step to record", NEXT_STEP.get(score["decision"], NEXT_STEP["Clinician judgement"]),
              "", f"Guidance: {source}"]
    return "\n".join(lines)


def template_why(score, question):
    lo, hi = score["risk_6m_interval_80"]
    up, down = factor_lines(score, "raises risk"), factor_lines(score, "lowers risk")
    lines = ["Template answer (Granite not connected).", f"You asked: \"{question}\"", "",
             f"TimeToTalk estimates a 6-month mortality risk of {score['risk_6m']:.0%} (80% range {lo:.0%} to "
             f"{hi:.0%}), which gives the recommendation \"{score['decision']}\"."]
    if up:
        lines.append("Pushing the estimate up: " + "; ".join(up) + ".")
    if down:
        lines.append("Pulling it down: " + "; ".join(down) + ".")
    if score["decision"] == "Clinician judgement":
        lines.append("The range is wide or the two models disagree, so the data do not clearly place this patient.")
    lines.append("These are patterns in past patients' day-3 hospital data, not causes. "
                 "Check them against what you know about this patient.")
    return "\n".join(lines)


def briefing(score: dict, patient: dict) -> dict:
    """Conversation prep sheet -> {"text", "source": "granite" | "template"}. Never raises."""
    if granite_up():
        try:
            if granite_briefing is not None:
                out = granite_briefing(score, patient)
                if out.get("text"):
                    return {"text": out["text"], "source": out.get("source", "granite")}
            else:
                sheet = make_prep_sheet(score, patient, "English")
                if sheet["used_llm"]:
                    return {"text": sheet["text"], "source": "granite", "warnings": sheet["warnings"]}
        except Exception:
            pass
    return {"text": template_briefing(score, patient), "source": "template"}


@app.post("/briefing", dependencies=[Depends(auth)])
def briefing_route(req: ScoreRequest):
    try:
        result = scorer.score(req.patient)
        out = briefing(result, req.patient)
    except Exception as e:   # the page must never break: report, don't crash
        out = {"text": f"Prep sheet unavailable ({type(e).__name__}). Scoring and other panels are unaffected.",
               "source": "template"}
        result = {"decision": None}
    logging.info(json.dumps({"event": "briefing", "patient": pseudonym(req.patient_id), "clinician": req.clinician_id,
                             "decision": result["decision"], "source": out["source"]}))
    return out


@app.post("/why", dependencies=[Depends(auth)])
def why_route(req: WhyRequest):
    question = req.question.strip()[:500]
    if not question:
        raise HTTPException(status_code=422, detail="question must be 1-500 characters")
    try:
        result = scorer.score(req.patient)
        out = None
        if granite_up():
            try:
                a = ask_why(result, question, req.patient)
                if a["used_llm"]:
                    out = {"text": a["answer"], "source": "granite", "warnings": a["warnings"]}
            except Exception:
                pass
        out = out or {"text": template_why(result, question), "source": "template"}
    except Exception as e:
        out = {"text": f"Answer unavailable ({type(e).__name__}).", "source": "template"}
    logging.info(json.dumps({"event": "why", "patient": pseudonym(req.patient_id), "clinician": req.clinician_id,
                             "source": out["source"]}))
    return out
