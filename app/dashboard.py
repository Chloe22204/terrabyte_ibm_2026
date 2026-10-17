"""
TimeToTalk clinician dashboard (FastAPI + server-rendered HTML, no JavaScript).

Run from the repo root:
  export TTT_ID_SALT=<long random string>          # required: keys the patient pseudonyms
  python app/manage_users.py add dr_tan             # create at least one user
  uvicorn app.dashboard:app --host 127.0.0.1 --port 8501

Scoring is local by default (same scoring.py the API uses). Set TTT_USE_API=1 with
TTT_API_URL and TTT_API_TOKEN to score through the TimeToTalk API on LinuxONE instead.
See app/README.md for the security controls.
"""
import csv, json, os, platform, re, secrets, sys, time, urllib.request
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT))

from security import AuditLog, LoginGuard, SessionStore, UserStore, pseudonym, same  # noqa: E402
from genai.prompt import friendly  # noqa: E402
from genai import prep_sheet  # noqa: E402   (retrieval + Granite + template fallback)
from genai import granite  # noqa: E402

# ---------------------------------------------------------------- configuration
ARTIFACTS = os.environ.get("TTT_ARTIFACTS", str(ROOT / "artifacts_logistic"))
DATA = os.environ.get("TTT_DATA", str(ROOT / "data" / "support2.csv"))
SALT = os.environ.get("TTT_ID_SALT")
if not SALT or len(SALT) < 16:
    raise SystemExit("Set TTT_ID_SALT (at least 16 random characters) before starting the dashboard")
COOKIE_SECURE = os.environ.get("TTT_COOKIE_SECURE", "1") != "0"
USE_API = os.environ.get("TTT_USE_API", "0") == "1"
API_URL = os.environ.get("TTT_API_URL", "").rstrip("/")
API_TOKEN = os.environ.get("TTT_API_TOKEN", "")
if USE_API and not (API_URL and API_TOKEN):
    raise SystemExit("TTT_USE_API=1 needs TTT_API_URL and TTT_API_TOKEN")
GAP = float(os.environ.get("TTT_GAP_THRESHOLD", "0.25"))     # doctor vs model 6-month survival gap
SESSION_COOKIE = "__Host-ttt_session" if COOKIE_SECURE else "ttt_session"
PRE_COOKIE = "__Host-ttt_pre" if COOKIE_SECURE else "ttt_pre"

users = UserStore(os.environ.get("TTT_USERS_FILE", str(HERE / "users.json")))
guard = LoginGuard()
sessions = SessionStore(idle_s=int(os.environ.get("TTT_IDLE_MIN", "15")) * 60)
audit = AuditLog(os.environ.get("TTT_DASH_AUDIT_LOG", str(HERE / "logs" / "dashboard_audit.log")))

DECISIONS = ["Conversation due now", "Clinician judgement", "Review in 2 weeks", "No action"]
DECISION_INFO = {
    "Conversation due now": ("due", "High 6-month risk and the model is confident. Plan the advance care "
                                    "planning conversation now."),
    "Clinician judgement": ("judge", "The model is uncertain (wide range or the two models disagree). "
                                     "Your own assessment should lead."),
    "Review in 2 weeks": ("review", "Moderate risk. Re-check in two weeks or sooner if the condition changes."),
    "No action": ("none", "Low 6-month risk on current information. No conversation prompted."),
}
LANGUAGE_KEYS = {"English": "English", "Chinese (中文)": "Chinese", "Malay (Melayu)": "Malay",
                 "Tamil (தமிழ்)": "Tamil"}      # display label -> genai/prep_sheet.py language
LANGUAGES = list(LANGUAGE_KEYS)
MAX_QUESTION = 500
BENCH_N = 500
PAGE_SIZE = 50

# ---------------------------------------------------------------- data (test-set patients only)
with open(f"{ARTIFACTS}/patients_scored.csv", newline="", encoding="utf-8") as f:
    SCORED = list(csv.DictReader(f))
RAW = {}
if os.path.exists(DATA):
    with open(DATA, newline="", encoding="utf-8") as f:
        RAW = {r["id"]: r for r in csv.DictReader(f)}
pre = json.load(open(f"{ARTIFACTS}/preprocess.json"))
INPUT_COLS = sorted(set(pre["fill_values"]) | {f.partition("_")[0] for f in pre["features"]
                                               if not f.endswith("_missing") and f not in pre["fill_values"]})
THRESHOLDS = json.load(open(f"{ARTIFACTS}/thresholds.json"))
scorer = None
if not USE_API:
    from scoring import TimeToTalkScorer
    scorer = TimeToTalkScorer(ARTIFACTS)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


PATIENTS = {}                       # pseudonym -> worklist row; raw IDs never leave this process
for r in SCORED:
    ref = pseudonym(SALT, r["id"])
    doctor = _num(RAW.get(r["id"], {}).get("prg6m"))
    risk = float(r["risk_6m"])
    gap = None if doctor is None else doctor - (1 - risk)
    PATIENTS[ref] = {"ref": ref, "id": r["id"], "age": int(float(r["age"])), "sex": r["sex"],
                     "dz": r["dzgroup"], "risk": risk, "lo": float(r["risk_6m_low"]),
                     "hi": float(r["risk_6m_high"]), "decision": r["decision"],
                     "doctor_surv": doctor, "gap": gap, "gap_flag": gap is not None and gap >= GAP}
DZ_GROUPS = sorted({p["dz"] for p in PATIENTS.values()})
ORDER = sorted(PATIENTS.values(), key=lambda p: (DECISIONS.index(p["decision"]), -p["risk"]))


# ---------------------------------------------------------------- app + security middleware
app = FastAPI(title="TimeToTalk dashboard", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")      # autoescaping on for .html

CSP = ("default-src 'self'; script-src 'none'; style-src 'self'; img-src 'self' data:; object-src 'none'; "
       "base-uri 'none'; form-action 'self'; frame-ancestors 'none'")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["Content-Security-Policy"] = CSP
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    if not request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-store"        # patient pages never cached by browser or proxy
    if request.url.scheme == "https":
        resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return resp


class Redirect(Exception):
    def __init__(self, to):
        self.to = to


@app.exception_handler(Redirect)
async def _redirect(request, exc):
    return RedirectResponse(exc.to, status_code=303)


@app.exception_handler(HTTPException)
async def _http_error(request, exc):
    msg = {403: "You don't have access to this page.", 404: "Not found."}.get(exc.status_code, "Request refused.")
    return templates.TemplateResponse(request, "error.html", {"message": msg, "s": None},
                                      status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def _bad_request(request, exc):
    # FastAPI's default 422 echoes the submitted values back; send a generic page instead
    return templates.TemplateResponse(request, "error.html", {"message": "Request refused.", "s": None},
                                      status_code=400)


@app.exception_handler(Exception)
async def _server_error(request, exc):
    audit.write("server_error", path=request.url.path, error=type(exc).__name__)   # no stack trace to users
    return templates.TemplateResponse(request, "error.html", {"message": "Something went wrong.", "s": None},
                                      status_code=500)


def client_ip(request):
    return request.client.host if request.client else "unknown"


def session(request, role="clinician"):
    s = sessions.get(request.cookies.get(SESSION_COOKIE))
    if not s:
        raise Redirect("/login")
    if role and s["role"] != role:
        audit.write("access_denied", user=s["user"], path=request.url.path)
        raise HTTPException(403)
    return s


def check_csrf(s, token):
    if not same(s["csrf"], token):
        audit.write("csrf_rejected", user=s["user"])
        raise HTTPException(403)


def set_cookie(resp, name, value, max_age=None):
    resp.set_cookie(name, value, max_age=max_age, path="/", httponly=True, secure=COOKIE_SECURE, samesite="strict")


# ---------------------------------------------------------------- scoring + presentation helpers
def model_inputs(raw):
    """Only the columns the model uses (data minimisation): outcomes, the doctor's estimate,
    race, income etc. are never sent to the scorer or API."""
    # the UCI download writes missing values as "NaN", other copies as "" or "NA"
    return {c: (None if str(raw.get(c, "")).strip().lower() in ("", "na", "nan") else raw.get(c)) for c in INPUT_COLS}


def api_post(path, payload, timeout=10):
    """POST to the TimeToTalk API with the bearer token (HTTPS certificates verified by default)."""
    req = urllib.request.Request(f"{API_URL}{path}", method="POST", data=json.dumps(payload).encode(),
                                 headers={"Authorization": f"Bearer {API_TOKEN}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def score(p, user):
    raw = RAW.get(p["id"])
    if raw is None:                                     # no raw record: fall back to the precomputed row
        return None
    inputs = model_inputs(raw)
    if not USE_API:
        return scorer.score(inputs)
    return api_post("/score", {"patient_id": p["ref"], "clinician_id": user, "patient": inputs})


# Granite on 3 vCPUs takes ~30-90 s per prep sheet; wait a little longer than genai/granite.py does
LLM_WAIT = float(os.environ.get("TTT_LLM_TIMEOUT", "180")) + 30


def clean_warnings(out):
    """Clinicians see plain messages; technical details (URLs, error codes) go to the audit log only."""
    for i, w in enumerate(out.get("warnings", [])):
        if w.startswith("Granite unavailable"):
            out["warnings"][i] = "Granite isn't running on the server right now, so this is the template version."
    return out


def make_sheet(p, user, language, result):
    return clean_warnings(_make_sheet(p, user, language, result))


def _make_sheet(p, user, language, result):
    """Conversation prep sheet from genai/prep_sheet.py: guideline retrieval + Granite, or its
    template fallback when Granite is down. Only model inputs are passed (summarised to an age band)."""
    inputs = model_inputs(RAW.get(p["id"], {}))
    if USE_API and RAW.get(p["id"]):
        try:
            return api_post("/prep_sheet", {"patient_id": p["ref"], "clinician_id": user, "patient": inputs,
                                            "language": language}, timeout=LLM_WAIT)
        except OSError:
            audit.write("api_unreachable", user=user, patient=p["ref"])
            sheet = prep_sheet.make_prep_sheet(result, inputs, language, use_llm=False)
            sheet["warnings"].insert(0, "TimeToTalk API unreachable: template prep sheet from this dashboard.")
            return sheet
    return prep_sheet.make_prep_sheet(result, inputs, language)


def ask(p, user, question, result):
    inputs = model_inputs(RAW.get(p["id"], {}))
    if USE_API and RAW.get(p["id"]):
        try:
            return api_post("/ask_why", {"patient_id": p["ref"], "clinician_id": user, "patient": inputs,
                                         "question": question}, timeout=LLM_WAIT)
        except OSError:
            audit.write("api_unreachable", user=user, patient=p["ref"])
            return prep_sheet.ask_why(result, question, inputs, use_llm=False)
    return prep_sheet.ask_why(result, question, inputs)


def fallback_result(p):
    row = next(r for r in SCORED if r["id"] == p["id"])
    days = [0, 30, 60, 90, 180, 365]
    factors = []
    for part in row["top_factors"].split("; "):
        if part.endswith(")"):                          # newer format: "Albumin 2.1 g/dL (raises risk)"
            label, _, effect = part[:-1].rpartition(" (")
            factors.append({"feature": label, "label": label, "effect": effect,
                            "log_odds_contribution": 1.0 if effect == "raises risk" else -1.0})
        else:                                           # older format: "alb +"
            name, _, sign = part.rpartition(" ")
            factors.append({"feature": name, "effect": "raises risk" if sign == "+" else "lowers risk",
                            "log_odds_contribution": 1.0 if sign == "+" else -1.0})
    return {"risk_6m": p["risk"], "risk_6m_interval_80": [p["lo"], p["hi"]],
            "survival_2m": float(row["surv_2m"]), "survival_6m": 1 - p["risk"],
            "survival_curve": {"days": days, "survival": [1.0] + [float(row[f"S_{d}d"]) for d in days[1:]]},
            "second_opinion_risk_6m": _num(row["second_opinion_6m"]), "models_disagree": False,
            "decision": p["decision"], "top_factors": factors, "thresholds": THRESHOLDS}


def curve_svg(days, surv, w=560, h=240, pad=40):
    """Survival curve as SVG coordinates (drawn server-side: no JavaScript, CSP-friendly)."""
    x = lambda d: pad + (w - 2 * pad) * d / 365
    y = lambda s: h - pad + (2 * pad - h) * s
    pts = " ".join(f"{x(d):.1f},{y(s):.1f}" for d, s in zip(days, surv))
    s180 = surv[days.index(180)] if 180 in days else None
    return {"w": w, "h": h, "pad": pad, "points": pts,
            "xticks": [(x(d), d) for d in (0, 90, 180, 270, 365)],
            "yticks": [(y(v), f"{v:.0%}") for v in (0, 0.25, 0.5, 0.75, 1)],
            "mark": (x(180), y(s180), f"{s180:.0%}") if s180 is not None else None,
            "x0": pad, "x1": w - pad, "y0": h - pad, "y1": pad}


def factor_bars(factors, w=220):
    m = max([abs(f["log_odds_contribution"]) for f in factors] + [1e-9])
    # live results carry a plain-English "label" from ml/explain.py; precomputed fallback rows don't
    return [{"name": f.get("label") or friendly(f["feature"]), "up": f["effect"] == "raises risk",
             "width": max(4, w * abs(f["log_odds_contribution"]) / m)} for f in factors]


def doctor_choice(field):
    """The doctor's-estimate form field: "recorded" (the estimate on file), "none" (cleared),
    or the clinician's own 6-month survival estimate as a whole percentage 0-100."""
    field = (field or "recorded").strip()
    if field in ("recorded", "none"):
        return field
    if field.isdigit() and 0 <= int(field) <= 100:
        return str(int(field))
    raise HTTPException(400)


def doctor_view(p, risk, field):
    """Doctor's estimate vs model, and the two combined. Combined = the same 50/50 average of the
    two 6-month risks as the API's blend_with_doctor (AUC 0.831 on held-out patients, ml/combo.py)."""
    d = p["doctor_surv"] if field == "recorded" else (None if field == "none" else int(field) / 100)
    if d is None:
        return {"field": field, "surv": None, "source": field}
    doctor_risk = 1 - d
    combined = (risk + doctor_risk) / 2
    gap = d - (1 - risk)
    bar = lambda v: max(2, 300 * v)
    return {"field": field, "source": field if field == "recorded" else "yours", "surv": d,
            "risk": doctor_risk, "combined": combined, "gap": gap, "gap_flag": gap >= GAP,
            "bars": {"doctor": bar(doctor_risk), "model": bar(risk), "combined": bar(combined)}}


_status = {"t": 0.0, "v": None}


def system_status():
    """Where inference runs, for the worklist's System panel. Cached for 30 s (Granite check is a network call)."""
    if _status["v"] is not None and time.monotonic() - _status["t"] < 30:
        return _status["v"]
    if USE_API:
        try:
            with urllib.request.urlopen(f"{API_URL}/health", timeout=5) as r:
                h = json.load(r)
            v = {"machine": h.get("machine"), "bootstraps": h.get("bootstrap_models"),
                 "second_opinion": bool(h.get("second_opinion")),
                 "granite": bool((h.get("granite") or {}).get("ok")), "where": "TimeToTalk API"}
        except (OSError, ValueError):
            v = {"error": "TimeToTalk API unreachable"}
    else:
        v = {"machine": platform.machine(), "bootstraps": len(scorer.boots), "second_opinion": scorer.xgb is not None,
             "granite": bool(granite.health().get("ok")), "where": "this dashboard"}
    v["linuxone"] = v.get("machine") == "s390x"
    _status.update(t=time.monotonic(), v=v)
    return v


def benchmark():
    """Score one test patient BENCH_N times (model speed only, no HTTP), locally or on the API."""
    p = next((q for q in ORDER if q["id"] in RAW), None)
    if p is None:
        return {"error": "Benchmark needs data/support2.csv for live scoring."}
    inputs = model_inputs(RAW[p["id"]])
    if USE_API:
        try:
            r = api_post(f"/benchmark?n={BENCH_N}", {"patient_id": p["ref"], "patient": inputs}, timeout=120)
        except OSError:
            return {"error": "TimeToTalk API unreachable."}
        return {"n": r["n"], "per_s": r["patients_per_s"], "ms": r["per_patient_ms"], "where": "TimeToTalk API"}
    t0 = time.perf_counter()
    for _ in range(BENCH_N):
        scorer.score(inputs)
    total = time.perf_counter() - t0
    return {"n": BENCH_N, "per_s": BENCH_N / total, "ms": total / BENCH_N * 1000, "where": "this dashboard"}


def current_result(p, user):
    """Live score (local or API); the precomputed row if that's impossible. Returns (result, live)."""
    try:
        result = score(p, user)
    except OSError:
        audit.write("api_unreachable", user=user, patient=p["ref"])
        result = None
    return (result, True) if result is not None else (fallback_result(p), False)


def patient_view(request, s, p, briefing=None, language="English", why=None, question="", doctor="recorded"):
    t0 = time.perf_counter()
    result, live = current_result(p, s["user"])
    score_ms = (time.perf_counter() - t0) * 1000
    lo, hi = result["risk_6m_interval_80"]
    risk = result["risk_6m"]
    ctx = {"s": s, "p": p, "r": result, "live": live, "mode": "LinuxONE API" if USE_API else "local scorer",
           "info": DECISION_INFO[result["decision"]],
           "curve": curve_svg(result["survival_curve"]["days"], result["survival_curve"]["survival"]),
           "bars": factor_bars(result["top_factors"]),
           "range": {"lo": 20 + 360 * lo, "hi": 20 + 360 * hi, "risk": 20 + 360 * risk,
                     "t_mid": 20 + 360 * THRESHOLDS["t_mid"], "t_high": 20 + 360 * THRESHOLDS["t_high"]},
           "languages": LANGUAGES, "language": language, "briefing": briefing,
           "why": why, "question": question, "max_question": MAX_QUESTION,
           "doc": doctor_view(p, risk, doctor), "score_ms": score_ms}
    return templates.TemplateResponse(request, "patient.html", ctx)


# ---------------------------------------------------------------- routes
@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    if sessions.get(request.cookies.get(SESSION_COOKIE)):
        return RedirectResponse("/", status_code=303)
    token = secrets.token_urlsafe(32)
    resp = templates.TemplateResponse(request, "login.html", {"csrf": token, "error": None, "s": None})
    set_cookie(resp, PRE_COOKIE, token, max_age=600)       # double-submit token: blocks login CSRF
    return resp


@app.post("/login", response_class=HTMLResponse)
def login(request: Request, username: str = Form(..., max_length=64), password: str = Form(..., max_length=256),
          csrf: str = Form(...)):
    ip = client_ip(request)
    if not same(request.cookies.get(PRE_COOKIE, ""), csrf):
        return RedirectResponse("/login", status_code=303)
    error = None
    if guard.is_locked(username, ip):
        audit.write("login_locked", user=username, ip=ip)
        error = "Too many failed attempts. Try again in 15 minutes."
    else:
        role = users.authenticate(username, password)
        if role:
            guard.succeeded(username)
            token = sessions.create(username, role)          # fresh token on every login (no fixation)
            audit.write("login", user=username, role=role, ip=ip)
            resp = RedirectResponse("/" if role == "clinician" else "/audit", status_code=303)
            set_cookie(resp, SESSION_COOKIE, token)
            resp.delete_cookie(PRE_COOKIE, path="/")
            return resp
        guard.failed(username, ip)
        audit.write("login_failed", user=username, ip=ip)
        error = "Wrong username or password."               # never says which one
    resp = templates.TemplateResponse(request, "login.html", {"csrf": csrf, "error": error, "s": None},
                                      status_code=401)
    return resp


@app.post("/logout")
def logout(request: Request, csrf: str = Form(...)):
    s = session(request, role=None)
    check_csrf(s, csrf)
    sessions.end(request.cookies.get(SESSION_COOKIE))
    audit.write("logout", user=s["user"])
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


@app.get("/", response_class=HTMLResponse)
def worklist(request: Request, decision: str = "", dz: str = "", gap: str = "", page: int = 1):
    s = session(request)
    return worklist_view(request, s, decision, dz, gap, page)


def worklist_view(request, s, decision="", dz="", gap="", page=1, bench=None):
    decision = decision if decision in DECISIONS else ""        # whitelist every filter value
    dz = dz if dz in DZ_GROUPS else ""
    rows = [p for p in ORDER if (not decision or p["decision"] == decision) and (not dz or p["dz"] == dz)
            and (gap != "1" or p["gap_flag"])]
    pages = max(1, -(-len(rows) // PAGE_SIZE))
    page = min(max(page, 1), pages)
    counts = {d: sum(p["decision"] == d for p in PATIENTS.values()) for d in DECISIONS}
    audit.write("view_worklist", user=s["user"], filter_decision=decision, filter_dz=dz, gap_only=gap == "1")
    return templates.TemplateResponse(request, "worklist.html", {
        "s": s, "rows": rows[(page - 1) * PAGE_SIZE: page * PAGE_SIZE], "n": len(rows), "page": page,
        "pages": pages, "decision": decision, "dz": dz, "gap": gap == "1", "decisions": DECISIONS,
        "dz_groups": DZ_GROUPS, "counts": counts, "n_gap": sum(p["gap_flag"] for p in PATIENTS.values()),
        "info": DECISION_INFO, "gap_threshold": GAP, "mode": "LinuxONE API" if USE_API else "local scorer",
        "system": system_status(), "bench": bench, "bench_n": BENCH_N})


@app.post("/benchmark", response_class=HTMLResponse)
def run_benchmark(request: Request, csrf: str = Form(...)):
    s = session(request)
    check_csrf(s, csrf)
    bench = benchmark()
    audit.write("benchmark", user=s["user"], n=bench.get("n"), patients_per_s=round(bench.get("per_s", 0)))
    return worklist_view(request, s, bench=bench)


@app.get("/patient/{ref}", response_class=HTMLResponse)
def patient(request: Request, ref: str):
    s = session(request)
    p = valid_patient(ref)
    audit.write("view_patient", user=s["user"], patient=ref)
    return patient_view(request, s, p)


def valid_patient(ref):
    if not re.fullmatch(r"[0-9a-f]{12}", ref) or ref not in PATIENTS:
        raise HTTPException(404)
    return PATIENTS[ref]


@app.post("/patient/{ref}/doctor", response_class=HTMLResponse)
def doctor_estimate(request: Request, ref: str, csrf: str = Form(...), doctor: str = Form(..., max_length=10)):
    """What-if: the clinician enters their own 6-month survival estimate to see it next to the model."""
    s = session(request)
    check_csrf(s, csrf)
    p = valid_patient(ref)
    doctor = doctor_choice(doctor)
    audit.write("doctor_estimate", user=s["user"], patient=ref,       # never the estimate itself
                source="yours" if doctor.isdigit() else doctor)
    return patient_view(request, s, p, doctor=doctor)


@app.post("/patient/{ref}/briefing", response_class=HTMLResponse)
def briefing(request: Request, ref: str, csrf: str = Form(...), language: str = Form("English"),
             doctor: str = Form("recorded", max_length=10)):
    s = session(request)
    check_csrf(s, csrf)
    p = valid_patient(ref)
    doctor = doctor_choice(doctor)
    if language not in LANGUAGES:
        raise HTTPException(404)
    result, _ = current_result(p, s["user"])
    sheet = make_sheet(p, s["user"], LANGUAGE_KEYS[language], result)
    audit.write("briefing_requested", user=s["user"], patient=ref, language=language,   # never the text
                used_llm=sheet["used_llm"], warnings=len(sheet["warnings"]), latency_s=sheet["latency_s"])
    return patient_view(request, s, p, briefing=sheet, language=language, doctor=doctor)


@app.post("/patient/{ref}/why", response_class=HTMLResponse)
def why(request: Request, ref: str, csrf: str = Form(...), question: str = Form(..., max_length=MAX_QUESTION),
        doctor: str = Form("recorded", max_length=10)):
    s = session(request)
    check_csrf(s, csrf)
    p = valid_patient(ref)
    doctor = doctor_choice(doctor)
    question = question.strip()
    if not question:
        raise HTTPException(404)
    result, _ = current_result(p, s["user"])
    out = ask(p, s["user"], question, result)
    audit.write("ask_why", user=s["user"], patient=ref, used_llm=out["used_llm"])   # never the question text
    return patient_view(request, s, p, why=out, question=question, doctor=doctor)


@app.get("/audit", response_class=HTMLResponse)
def audit_page(request: Request):
    s = session(request, role="auditor")
    audit.write("view_audit_log", user=s["user"])
    return templates.TemplateResponse(request, "audit.html", {"s": s, "entries": audit.tail(300)})
