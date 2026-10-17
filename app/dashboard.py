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
import csv, json, os, re, secrets, sys, urllib.request
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
from genai.prompt import build_prompt, friendly  # noqa: E402

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
LANGUAGES = ["English", "Chinese (中文)", "Malay (Melayu)", "Tamil (தமிழ்)"]
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
    return {c: (raw.get(c) if raw.get(c) not in ("", "NA", "nan") else None) for c in INPUT_COLS}


def score(p, user):
    raw = RAW.get(p["id"])
    if raw is None:                                     # no raw record: fall back to the precomputed row
        return None
    inputs = model_inputs(raw)
    if not USE_API:
        return scorer.score(inputs)
    req = urllib.request.Request(f"{API_URL}/score", method="POST",
                                 data=json.dumps({"patient_id": p["ref"], "clinician_id": user,
                                                  "patient": inputs}).encode(),
                                 headers={"Authorization": f"Bearer {API_TOKEN}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:   # HTTPS certificates verified by default
        return json.load(r)


def fallback_result(p):
    row = next(r for r in SCORED if r["id"] == p["id"])
    days = [0, 30, 60, 90, 180, 365]
    factors = []
    for part in row["top_factors"].split("; "):
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


def patient_view(request, s, p, briefing=None, language="English"):
    result = None
    try:
        result = score(p, s["user"])
    except OSError:
        audit.write("api_unreachable", user=s["user"], patient=p["ref"])
    live = result is not None
    result = result or fallback_result(p)
    lo, hi = result["risk_6m_interval_80"]
    risk = result["risk_6m"]
    ctx = {"s": s, "p": p, "r": result, "live": live, "mode": "LinuxONE API" if USE_API else "local scorer",
           "info": DECISION_INFO[result["decision"]],
           "curve": curve_svg(result["survival_curve"]["days"], result["survival_curve"]["survival"]),
           "bars": factor_bars(result["top_factors"]),
           "range": {"lo": 20 + 360 * lo, "hi": 20 + 360 * hi, "risk": 20 + 360 * risk,
                     "t_mid": 20 + 360 * THRESHOLDS["t_mid"], "t_high": 20 + 360 * THRESHOLDS["t_high"]},
           "languages": LANGUAGES, "language": language, "briefing": briefing}
    return templates.TemplateResponse(request, "patient.html", ctx)


def summary_text(p):
    raw = RAW.get(p["id"], {})
    parts = [f"{p['age']}-year-old {p['sex']}", f"diagnosis group: {p['dz']}"]
    if raw.get("num.co"):
        parts.append(f"{raw['num.co']} other illnesses")
    if raw.get("ca"):
        parts.append(f"cancer status: {raw['ca']}")
    return "; ".join(parts)


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
        "info": DECISION_INFO, "gap_threshold": GAP, "mode": "LinuxONE API" if USE_API else "local scorer"})


@app.get("/patient/{ref}", response_class=HTMLResponse)
def patient(request: Request, ref: str):
    s = session(request)
    if not re.fullmatch(r"[0-9a-f]{12}", ref) or ref not in PATIENTS:
        raise HTTPException(404)
    audit.write("view_patient", user=s["user"], patient=ref)
    return patient_view(request, s, PATIENTS[ref])


@app.post("/patient/{ref}/briefing", response_class=HTMLResponse)
def briefing(request: Request, ref: str, csrf: str = Form(...), language: str = Form("English")):
    s = session(request)
    check_csrf(s, csrf)
    if not re.fullmatch(r"[0-9a-f]{12}", ref) or ref not in PATIENTS or language not in LANGUAGES:
        raise HTTPException(404)
    p = PATIENTS[ref]
    result = None
    try:
        result = score(p, s["user"])
    except OSError:
        pass
    result = result or fallback_result(p)
    messages = build_prompt(result, summary_text(p), [], language.split(" (")[0])
    text = None
    try:                                     # hook for the Granite module (genai/briefing.py, when ready)
        from genai.briefing import generate_briefing
        text = generate_briefing(messages)
    except ImportError:
        pass
    audit.write("briefing_requested", user=s["user"], patient=ref, language=language, generated=text is not None)
    return patient_view(request, s, p, briefing={"text": text, "prompt": messages}, language=language)


@app.get("/audit", response_class=HTMLResponse)
def audit_page(request: Request):
    s = session(request, role="auditor")
    audit.write("view_audit_log", user=s["user"])
    return templates.TemplateResponse(request, "audit.html", {"s": s, "entries": audit.tail(300)})
