"""The dashboard over HTTP: login, sessions, roles, headers, and hostile input."""
import json, re

import pytest

from app import dashboard
from conftest import PASSWORDS, csrf_of, login, needs_raw

COOKIE = dashboard.SESSION_COOKIE


# ---------------------------------------------------------------- login & sessions
@pytest.mark.parametrize("path", ["/", "/audit", "/patient/000000000000", "/?decision=No%20action"])
def test_signed_out_users_are_sent_to_login(client, path):
    r = client.get(path, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"


def test_successful_login_sets_hardened_cookie(client):
    r = login(client, "clin")
    assert r.status_code == 303 and r.headers["location"] == "/"
    cookie = [h for h in r.headers.get_list("set-cookie") if h.startswith(COOKIE)][0].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "path=/" in cookie


def test_auditor_lands_on_audit_log(client):
    assert login(client, "audit").headers["location"] == "/audit"


@pytest.mark.parametrize("user,password", [("clin", "wrong-password"), ("nobody", "whatever-123"),
                                           ("clin", ""), ("CLIN", None), ("gone", None)])
def test_failed_logins_look_identical(client, user, password):
    if password is None:
        password = PASSWORDS.get(user.lower(), "x")
    r = login(client, user, password)
    assert r.status_code == 401 or r.status_code == 400
    if r.status_code == 401:
        assert "Wrong username or password." in r.text
    assert COOKIE not in r.headers.get("set-cookie", "")


def test_login_without_matching_csrf_is_ignored(client):
    client.get("/login")
    r = client.post("/login", data={"username": "clin", "password": PASSWORDS["clin"], "csrf": "forged"},
                    follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert client.get("/", follow_redirects=False).status_code == 303          # still signed out


def test_lockout_after_five_failures_blocks_even_the_right_password(client):
    for i in range(5):
        login(client, "clin", f"wrong-{i}")
    r = login(client, "clin")
    assert "Too many failed attempts" in r.text and r.status_code == 401


def test_logout_kills_the_session_server_side(clinician):
    old = clinician.cookies.get(COOKIE)
    page = clinician.get("/")
    r = clinician.post("/logout", data={"csrf": csrf_of(page.text)}, follow_redirects=False)
    assert r.status_code == 303
    clinician.cookies.set(COOKIE, old)                  # replay the old cookie
    assert clinician.get("/", follow_redirects=False).status_code == 303


def test_logout_requires_csrf(clinician):
    assert clinician.post("/logout", data={"csrf": "forged"}).status_code == 403
    assert clinician.get("/").status_code == 200       # still signed in


def test_forged_session_cookie_is_rejected(client):
    client.cookies.set(COOKIE, "A" * 43)
    assert client.get("/", follow_redirects=False).status_code == 303


def test_new_token_on_every_login(client):
    login(client, "clin")
    first = client.cookies.get(COOKIE)
    client.cookies.clear()
    login(client, "clin")
    assert client.cookies.get(COOKIE) != first


def test_idle_timeout_signs_you_out(clinician, monkeypatch):
    monkeypatch.setattr(dashboard.sessions, "idle", -1)
    assert clinician.get("/", follow_redirects=False).status_code == 303


# ---------------------------------------------------------------- roles
def test_clinician_cannot_open_audit_log(clinician):
    r = clinician.get("/audit")
    assert r.status_code == 403 and "<h1>Audit log</h1>" not in r.text


def test_auditor_cannot_open_patients(auditor, some_ref):
    assert auditor.get(f"/patient/{some_ref}").status_code == 403
    assert auditor.get("/").status_code == 403


def test_denials_are_audited(clinician):
    clinician.get("/audit")
    events = [e["event"] for e in dashboard.audit.tail(5)]
    assert "access_denied" in events


# ---------------------------------------------------------------- security headers
@pytest.mark.parametrize("path", ["/login", "/healthz", "/static/style.css", "/nope"])
def test_security_headers_everywhere(client, path):
    h = client.get(path).headers
    assert "script-src 'none'" in h["content-security-policy"]
    assert "frame-ancestors 'none'" in h["content-security-policy"]
    assert h["x-frame-options"] == "DENY"
    assert h["x-content-type-options"] == "nosniff"
    assert h["referrer-policy"] == "no-referrer"


def test_patient_pages_are_never_cached(clinician, some_ref):
    for path in ("/", f"/patient/{some_ref}"):
        assert clinician.get(path).headers["cache-control"] == "no-store"


def test_no_javascript_anywhere(clinician, some_ref):
    for path in ("/login", "/", f"/patient/{some_ref}"):
        html = clinician.get(path).text.lower()
        assert "<script" not in html and "onclick" not in html and "javascript:" not in html


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_api_docs_are_disabled(client, path):
    assert client.get(path).status_code == 404


# ---------------------------------------------------------------- hostile input
@pytest.mark.parametrize("ref", ["zzz", "000000000000", "ABCDEF123456", "..%2f..%2fetc%2fpasswd",
                                 "1234567890123", "%00", "<script>"])
def test_bad_patient_refs_are_404(clinician, ref):
    assert clinician.get(f"/patient/{ref}").status_code == 404


@pytest.mark.parametrize("query", ["decision=<script>alert(1)</script>", "dz=' OR 1=1 --",
                                   "decision=No%20action%00", "dz=" + "A" * 5000])
def test_injected_filters_are_ignored_not_reflected(clinician, query):
    r = clinician.get(f"/?{query}")
    assert r.status_code == 200
    assert "<script>alert" not in r.text and "OR 1=1" not in r.text and "AAAAAAAAAA" not in r.text


@pytest.mark.parametrize("page", ["abc", "1e9", "-5", "99999999", "0"])
def test_weird_page_numbers_dont_crash(clinician, page):
    assert clinician.get(f"/?page={page}").status_code in (200, 400)


def test_validation_errors_do_not_echo_input(client):
    r = client.post("/login", data={"username": "u" * 500, "password": "p", "csrf": "x"})
    assert r.status_code == 400 and "uuuuuuuuuu" not in r.text


def test_briefing_requires_csrf_and_valid_language(clinician, some_ref):
    page = clinician.get(f"/patient/{some_ref}")
    token = csrf_of(page.text)
    assert clinician.post(f"/patient/{some_ref}/briefing", data={"csrf": "forged"}).status_code == 403
    assert clinician.post(f"/patient/{some_ref}/briefing",
                          data={"csrf": token, "language": "Klingon"}).status_code == 404
    r = clinician.post(f"/patient/{some_ref}/briefing", data={"csrf": token, "language": "Tamil (தமிழ்)"})
    assert r.status_code == 200 and "Prompt sent to Granite" in r.text and "Tamil" in r.text


def test_csrf_token_from_another_session_is_rejected(client, some_ref):
    from fastapi.testclient import TestClient
    with TestClient(dashboard.app) as other:
        login(other, "clin")
        stolen = csrf_of(other.get("/").text)
    login(client, "clin")
    assert client.post(f"/patient/{some_ref}/briefing", data={"csrf": stolen}).status_code == 403


# ---------------------------------------------------------------- privacy
def test_raw_patient_ids_never_appear(clinician, some_ref):
    raw_id = dashboard.PATIENTS[some_ref]["id"]
    html = clinician.get(f"/patient/{some_ref}").text
    assert f"/patient/{raw_id}\"" not in html
    assert all(re.fullmatch(r"[0-9a-f]{12}", m) for m in re.findall(r'/patient/([^"/]+)"', clinician.get("/").text))


def test_audit_log_has_pseudonyms_and_no_clinical_values(clinician, some_ref):
    clinician.get(f"/patient/{some_ref}")
    lines = [json.loads(l) for l in open(dashboard.audit.path, encoding="utf-8")]
    raw_ids = {p["id"] for p in dashboard.PATIENTS.values()}
    for e in lines:
        if "patient" in e:
            assert re.fullmatch(r"[0-9a-f]{12}", e["patient"]) and e["patient"] not in raw_ids
        assert not {"risk", "risk_6m", "decision", "age", "diagnosis", "password"} & set(e)


def test_scorer_receives_only_model_inputs():
    raw = next(iter(dashboard.RAW.values()), None) or {"death": "1", "race": "x", "prg6m": "0.5", "age": "70"}
    sent = dashboard.model_inputs(raw)
    assert not {"death", "d.time", "hospdead", "race", "income", "edu", "prg6m", "id"} & set(sent)


# ---------------------------------------------------------------- pages work
def test_every_filter_combination_renders(clinician):
    for d in [""] + dashboard.DECISIONS:
        for gap in ("", "1"):
            r = clinician.get("/", params={"decision": d, "gap": gap})
            assert r.status_code == 200


def test_worklist_is_sorted_by_priority_then_risk():
    order = dashboard.ORDER
    keys = [(dashboard.DECISIONS.index(p["decision"]), -p["risk"]) for p in order]
    assert keys == sorted(keys)


def test_pagination_covers_every_patient(clinician):
    seen = set()
    page = 1
    while True:
        html = clinician.get("/", params={"page": page}).text
        refs = set(re.findall(r'/patient/([0-9a-f]{12})"', html))
        if not refs or refs <= seen:
            break
        seen |= refs
        page += 1
    assert len(seen) == len(dashboard.PATIENTS)


@needs_raw
def test_patient_page_scores_live(clinician, some_ref):
    html = clinician.get(f"/patient/{some_ref}").text
    assert "Scored live: local scorer" in html and "<polyline" in html


def test_patient_page_falls_back_without_raw_record(clinician, some_ref, monkeypatch):
    monkeypatch.setattr(dashboard, "RAW", {})
    html = clinician.get(f"/patient/{some_ref}").text
    assert "Precomputed result" in html


def test_api_outage_falls_back_and_is_audited(clinician, some_ref, monkeypatch):
    def down(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(dashboard, "score", down)
    r = clinician.get(f"/patient/{some_ref}")
    assert r.status_code == 200 and "Precomputed result" in r.text
    assert dashboard.audit.tail(1)[0]["event"] == "api_unreachable"


def test_unexpected_errors_show_generic_page(clinician, some_ref, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(dashboard, "patient_view", boom)
    from fastapi.testclient import TestClient
    with TestClient(dashboard.app, raise_server_exceptions=False) as c:
        c.cookies = clinician.cookies
        r = c.get(f"/patient/{some_ref}")
    assert r.status_code == 500 and "secret internal detail" not in r.text and "Traceback" not in r.text
