"""Doctor's-estimate what-if, the System panel and the benchmark button (ported from the API demo screen)."""
import pytest

from app import dashboard
from conftest import csrf_of, needs_raw


def post_doctor(c, ref, value):
    token = csrf_of(c.get(f"/patient/{ref}").text)
    return c.post(f"/patient/{ref}/doctor", data={"csrf": token, "doctor": value})


def last_event(name):
    return next(e for e in dashboard.audit.tail(50) if e["event"] == name)


# ---------------------------------------------------------------- doctor's estimate
@needs_raw
def test_own_estimate_shows_doctor_model_and_combined(doctor, some_ref):
    r = post_doctor(doctor, some_ref, "30")
    assert r.status_code == 200
    assert "Your estimate." in r.text and "Combined" in r.text and "Doctor 70%" in r.text   # 30% survival = 70% risk
    assert 'name="doctor" value="30"' in r.text           # carried into the prep-sheet and ask-why forms


def test_doctor_combined_is_the_api_blend():
    p = {"doctor_surv": 0.5}
    d = dashboard.doctor_view(p, 0.8, "recorded")
    assert d["risk"] == pytest.approx(0.5) and d["combined"] == pytest.approx(0.65)
    assert dashboard.doctor_view(p, 0.8, "none")["surv"] is None
    assert dashboard.doctor_view(p, 0.4, "100")["gap_flag"]          # 100% vs model 60% survival: 40 pts more optimistic


def test_cleared_estimate_hides_comparison(doctor, some_ref):
    r = post_doctor(doctor, some_ref, "none")
    assert r.status_code == 200 and "Estimate cleared." in r.text and "Combined =" not in r.text


@pytest.mark.parametrize("value", ["101", "-5", "abc", "0.5", "1e2", "recorded; drop"])
def test_bad_estimates_are_refused(doctor, some_ref, value):
    assert post_doctor(doctor, some_ref, value).status_code == 400


def test_doctor_estimate_needs_csrf(doctor, some_ref):
    r = doctor.post(f"/patient/{some_ref}/doctor", data={"csrf": "forged", "doctor": "40"})
    assert r.status_code == 403


def test_estimate_value_is_never_audited(doctor, some_ref):
    post_doctor(doctor, some_ref, "37")
    e = last_event("doctor_estimate")
    assert e["source"] == "yours" and set(e) == {"ts", "event", "user", "patient", "source"}


@needs_raw
def test_briefing_keeps_the_estimate(doctor, some_ref, monkeypatch):
    monkeypatch.setattr(dashboard.prep_sheet, "chat", lambda *a, **k: "1. Why now\nok")
    token = csrf_of(doctor.get(f"/patient/{some_ref}").text)
    r = doctor.post(f"/patient/{some_ref}/briefing", data={"csrf": token, "language": "English", "doctor": "45"})
    assert r.status_code == 200 and 'name="doctor" value="45"' in r.text


# ---------------------------------------------------------------- system panel + benchmark
def test_worklist_shows_where_inference_runs(doctor):
    html = doctor.get("/").text
    assert "Bootstrap models" in html and "Granite" in html
    assert "All inference runs on this LinuxONE VM." in html or "not LinuxONE" in html


def test_linuxone_line_only_on_s390x(monkeypatch):
    monkeypatch.setattr(dashboard.platform, "machine", lambda: "s390x")
    dashboard._status.update(t=0.0, v=None)
    assert dashboard.system_status()["linuxone"]
    dashboard._status.update(t=0.0, v=None)


@needs_raw
def test_benchmark_reports_speed_and_is_audited(doctor, monkeypatch):
    monkeypatch.setattr(dashboard, "BENCH_N", 20)
    token = csrf_of(doctor.get("/").text)
    r = doctor.post("/benchmark", data={"csrf": token})
    assert r.status_code == 200 and "patients per second" in r.text
    assert last_event("benchmark")["n"] == 20


def test_benchmark_needs_csrf(doctor):
    assert doctor.post("/benchmark", data={"csrf": "forged"}).status_code == 403


def test_auditor_cannot_benchmark(auditor):
    assert auditor.post("/benchmark", data={"csrf": "x"}).status_code == 403
