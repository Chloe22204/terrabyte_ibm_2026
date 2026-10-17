"""The prep sheet and "Ask the guidelines" boxes, wired to genai/prep_sheet.py (Granite is mocked here)."""
import json

import pytest

from app import dashboard
from conftest import csrf_of, needs_raw

GRANITE_TEXT = "1. Why now\nThe risk is high.\n2. Before you go in\nInvite the family."


@pytest.fixture
def granite_up(monkeypatch):
    """Pretend llama-server answered; records the messages Granite would have received."""
    calls = []

    def fake_chat(messages, max_tokens=350, temperature=0.2):
        calls.append(messages)
        return GRANITE_TEXT
    monkeypatch.setattr(dashboard.prep_sheet, "chat", fake_chat)
    return calls


@pytest.fixture
def granite_down(monkeypatch):
    def down(*a, **k):
        raise dashboard.prep_sheet.GraniteUnavailable("llama-server not reachable")
    monkeypatch.setattr(dashboard.prep_sheet, "chat", down)


def post_briefing(c, ref, language="English"):
    token = csrf_of(c.get(f"/patient/{ref}").text)
    return c.post(f"/patient/{ref}/briefing", data={"csrf": token, "language": language})


def post_why(c, ref, question):
    token = csrf_of(c.get(f"/patient/{ref}").text)
    return c.post(f"/patient/{ref}/why", data={"csrf": token, "question": question})


def last_event(name):
    return next(e for e in dashboard.audit.tail(50) if e["event"] == name)


# ---------------------------------------------------------------- prep sheet
def test_granite_sheet_is_shown(doctor, some_ref, granite_up):
    r = post_briefing(doctor, some_ref)
    assert r.status_code == 200 and "Written by Granite" in r.text
    assert "Invite the family." in r.text and "Guidance used" in r.text
    assert len(granite_up) == 1


def test_template_when_granite_is_down(doctor, some_ref, granite_down):
    r = post_briefing(doctor, some_ref)
    assert r.status_code == 200 and "Template (Granite unavailable)" in r.text
    assert "1. Why now" in r.text and "running on the server right now" in r.text
    assert "llama-server not reachable" not in r.text           # no technical detail shown to doctors


@pytest.mark.parametrize("label,key", [("English", "English"), ("Chinese (中文)", "Simplified Chinese"),
                                       ("Malay (Melayu)", "Bahasa Melayu"), ("Tamil (தமிழ்)", "Tamil")])
def test_language_reaches_granite_prompt(doctor, some_ref, granite_up, label, key):
    post_briefing(doctor, some_ref, label)
    assert f"in {key}" in granite_up[0][1]["content"]


@needs_raw
def test_prompt_is_deidentified(doctor, some_ref, granite_up):
    post_briefing(doctor, some_ref)
    prompt = json.dumps(granite_up[0])
    p = dashboard.PATIENTS[some_ref]
    assert some_ref not in prompt and f"id {p['id']}" not in prompt
    assert f"{p['age']}-year" not in prompt              # age band only ("in their 60s")
    assert "in their" in prompt


def test_unsafe_granite_output_is_flagged_and_escaped(doctor, some_ref, monkeypatch):
    monkeypatch.setattr(dashboard.prep_sheet, "chat",
                        lambda *a, **k: "<script>alert(1)</script> She will die in 3 months. Risk 12%.")
    r = post_briefing(doctor, some_ref)
    assert "<script>alert(1)</script>" not in r.text and "&lt;script&gt;" in r.text
    assert "time-to-death statement" in r.text            # genai guardrail warning shown
    assert "not one of the model" in r.text               # invented number flagged


def test_sheet_text_is_never_audited(doctor, some_ref, granite_up):
    post_briefing(doctor, some_ref)
    e = last_event("briefing_requested")
    assert e["used_llm"] is True and e["patient"] == some_ref
    assert "Invite the family" not in json.dumps(dashboard.audit.tail(50))


# ---------------------------------------------------------------- ask the guidelines (conversation coach)
COACH_Q = "The family wants everything done. How do I respond?"


def test_coach_answer_is_shown_with_guidance(doctor, some_ref, monkeypatch):
    monkeypatch.setattr(dashboard.prep_sheet, "chat", lambda *a, **k: "Ask about their goals first.")
    r = post_why(doctor, some_ref, COACH_Q)
    assert r.status_code == 200 and "Answered by Granite" in r.text and "Ask about their goals first." in r.text
    assert "Guidance used" in r.text and "language_to_avoid.md" in r.text


def test_coach_prompt_carries_retrieved_passages(doctor, some_ref, granite_up):
    post_why(doctor, some_ref, COACH_Q)
    prompt = granite_up[0][1]["content"]
    assert COACH_Q in prompt and "[language_to_avoid.md]" in prompt
    assert "GUIDELINE PASSAGES" in prompt


def test_coach_off_topic_question_skips_granite(doctor, some_ref, granite_up):
    r = post_why(doctor, some_ref, "What is the capital of France?")
    assert "Not covered by the notes" in r.text and "cover this question" in r.text
    assert granite_up == []                                   # no Granite call for an unanswerable question


def test_coach_without_granite_shows_guidance(doctor, some_ref, granite_down):
    r = post_why(doctor, some_ref, COACH_Q)
    assert "Granite unavailable: guidance shown" in r.text and "Do you want us to do everything?" in r.text


def test_ask_why_input_limits(doctor, some_ref, granite_up):
    assert post_why(doctor, some_ref, "   ").status_code in (400, 404)
    assert post_why(doctor, some_ref, "x" * 501).status_code == 400
    assert doctor.post(f"/patient/{some_ref}/why", data={"csrf": "forged", "question": "hi"}).status_code == 403


def test_ask_why_question_is_escaped_and_not_logged(doctor, some_ref, granite_up):
    q = "<img src=x onerror=alert(1)> secret-question-text"
    r = post_why(doctor, some_ref, q)
    assert "<img src=x" not in r.text and "&lt;img" in r.text
    assert "secret-question-text" not in json.dumps(dashboard.audit.tail(50))


# ---------------------------------------------------------------- API mode
def test_api_mode_calls_prep_sheet_endpoint(doctor, some_ref, monkeypatch):
    if not dashboard.RAW:
        pytest.skip("needs data/support2.csv")
    sent = []

    def fake_api(path, payload, timeout=10):
        sent.append((path, payload, timeout))
        if path == "/score":
            return dashboard.scorer.score(payload["patient"])
        return {"text": "From the API", "language": payload["language"], "used_llm": True, "latency_s": 42.0,
                "warnings": [], "sources": [], "passages": []}
    monkeypatch.setattr(dashboard, "USE_API", True)
    monkeypatch.setattr(dashboard, "api_post", fake_api)
    r = post_briefing(doctor, some_ref, "Malay (Melayu)")
    assert "From the API" in r.text
    path, payload, timeout = [s for s in sent if s[0] == "/prep_sheet"][0]
    assert payload["language"] == "Malay" and payload["patient_id"] == some_ref
    assert not {"death", "race", "income", "prg6m"} & set(payload["patient"])     # minimised
    assert timeout > 180                                  # long enough for Granite on 3 vCPUs


def test_api_down_falls_back_to_local_template(doctor, some_ref, monkeypatch):
    if not dashboard.RAW:
        pytest.skip("needs data/support2.csv")

    def down(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(dashboard, "USE_API", True)
    monkeypatch.setattr(dashboard, "api_post", down)
    r = post_briefing(doctor, some_ref)
    assert r.status_code == 200 and "API unreachable" in r.text and "1. Why now" in r.text
