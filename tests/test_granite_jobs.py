"""Slow Granite answers: the page shows progress and refreshes itself until the background job finishes."""
import re
import time

import pytest

from app import dashboard
from conftest import csrf_of, login, needs_raw

REFRESH = re.compile(r'http-equiv="refresh" content="3;url=(/patient/[0-9a-f]{12}/job/[A-Za-z0-9_-]+)\?t=\d+#(\w+)"')


@pytest.fixture
def slow_granite(monkeypatch):
    monkeypatch.setattr(dashboard, "JOB_WAIT_S", 0.05)
    monkeypatch.setattr(dashboard.prep_sheet, "chat", lambda *a, **k: (time.sleep(0.4), "1. Why now\nSlow but here.")[1])


def wait_for(c, url, tries=40):
    for _ in range(tries):
        r = c.get(url)
        if not REFRESH.search(r.text):
            return r
        time.sleep(0.05)
    raise AssertionError("job never finished")


@needs_raw
def test_slow_prep_sheet_shows_progress_then_the_sheet(doctor, some_ref, slow_granite):
    token = csrf_of(doctor.get(f"/patient/{some_ref}").text)
    r = doctor.post(f"/patient/{some_ref}/briefing", data={"csrf": token, "language": "English"})
    assert r.status_code == 200 and "Granite is writing the prep sheet" in r.text
    url, anchor = REFRESH.search(r.text).groups()
    assert anchor == "briefing" and "Draft prep sheet</button>" in r.text and "disabled" in r.text
    done = wait_for(doctor, url)
    assert "Written by Granite" in done.text and "Slow but here." in done.text


@needs_raw
def test_slow_answer_shows_progress_then_the_answer(doctor, some_ref, slow_granite):
    token = csrf_of(doctor.get(f"/patient/{some_ref}").text)
    r = doctor.post(f"/patient/{some_ref}/why", data={"csrf": token, "question": "How do I start?"})
    assert "Granite is looking through the guidelines" in r.text
    url, anchor = REFRESH.search(r.text).groups()
    assert anchor == "why"
    assert "Answered by Granite" in wait_for(doctor, url).text


@needs_raw
def test_job_page_is_only_for_the_user_who_started_it(client, some_ref, slow_granite):
    login(client, "clin")
    token = csrf_of(client.get(f"/patient/{some_ref}").text)
    url = REFRESH.search(client.post(f"/patient/{some_ref}/briefing",
                                     data={"csrf": token, "language": "English"}).text).group(1)
    job = dashboard.jobs[url.rsplit("/", 1)[1]]
    job["user"] = "someone-else"
    assert client.get(url).status_code == 404


@pytest.mark.parametrize("job_id", ["nope", "../../audit", "A" * 40, "x" * 22])
def test_unknown_job_ids_are_not_found(doctor, some_ref, job_id):
    assert doctor.get(f"/patient/{some_ref}/job/{job_id}").status_code == 404


def test_fast_answers_still_come_back_in_one_request(doctor, some_ref, monkeypatch):
    monkeypatch.setattr(dashboard.prep_sheet, "chat", lambda *a, **k: "1. Why now\nQuick.")
    token = csrf_of(doctor.get(f"/patient/{some_ref}").text)
    r = doctor.post(f"/patient/{some_ref}/briefing", data={"csrf": token, "language": "English"})
    assert "Quick." in r.text and not REFRESH.search(r.text)
