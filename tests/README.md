# Tests

Automated tests for the dashboard, its security controls and the scoring model.

```bash
pip install -r tests/requirements.txt
pytest                 # 100 functional + security tests, about 15 s
pytest -m stress -s    # 6 stress tests against a real server, about 40 s (-s prints timings)
```
Run from the repo root. Tests use throwaway users, a random salt and a temporary audit log, so
they never touch `app/users.json` or real logs. Tests that need `data/support2.csv` skip
themselves if it isn't there.

## What's covered

| File | Tests | What it checks |
|---|---|---|
| `test_security_unit.py` | 28 | scrypt hashing (salted, malformed hashes rejected), lockout per user / per IP / expiry, session tokens (unique, timeouts), pseudonyms (stable, salted, no collisions on 9,105 IDs), audit log (no log injection, 20 threads writing at once) |
| `test_dashboard_web.py` | 58 | Login redirects, cookie flags, identical failed-login responses, login CSRF, lockout beats the right password, logout kills the session server-side, forged cookies, roles both ways, security headers on every page, no JavaScript, API docs off, path traversal / XSS / SQL-style / oversized input, CSRF on briefings (including a token stolen from another session), raw IDs never shown or logged, API-outage fallback, no stack traces on errors |
| `test_scoring.py` | 14 | Live scores match `train.py`'s results exactly, valid probabilities and survival curves, grouped explanations, missing / unknown / extreme / non-numeric inputs, determinism |
| `test_stress.py` | 6 | Real server under load (see below) |

## Stress test results

Windows laptop, 17 Oct 2026 (LinuxONE numbers to be added):

| Test | Load | Result |
|---|---|---|
| Concurrent clinicians | 30 users at once, 600 requests | 0 errors, median 336 ms, p95 434 ms, 54 req/s |
| Scoring burst | 200 patient pages from 50 threads | All live-scored, p95 547 ms |
| Role isolation under load | 20 mixed clinician/auditor users | No role ever saw the other's pages |
| Password spraying | 200 usernames from one IP | Locked after exactly 20; correct password then refused |
| Simultaneous logins | 50 at the same moment | 50 unique working sessions |
| Model throughput | All 1,821 test patients | 212 patients/s (4.7 ms each, incl. 20 bootstrap models + XGBoost) |
