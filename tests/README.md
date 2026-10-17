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

Same 6 tests on a Windows laptop and on our **IBM LinuxONE z17** server (3 virtual CPUs), 17 Oct 2026. All 6 pass on both.

| Test | Load | Windows laptop | **IBM z17 (LinuxONE)** |
|---|---|---|---|
| Concurrent doctors | 30 users at once, 600 requests | 0 errors, median 336 ms, p95 434 ms, 54 req/s | **0 errors, median 76 ms, p95 126 ms, 241 req/s** |
| Scoring burst | 200 patient pages from 50 threads | All live-scored, p95 547 ms | **All live-scored, median 103 ms, p95 128 ms** |
| Role isolation under load | 20 mixed doctor/auditor users | No role ever saw the other's pages | **Same: no leaks** |
| Password spraying | 200 usernames from one IP | Locked after exactly 20; correct password then refused | **Same: locked after 20 (200 attempts in 1.1 s)** |
| Simultaneous logins | 50 at the same moment | 50 unique working sessions | **Same: 50 unique sessions** |
| Model throughput | All 1,821 test patients | 212 patients/s (4.7 ms each) | **1,381 patients/s (0.7 ms each)** |

Throughput counts the full model per patient (main model, 20 bootstrap models and XGBoost). On the z17, the dashboard served about **4.5x more requests per second** than on the laptop, and scored patients about **6.5x faster**. The z17 run was done while Granite was idle, so the two didn't compete for CPU.
