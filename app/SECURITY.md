# Dashboard security

The TimeToTalk dashboard shows mortality risk for seriously ill patients, so it is built as if it
held real health data. This page lists every control, where it lives in the code, how it was
tested, and what a real hospital deployment would still need.

Code: [security.py](security.py) (passwords, lockout, sessions, audit, pseudonyms) and
[dashboard.py](dashboard.py) (routes, headers, roles, input checks).

## Threats considered

| Threat | Main controls |
|---|---|
| Someone without an account reads patient data | Login on every page, roles, HTTPS + SSH tunnel |
| Password guessing | scrypt hashes, lockout, generic error messages |
| A stolen password database | scrypt with per-user salt (passwords can't be read back) |
| Hijacking a logged-in session | Random server-side sessions, `HttpOnly`/`Secure`/`SameSite` cookie, timeouts |
| Another website acting as the user (CSRF) | Per-session CSRF tokens, `SameSite=Strict` |
| Injected scripts (XSS) | No JavaScript, strict Content-Security-Policy, auto-escaped HTML, whitelisted inputs |
| Patient identities leaking through screens, URLs, logs or the API | Keyed pseudonyms, data minimisation |
| Misuse by a legitimate user | Audit log of every patient view, role separation |

## Controls

### 1. Authentication
| Control | Detail |
|---|---|
| Login required | Every page except `/login` and `/healthz` needs a session; others redirect to sign-in. |
| Password storage | **scrypt** (N=2^14, r=8, p=1), 16-byte random salt per user, 32-byte key. Only the hash is stored, in `app/users.json` (git-ignored, file mode 600). |
| Constant-time checks | Hash comparison uses `hmac.compare_digest`. An unknown username is checked against a dummy hash, so it takes as long as a wrong password. |
| Generic errors | Always "Wrong username or password", never which one. |
| Brute-force lockout | 5 failures for one username in 15 min locks it for 15 min; 20 from one IP locks the IP. |
| Password policy | Minimum 12 characters, entered at a hidden prompt (`manage_users.py`), never on the command line. |
| Account management | `manage_users.py add / disable / remove / list`. Disabled accounts can't log in. |

### 2. Sessions
| Control | Detail |
|---|---|
| Server-side sessions | The cookie holds only a 256-bit random token (`secrets.token_urlsafe(32)`). User, role and CSRF token stay on the server, so editing the cookie can't change your role. |
| Cookie flags | `HttpOnly`, `Secure`, `SameSite=Strict`, `Path=/`, and the `__Host-` name prefix (browser enforces HTTPS, no domain sharing). |
| Rotation | A brand-new token on every login (prevents session fixation). |
| Timeouts | 15 minutes idle (`TTT_IDLE_MIN`), 8 hours absolute. |
| Logout | Ends the session on the server, not just in the browser. Logout is a CSRF-protected POST. |

### 3. Authorisation
| Role | Can | Can't |
|---|---|---|
| `clinician` | Worklist, patient pages, briefings | Audit log |
| `auditor` | Audit log | Any patient page |

Every denial returns `403` and is written to the audit log.

### 4. Patient privacy
| Control | Detail |
|---|---|
| Pseudonyms | Patient IDs are replaced by `HMAC-SHA256(TTT_ID_SALT, id)`, first 12 hex characters. Raw IDs never appear in a page, URL, log or API call. Without the salt, a pseudonym can't be reversed or recomputed. |
| Data minimisation | Only the model's input columns are sent for scoring. Outcomes, race, income, education and the doctor's estimate are never sent to the scorer or API. |
| No caching | `Cache-Control: no-store` on every page. |
| On-premises | Scoring and Granite run on the LinuxONE server; no patient data goes to an external service. |
| Clinician-only | Footer and design: decision support for clinicians, never shown to patients. |

### 5. Browser protections
| Header / control | Value |
|---|---|
| Content-Security-Policy | `default-src 'self'; script-src 'none'; style-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'` |
| No JavaScript | Charts are server-drawn SVG, so the page needs no scripts at all and the CSP can forbid them. |
| X-Frame-Options | `DENY` (no clickjacking) |
| X-Content-Type-Options | `nosniff` |
| Referrer-Policy | `no-referrer` (pseudonyms in URLs don't leak to other sites) |
| Permissions-Policy | camera, microphone and location disabled |
| Cross-Origin-Opener-Policy | `same-origin` |
| Strict-Transport-Security | `max-age=31536000; includeSubDomains` when served over HTTPS |
| CSRF | Per-session token in every POST form; login uses a double-submit token cookie. |

### 6. Input handling
- Filter values (decision, diagnosis) and briefing languages are **whitelisted**.
- Patient references must match `^[0-9a-f]{12}$`.
- Form fields are length-limited (username 64, password 256).
- All output is HTML-escaped by Jinja2 autoescaping.
- API docs (`/docs`, `/openapi.json`) are disabled; errors show a generic page with no stack trace (the error type goes to the audit log).

### 7. Audit log
JSON lines in `app/logs/dashboard_audit.log` (git-ignored), viewable by auditors at `/audit`.

| Event | Fields |
|---|---|
| `login`, `login_failed`, `login_locked`, `logout` | user, role, IP |
| `view_worklist` | user, filters used |
| `view_patient`, `briefing_requested` | user, patient pseudonym, language |
| `access_denied`, `csrf_rejected` | user, path |
| `api_unreachable`, `server_error` | user / path, error type |

No clinical values (risk, diagnosis, labs) are ever written to the log.

### 8. Deployment (LinuxONE)
| Control | Detail |
|---|---|
| HTTPS only | The container passes `--ssl-keyfile`/`--ssl-certfile` and won't start without a certificate in `/certs`. |
| No open port | Reached through an SSH tunnel (`ssh -L 8501:localhost:8501 ...`). |
| Non-root | The container runs as `appuser`. |
| Read-only mounts | Data, users file and certificates are mounted read-only, never baked into the image. |
| Secrets | `TTT_ID_SALT` and `TTT_API_TOKEN` come from environment variables; the dashboard refuses to start without a salt of at least 16 characters. `users.json`, logs and `*.pem` are git-ignored. |
| API calls | When `TTT_USE_API=1`, the dashboard calls the API with a bearer token over HTTP(S) with certificate verification and a 10-second timeout. |

## How it was tested

Tested on 17 Oct 2026 against the running dashboard (local mode and API mode), using a browser and
`curl`:

| # | Test | Expected | Result |
|---|---|---|---|
| 1 | Open `/` while signed out | Redirect to `/login` | ✅ |
| 2 | Response headers | CSP, `DENY`, `nosniff`, `no-referrer`, `no-store`, `HttpOnly; SameSite=strict` cookie | ✅ |
| 3 | Wrong password | Generic error, `401` | ✅ |
| 4 | Login POST with a forged CSRF token | Rejected, back to `/login` | ✅ |
| 5 | Briefing POST with a forged CSRF token | `403`, logged as `csrf_rejected` | ✅ |
| 6 | Clinician opens `/audit` | `403`, logged as `access_denied` | ✅ |
| 7 | Auditor opens a patient page | `403` | ✅ |
| 8 | Invalid / path-traversal patient ref | `404` | ✅ |
| 9 | `<script>` injected in a filter | Ignored, never rendered | ✅ |
| 10 | `/docs`, `/openapi.json` | `404` | ✅ |
| 11 | 5 wrong passwords, then a 6th try | "Too many failed attempts" | ✅ |
| 12 | Audit log contents | Pseudonyms only, no raw IDs or clinical values | ✅ |
| 13 | Scoring through the API | Same result as local; API log shows only a pseudonym | ✅ |

Not yet tested: the HTTPS container on LinuxONE (certificate, `Secure` cookies, HSTS).

These checks are now automated, with 100+ more, in [tests/](../tests/README.md), including stress tests under concurrent load.

### Reproduce the checks (screenshots for judging)
```bash
curl -s -D - -o /dev/null https://localhost:8501/login -k        # security headers
curl -s -o /dev/null -w "%{http_code}\n" https://localhost:8501/docs -k   # 404
```
In the browser: try 6 wrong passwords (lockout message), sign in as a clinician and open `/audit`
(403 page), then sign in as an auditor to show the audit log with pseudonyms only.

## Known limitations (production next steps)
| Gap | Production approach |
|---|---|
| Local username/password | Hospital single sign-on (OIDC/SAML) with multi-factor authentication |
| Self-signed certificate | Certificate from the hospital's certificate authority |
| Sessions and lockout counters in memory (lost on restart, single instance only) | Shared store such as Redis for multiple instances |
| Audit log as a local file | Ship to a central SIEM; tamper-evident (append-only, hash-chained) storage |
| Salt and tokens in environment variables | A secrets manager or hardware-backed keys (IBM Z crypto / HSM) |
| No password expiry or reset flow | Handled by the hospital identity provider |
| Dataset is 1989–94 US research data | Retrain and validate on local hospital records before any clinical use |
