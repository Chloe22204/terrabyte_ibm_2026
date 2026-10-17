# TimeToTalk doctor dashboard

FastAPI + server-rendered HTML (no JavaScript). Runs on LinuxONE (s390x): every dependency is
pure Python or has an s390x wheel. Streamlit was rejected because its `pyarrow` dependency has
no s390x build.

## Pages
- **Worklist**: test-set patients sorted by priority (Due now → Doctor judgement → Review → No action),
  filters by decision / diagnosis / optimism gap, counts per decision. A **System** panel shows where
  inference runs (machine, bootstrap models, second opinion, Granite) and a **Benchmark** button
  scores one patient 500 times and reports patients per second.
- **Patient**: 6-month risk with its 80% range and the decision thresholds, second opinion, survival
  curve, main factors (with the patient's values), doctor's estimate vs model, and the Granite briefing.
  The doctor can enter their own 6-month survival estimate to see doctor, model and combined risk side by side
  (combined = the API's 50/50 blend); the estimate itself is never written to the audit log.
- **Audit log** (auditor role only).

## Run locally (Windows PowerShell, from the repo root)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r app/requirements.txt
$env:TTT_ID_SALT = python -c "import secrets; print(secrets.token_hex(16))"
$env:TTT_COOKIE_SECURE = "0"            # plain http on localhost only
python app/manage_users.py add <your_username> --role doctor
uvicorn app.dashboard:app --host 127.0.0.1 --port 8501
```
Then open http://localhost:8501. Needs `data/support2.csv` for live scoring; without it the
dashboard shows precomputed results.

## Run on LinuxONE (Podman, HTTPS)
```bash
mkdir -p ~/ttt-certs ~/ttt-config
openssl req -x509 -newkey rsa:3072 -nodes -days 30 -subj "/CN=timetotalk" \
  -keyout ~/ttt-certs/key.pem -out ~/ttt-certs/cert.pem
python app/manage_users.py add dr_tan --file ~/ttt-config/users.json
podman build -t timetotalk-dashboard -f app/Containerfile .
podman run -d --name ttt-dash --network host \
  -e TTT_ID_SALT=$(openssl rand -hex 16) \
  -v ~/ttt-certs:/certs:ro,Z -v ~/ttt-config:/config:ro,Z -v $PWD/data:/data:ro,Z \
  timetotalk-dashboard
```
`--network host` lets the dashboard reach Granite (llama-server on the VM's 127.0.0.1:8080, see
docs/SETUP.md step 3e). With `-p 8501:8501` instead, the container has its own network, can't see
Granite, and every briefing falls back to the template.

Open it through an SSH tunnel rather than exposing the port: `ssh -L 8501:localhost:8501 linux1@<server>`,
then https://localhost:8501.

To score through the API instead of locally, add `-e TTT_USE_API=1 -e TTT_API_URL=... -e TTT_API_TOKEN=...`.

## Security
Full details, test results and known limitations: **[SECURITY.md](SECURITY.md)**.

In short:
- **Login:** scrypt-hashed passwords, lockout after 5 failed attempts, generic errors.
- **Sessions:** server-side random tokens in an `HttpOnly`/`Secure`/`SameSite=Strict` cookie, with 15-minute idle timeout.
- **Roles:** doctors see patients; auditors see only the audit log.
- **Privacy:** patient IDs replaced by keyed pseudonyms everywhere, and only model inputs are sent for scoring.
- **Browser:** no JavaScript, strict Content-Security-Policy, CSRF tokens, anti-framing, no caching.
- **Audit:** every login and patient view is logged, with no clinical values.
- **Deployment:** HTTPS-only, non-root container reached through an SSH tunnel.
