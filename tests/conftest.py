"""
Shared test setup. Everything runs against a throwaway users file, audit log and salt in a temp
folder, so tests never touch app/users.json or the real audit log.
"""
import os, re, secrets, sys, tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="ttt-tests-"))

# must be set before app.dashboard is imported (it reads config at import time)
os.environ.update({
    "TTT_ID_SALT": secrets.token_hex(16),
    "TTT_COOKIE_SECURE": "0",                       # TestClient talks plain http
    "TTT_USE_API": "0",
    "TTT_USERS_FILE": str(TMP / "users.json"),
    "TTT_DASH_AUDIT_LOG": str(TMP / "audit.log"),
    "TTT_ARTIFACTS": str(ROOT / "artifacts_logistic"),
    "TTT_DATA": str(ROOT / "data" / "support2.csv"),
})
for p in (ROOT, ROOT / "app", ROOT / "ml"):
    sys.path.insert(0, str(p))

from security import UserStore, hash_password  # noqa: E402

# test-only accounts with random passwords generated per run
PASSWORDS = {"clin": secrets.token_urlsafe(18), "audit": secrets.token_urlsafe(18), "gone": secrets.token_urlsafe(18)}
UserStore(os.environ["TTT_USERS_FILE"]).save({
    "clin": {"hash": hash_password(PASSWORDS["clin"]), "role": "doctor"},
    "audit": {"hash": hash_password(PASSWORDS["audit"]), "role": "auditor"},
    "gone": {"hash": hash_password(PASSWORDS["gone"]), "role": "doctor", "disabled": True},
})

from app import dashboard  # noqa: E402

HAS_RAW = bool(dashboard.RAW)
needs_raw = pytest.mark.skipif(not HAS_RAW, reason="data/support2.csv not present (live scoring disabled)")
CSRF_RE = re.compile(r'name="csrf" value="([^"]+)"')


@pytest.fixture(autouse=True)
def reset_state():
    """Lockout counters are per-process; clear them so tests don't lock each other out."""
    dashboard.guard.fails.clear()
    yield
    dashboard.guard.fails.clear()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    with TestClient(dashboard.app) as c:
        yield c


def login(client, user, password=None, follow=False):
    r = client.get("/login")
    token = CSRF_RE.search(r.text).group(1)
    return client.post("/login", data={"username": user, "password": PASSWORDS.get(user) if password is None else password,
                                       "csrf": token}, follow_redirects=follow)


def csrf_of(page_text):
    return CSRF_RE.search(page_text).group(1)


@pytest.fixture
def doctor(client):
    assert login(client, "clin").status_code == 303
    return client


@pytest.fixture
def auditor(client):
    assert login(client, "audit").status_code == 303
    return client


@pytest.fixture
def some_ref():
    return dashboard.ORDER[0]["ref"]
