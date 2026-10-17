"""
Security building blocks for the TimeToTalk dashboard. Standard library only, so it runs
anywhere (including s390x) with nothing to compile.

  - passwords: scrypt with a per-user random salt, constant-time comparison
  - brute force: lockout after repeated failures, per username and per client IP
  - sessions: server-side, 256-bit random tokens, idle + absolute timeouts, rotated on login
  - CSRF: per-session token checked on every state-changing request
  - audit: append-only JSON lines, never any clinical values
  - pseudonyms: patient IDs replaced by a keyed hash before they reach a page, URL or log
"""
import hashlib, hmac, json, os, secrets, threading, time
from datetime import datetime, timezone

# scrypt cost: 2^14 iterations, 16 MB memory per hash (OWASP-recommended minimum)
SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}
ROLES = ("doctor", "auditor")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **SCRYPT)
    return f"scrypt${SCRYPT['n']}${SCRYPT['r']}${SCRYPT['p']}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, dk = stored.split("$")
        expected = bytes.fromhex(dk)
        got = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), dklen=len(expected),
                             n=int(n), r=int(r), p=int(p))
        return hmac.compare_digest(got, expected)
    except (ValueError, TypeError, AttributeError):    # malformed or missing hash never verifies
        return False


# checked against when the username doesn't exist, so a wrong username takes as long as a wrong password
_DUMMY_HASH = hash_password(secrets.token_hex(16))


class UserStore:
    """Users live in a JSON file outside git (see manage_users.py). Re-read on each login so
    adding or removing a user takes effect without a restart."""

    def __init__(self, path):
        self.path = path

    def load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f).get("users", {})
        except FileNotFoundError:
            return {}

    def save(self, users: dict):
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"users": users}, f, indent=1)
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)     # owner read/write only (no effect on Windows)
        except OSError:
            pass

    def authenticate(self, username: str, password: str):
        """Returns the user's role, or None. Same work whether or not the user exists."""
        user = self.load().get(username)
        ok = verify_password(password, user["hash"] if user else _DUMMY_HASH)
        return user["role"] if (user and ok and not user.get("disabled")) else None


class LoginGuard:
    """Locks a username after 5 failures in 15 minutes, and a client IP after 20."""

    def __init__(self, max_user=5, max_ip=20, window_s=900):
        self.max_user, self.max_ip, self.window = max_user, max_ip, window_s
        self.fails = {}
        self.lock = threading.Lock()

    def _recent(self, key, now):
        self.fails[key] = [t for t in self.fails.get(key, []) if now - t < self.window]
        return self.fails[key]

    def is_locked(self, username, ip) -> bool:
        now = time.time()
        with self.lock:
            return (len(self._recent(("u", username), now)) >= self.max_user
                    or len(self._recent(("ip", ip), now)) >= self.max_ip)

    def failed(self, username, ip):
        now = time.time()
        with self.lock:
            self._recent(("u", username), now).append(now)
            self._recent(("ip", ip), now).append(now)

    def succeeded(self, username):
        with self.lock:
            self.fails.pop(("u", username), None)


class SessionStore:
    """Server-side sessions. The cookie holds only a random token; the role and CSRF token
    stay on the server, so a cookie can't be edited to gain access."""

    def __init__(self, idle_s=15 * 60, absolute_s=8 * 3600):
        self.idle, self.absolute = idle_s, absolute_s
        self.sessions = {}
        self.lock = threading.Lock()

    def create(self, username, role) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self.lock:
            self.sessions[token] = {"user": username, "role": role, "csrf": secrets.token_urlsafe(32),
                                    "created": now, "seen": now}
        return token

    def get(self, token):
        if not token:
            return None
        now = time.time()
        with self.lock:
            s = self.sessions.get(token)
            if s and (now - s["seen"] > self.idle or now - s["created"] > self.absolute):
                del self.sessions[token]
                return None
            if s:
                s["seen"] = now
            return s

    def end(self, token):
        with self.lock:
            self.sessions.pop(token, None)


class AuditLog:
    """Who did what, when. Patient references are pseudonyms; clinical values are never written."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def write(self, event, **fields):
        line = json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "event": event, **fields})
        with self.lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def tail(self, n=200):
        try:
            with open(self.path, encoding="utf-8") as f:
                lines = f.readlines()[-n:]
        except FileNotFoundError:
            return []
        return [json.loads(l) for l in reversed(lines) if l.strip()]


def pseudonym(salt: str, patient_id) -> str:
    """Keyed hash of a patient ID (HMAC-SHA256). Without the salt it can't be reversed or recomputed."""
    return hmac.new(salt.encode(), str(patient_id).encode(), hashlib.sha256).hexdigest()[:12]


def same(a: str, b: str) -> bool:
    return bool(a) and bool(b) and hmac.compare_digest(a, b)
