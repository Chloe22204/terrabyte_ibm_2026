"""
Stress tests: a real uvicorn server on a random local port, hit by many concurrent clients.
Run with:  pytest -m stress -s     (-s prints the timing summary)
"""
import http.cookiejar, re, statistics, threading, time, urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest
import uvicorn

from app import dashboard
from conftest import HAS_RAW, PASSWORDS

pytestmark = pytest.mark.stress
CSRF_RE = re.compile(r'name="csrf" value="([^"]+)"')


@pytest.fixture(scope="module")
def server():
    config = uvicorn.Config(dashboard.app, host="127.0.0.1", port=0, log_level="warning")
    srv = uvicorn.Server(config)
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    while not srv.started:
        time.sleep(0.05)
    port = srv.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    t.join(timeout=10)


class Browser:
    """One user with their own cookie jar, like a separate browser."""

    def __init__(self, base):
        self.base = base
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def get(self, path):
        t0 = time.perf_counter()
        try:
            with self.opener.open(self.base + path, timeout=30) as r:
                return r.status, r.read().decode(), time.perf_counter() - t0
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(), time.perf_counter() - t0

    def post(self, path, data):
        body = urllib.parse.urlencode(data).encode()
        try:
            with self.opener.open(self.base + path, data=body, timeout=30) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    def login(self, user, password):
        _, html, _ = self.get("/login")
        return self.post("/login", {"username": user, "password": password, "csrf": CSRF_RE.search(html).group(1)})


def report(name, latencies):
    lat = sorted(latencies)
    p95 = lat[int(0.95 * (len(lat) - 1))]
    print(f"\n  {name}: {len(lat)} requests, median {statistics.median(lat) * 1000:.0f} ms, "
          f"p95 {p95 * 1000:.0f} ms, max {lat[-1] * 1000:.0f} ms")
    return p95


def test_concurrent_doctors_browsing(server):
    """30 doctors signed in at once, each opening 10 worklist pages and 10 patients (600 requests)."""
    refs = [p["ref"] for p in dashboard.ORDER[:300]]
    errors, latencies = [], []
    lock = threading.Lock()

    def doctor(n):
        b = Browser(server)
        status, _ = b.login("clin", PASSWORDS["clin"])
        if status != 200:
            errors.append(("login", status))
            return
        for i in range(10):
            for path in (f"/?page={(n + i) % 37 + 1}", f"/patient/{refs[(n * 10 + i) % len(refs)]}"):
                s, html, dt = b.get(path)
                with lock:
                    latencies.append(dt)
                if s != 200 or "Sign in" in html[:600]:
                    errors.append((path, s))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=30) as pool:
        list(pool.map(doctor, range(30)))
    wall = time.perf_counter() - t0
    p95 = report("concurrent browsing", latencies)
    print(f"  throughput: {len(latencies) / wall:.0f} requests/s over {wall:.1f} s")
    assert not errors, errors[:5]
    assert len(latencies) == 600
    assert p95 < 5.0


def test_burst_of_patient_scoring(server):
    """200 patient pages (each scored live with 20 bootstrap models) fired from 50 threads."""
    b = Browser(server)
    b.login("clin", PASSWORDS["clin"])
    refs = [p["ref"] for p in dashboard.ORDER[::9]][:200]

    def open_patient(ref):
        s, html, dt = b.get(f"/patient/{ref}")
        return s, ("Scored live" in html) == HAS_RAW, dt

    with ThreadPoolExecutor(max_workers=50) as pool:
        results = list(pool.map(open_patient, refs))
    report("patient scoring burst", [r[2] for r in results])
    assert all(s == 200 for s, _, _ in results)
    assert all(live_ok for _, live_ok, _ in results)


def test_sessions_stay_isolated_under_load(server):
    """Doctors and auditors interleaved: nobody ever sees the other role's pages."""
    leaks = []

    def worker(n):
        role = "clin" if n % 2 else "audit"
        b = Browser(server)
        b.login(role, PASSWORDS[role])
        for _ in range(10):
            s_audit, _, _ = b.get("/audit")
            s_list, _, _ = b.get("/")
            expected = (403, 200) if role == "clin" else (200, 403)
            if (s_audit, s_list) != expected:
                leaks.append((role, s_audit, s_list))

    with ThreadPoolExecutor(max_workers=20) as pool:
        list(pool.map(worker, range(20)))
    assert not leaks, leaks[:5]


def test_password_spraying_is_stopped():
    """One attacker IP tries 200 usernames: blocked after 20 failures, even for a real account."""
    from fastapi.testclient import TestClient
    dashboard.guard.fails.clear()
    with TestClient(dashboard.app) as c:
        def attempt(user, pw):
            html = c.get("/login").text
            return c.post("/login", data={"username": user, "password": pw,
                                          "csrf": CSRF_RE.search(html).group(1)})
        t0 = time.perf_counter()
        locked_at = None
        for i in range(200):
            r = attempt(f"guess{i}", "Password123!")
            if "Too many failed attempts" in r.text and locked_at is None:
                locked_at = i
        wall = time.perf_counter() - t0
        real = attempt("clin", PASSWORDS["clin"])
    print(f"\n  password spraying: locked out after {locked_at} attempts; 200 attempts took {wall:.1f} s")
    assert locked_at == 20
    assert "Too many failed attempts" in real.text            # even correct credentials are refused
    dashboard.guard.fails.clear()


def test_many_simultaneous_logins_get_unique_sessions(server):
    """50 logins at the same moment: every session token distinct, every one works."""
    browsers = [Browser(server) for _ in range(50)]
    with ThreadPoolExecutor(max_workers=50) as pool:
        statuses = list(pool.map(lambda b: b.login("clin", PASSWORDS["clin"])[0], browsers))
    assert all(s == 200 for s in statuses)
    tokens = {c.value for b in browsers for h in b.opener.handlers
              if isinstance(h, urllib.request.HTTPCookieProcessor)
              for c in h.cookiejar if c.name == dashboard.SESSION_COOKIE}
    assert len(tokens) == 50


@pytest.mark.skipif(not HAS_RAW, reason="needs data/support2.csv")
def test_scoring_throughput():
    """Raw model speed, no HTTP: score every test patient once."""
    patients = list(dashboard.PATIENTS.values())
    t0 = time.perf_counter()
    for p in patients:
        dashboard.scorer.score(dashboard.model_inputs(dashboard.RAW[p["id"]]))
    dt = time.perf_counter() - t0
    print(f"\n  scoring throughput: {len(patients)} patients in {dt:.1f} s "
          f"({len(patients) / dt:.0f} patients/s, {dt / len(patients) * 1000:.1f} ms each, "
          f"incl. 20 bootstrap models + XGBoost)")
    assert dt / len(patients) < 0.5
