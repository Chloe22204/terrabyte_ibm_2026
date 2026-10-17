"""Unit tests for app/security.py: the building blocks, tested without the web app."""
import json, os, re, threading

import pytest

import security
from security import AuditLog, LoginGuard, SessionStore, UserStore, hash_password, pseudonym, same, verify_password


# ---------------------------------------------------------------- passwords
def test_password_roundtrip():
    h = hash_password("correct horse battery")
    assert verify_password("correct horse battery", h)
    assert not verify_password("correct horse batterY", h)
    assert not verify_password("", h)


def test_hash_format_and_unique_salt():
    a, b = hash_password("same password"), hash_password("same password")
    assert a != b                                       # random salt every time
    assert a.startswith("scrypt$16384$8$1$")
    assert "same password" not in a


@pytest.mark.parametrize("bad", ["", "plain", "scrypt$1$2", "scrypt$x$y$z$zz$zz", "md5$abc", None])
def test_malformed_hash_never_verifies(bad):
    assert verify_password("anything", bad) is False


def test_unicode_and_long_passwords():
    pw = "pässwörd-密码-" + "x" * 200
    assert verify_password(pw, hash_password(pw))


def test_same_is_constant_time_and_rejects_empty():
    assert same("abc", "abc")
    assert not same("abc", "abd")
    assert not same("", "")                             # an empty token must never match


# ---------------------------------------------------------------- users
def test_userstore_authenticate(tmp_path):
    store = UserStore(tmp_path / "u.json")
    store.save({"a": {"hash": hash_password("pw-aaaaaaaaaaaa"), "role": "clinician"},
                "d": {"hash": hash_password("pw-dddddddddddd"), "role": "clinician", "disabled": True}})
    assert store.authenticate("a", "pw-aaaaaaaaaaaa") == "clinician"
    assert store.authenticate("a", "wrong") is None
    assert store.authenticate("nobody", "pw-aaaaaaaaaaaa") is None
    assert store.authenticate("d", "pw-dddddddddddd") is None     # disabled account


def test_userstore_missing_file_means_no_users(tmp_path):
    assert UserStore(tmp_path / "none.json").authenticate("a", "b") is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX file permissions")
def test_users_file_is_owner_only(tmp_path):
    p = tmp_path / "u.json"
    UserStore(p).save({})
    assert oct(os.stat(p).st_mode & 0o777) == "0o600"


# ---------------------------------------------------------------- lockout
def test_guard_locks_user_after_5_failures():
    g = LoginGuard()
    for _ in range(4):
        g.failed("bob", "1.1.1.1")
    assert not g.is_locked("bob", "1.1.1.1")
    g.failed("bob", "1.1.1.1")
    assert g.is_locked("bob", "1.1.1.1")
    assert g.is_locked("bob", "9.9.9.9")                # locked whichever IP tries
    assert not g.is_locked("alice", "9.9.9.9")


def test_guard_locks_ip_after_20_failures_across_users():
    g = LoginGuard()
    for i in range(20):
        g.failed(f"user{i}", "6.6.6.6")
    assert g.is_locked("brand_new_user", "6.6.6.6")
    assert not g.is_locked("brand_new_user", "7.7.7.7")


def test_guard_success_clears_user_counter():
    g = LoginGuard()
    for _ in range(4):
        g.failed("bob", "ip")
    g.succeeded("bob")
    g.failed("bob", "ip")
    assert not g.is_locked("bob", "ip2")


def test_guard_window_expires(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(security.time, "time", lambda: now[0])
    g = LoginGuard(window_s=900)
    for _ in range(5):
        g.failed("bob", "ip")
    assert g.is_locked("bob", "ip")
    now[0] += 901
    assert not g.is_locked("bob", "ip")


# ---------------------------------------------------------------- sessions
def test_session_tokens_are_long_and_unique():
    s = SessionStore()
    tokens = {s.create("u", "clinician") for _ in range(2000)}
    assert len(tokens) == 2000
    assert all(len(t) >= 43 for t in tokens)           # 32 random bytes, base64url


def test_session_holds_role_and_csrf_server_side():
    s = SessionStore()
    t = s.create("u", "auditor")
    sess = s.get(t)
    assert sess["role"] == "auditor" and len(sess["csrf"]) >= 43
    assert sess["csrf"] != t


def test_session_idle_and_absolute_timeouts(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(security.time, "time", lambda: now[0])
    s = SessionStore(idle_s=60, absolute_s=300)
    t = s.create("u", "clinician")
    now[0] = 59
    assert s.get(t)                                     # activity resets the idle clock
    now[0] = 118
    assert s.get(t)
    now[0] = 179                                        # 61 s idle
    assert s.get(t) is None
    t2 = s.create("u", "clinician")
    for step in range(1, 8):                            # active every 50 s, but past 300 s total
        now[0] = 179 + 50 * step
        alive = s.get(t2)
    assert alive is None


def test_session_end_and_unknown_tokens():
    s = SessionStore()
    t = s.create("u", "clinician")
    s.end(t)
    assert s.get(t) is None
    assert s.get(None) is None and s.get("") is None and s.get("forged") is None


# ---------------------------------------------------------------- pseudonyms
def test_pseudonym_properties():
    a = pseudonym("salt-one-xxxxxxxx", 123)
    assert re.fullmatch(r"[0-9a-f]{12}", a)
    assert a == pseudonym("salt-one-xxxxxxxx", "123")  # stable for the same salt
    assert a != pseudonym("salt-two-xxxxxxxx", 123)    # different salt, different pseudonym
    assert "123" not in a


def test_pseudonyms_do_not_collide_on_dataset_sized_ids():
    refs = {pseudonym("salt-xxxxxxxxxxxxxxxx", i) for i in range(1, 9106)}
    assert len(refs) == 9105


# ---------------------------------------------------------------- audit log
def test_audit_log_writes_json_lines(tmp_path):
    log = AuditLog(str(tmp_path / "sub" / "a.log"))   # creates the folder
    log.write("login", user="u", ip="1.2.3.4")
    log.write("view_patient", user="u", patient="abc123abc123")
    entries = log.tail()
    assert [e["event"] for e in entries] == ["view_patient", "login"]   # newest first
    assert all("ts" in e for e in entries)


def test_audit_log_survives_hostile_values(tmp_path):
    log = AuditLog(str(tmp_path / "a.log"))
    log.write("login_failed", user='evil"}\n{"event":"login","user":"admin')
    lines = open(tmp_path / "a.log", encoding="utf-8").read().splitlines()
    assert len(lines) == 1                              # no log injection via newlines or quotes
    assert json.loads(lines[0])["event"] == "login_failed"


def test_audit_log_concurrent_writes_stay_intact(tmp_path):
    log = AuditLog(str(tmp_path / "a.log"))

    def worker(n):
        for i in range(100):
            log.write("view_patient", user=f"u{n}", patient=f"{i:012x}")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = open(tmp_path / "a.log", encoding="utf-8").read().splitlines()
    assert len(lines) == 2000
    assert all(json.loads(l)["event"] == "view_patient" for l in lines)
