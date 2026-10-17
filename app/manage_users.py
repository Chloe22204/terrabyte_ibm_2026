"""
Add, remove or list dashboard users. Passwords are typed at a hidden prompt and stored only as
scrypt hashes in app/users.json (git-ignored).

  python app/manage_users.py add dr_tan --role clinician
  python app/manage_users.py add audit1 --role auditor
  python app/manage_users.py disable dr_tan
  python app/manage_users.py remove dr_tan
  python app/manage_users.py list
"""
import argparse, getpass, os, re, sys

from security import ROLES, UserStore, hash_password

DEFAULT_FILE = os.environ.get("TTT_USERS_FILE", os.path.join(os.path.dirname(__file__), "users.json"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["add", "remove", "disable", "list"])
    p.add_argument("username", nargs="?")
    p.add_argument("--role", choices=ROLES, default="clinician")
    p.add_argument("--file", default=DEFAULT_FILE)
    a = p.parse_args()
    store = UserStore(a.file)
    users = store.load()

    if a.action == "list":
        for name, u in sorted(users.items()):
            print(f"{name:20s} {u['role']:10s} {'DISABLED' if u.get('disabled') else ''}")
        return
    if not a.username or not re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", a.username):
        sys.exit("username: 3-32 characters, letters/digits/_ . - only")

    if a.action == "add":
        pw = getpass.getpass("Password (min 12 characters): ")
        if len(pw) < 12:
            sys.exit("Password too short (min 12 characters).")
        if pw != getpass.getpass("Repeat password: "):
            sys.exit("Passwords don't match.")
        users[a.username] = {"hash": hash_password(pw), "role": a.role}
        print(f"Saved {a.username} ({a.role}).")
    elif a.action == "disable":
        if a.username not in users:
            sys.exit("No such user.")
        users[a.username]["disabled"] = True
        print(f"Disabled {a.username}.")
    elif a.action == "remove":
        if users.pop(a.username, None) is None:
            sys.exit("No such user.")
        print(f"Removed {a.username}.")
    store.save(users)


if __name__ == "__main__":
    main()
