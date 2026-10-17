#!/usr/bin/env bash
# Start (or confirm) Granite + the dashboard on the LinuxONE VM. Safe to run again: anything
# already running is left alone. Everything is detached, so it keeps running after you log out.
#
# From your laptop (one command: starts everything and opens the tunnel):
#   ssh -t -L 8503:localhost:8503 linux1@<vm-ip> "bash ~/terrabyte_ibm_2026/scripts/start_vm.sh; exec bash"
# then open http://localhost:8503
#
# Usage on the VM:  bash scripts/start_vm.sh [dashboard-port]     (default 8503)
PORT="${1:-8503}"
REPO="$HOME/terrabyte_ibm_2026"
cd "$REPO" || { echo "repo not found at $REPO"; exit 1; }
source "$HOME/ttt-venv/bin/activate" || { echo "venv missing: see docs/SETUP.md step 1b"; exit 1; }

# 1. Granite (llama-server on 127.0.0.1:8080)
if pgrep -x llama-server >/dev/null; then
  echo "Granite: already running"
else
  echo "Granite: starting (model load takes ~30 s)"
  setsid nohup "$HOME/llama.cpp/build/bin/llama-server" -m "$HOME/models/granite-be.gguf" \
    -t 3 -c 2048 --host 127.0.0.1 --port 8080 > "$HOME/llama.log" 2>&1 < /dev/null &
fi

# 2. Dashboard (uvicorn on 127.0.0.1:$PORT). Same salt every restart, so patient pseudonyms stay stable.
[ -f "$HOME/.ttt_salt" ] || { openssl rand -hex 16 > "$HOME/.ttt_salt"; chmod 600 "$HOME/.ttt_salt"; }
if pgrep -f "uvicorn app.dashboard:app.*--port $PORT" >/dev/null; then
  echo "Dashboard: already running on port $PORT"
elif ss -ltn | grep -q ":$PORT "; then
  echo "Dashboard: port $PORT is used by something else. Rerun with another port: bash scripts/start_vm.sh 8504"
  exit 1
else
  echo "Dashboard: starting on port $PORT"
  TTT_ID_SALT="$(cat "$HOME/.ttt_salt")" TTT_COOKIE_SECURE=0 \
    setsid nohup uvicorn app.dashboard:app --host 127.0.0.1 --port "$PORT" > "$HOME/dashboard.log" 2>&1 < /dev/null &
fi

# 3. Wait for both, then report
for i in $(seq 1 24); do
  curl -s localhost:8080/health | grep -q '"ok"' && break
  sleep 5
done
echo
echo "Granite health:   $(curl -s localhost:8080/health || echo 'not responding: tail ~/llama.log')"
echo "Dashboard:        $(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/" | sed 's/^000$/not responding: tail ~\/dashboard.log/;s/^[23]..$/up/')"
echo "Users:            $(python3 app/manage_users.py list | awk '{print $1}' | paste -sd' ' -)"
echo
echo "Open http://localhost:$PORT on your laptop (through the ssh -L tunnel)."
echo "No user yet? python3 app/manage_users.py add dr_tan"
