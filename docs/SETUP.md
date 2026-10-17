# TimeToTalk: setup and run guide (LinuxONE)

Everything runs on one LinuxONE VM. Work through the steps in order. Each step ends with a **✅ check**: don't move on until it passes.

| Step | What | Time | Can run in parallel with |
|---|---|---|---|
| 0 | Check the VM | 2 min | – |
| 1 | Get the code and data | 5 min | – |
| 2 | Train the models | 2 min | 3 (after 3a has finished building) |
| 3 | Get Granite running | 45–60 min | 2 |
| 4 | Test the prep sheet (genai) | 5 min | – |
| 5 | Run the API in a container | 10 min | – |
| 6 | Dashboard | – | – |
| 7 | Capture evidence for judging | 15 min | – |

Our VM: s390x, 3 vCPUs, 7.8 GB RAM (about 4.9 GB free), **no swap**, 35 GB free disk.
With no swap, running out of memory kills a process instead of slowing it down. **Never build llama.cpp and train at the same time.**

---

## 0. Check the VM
```bash
uname -m          # s390x
nproc             # 3
free -h           # "available" should be at least 4 GB
df -h ~           # at least 10 GB free
podman ps         # is the IBM lab's Jupyter container running? (it uses memory)
```
✅ `s390x` and at least 4 GB available.

## 1. Get the code, Python packages and data

**1a. Code.** Clone the first time; after that, just pull.
```bash
cd ~
[ -d terrabyte_ibm_2026 ] && git -C terrabyte_ibm_2026 pull || git clone https://github.com/Chloe22204/terrabyte_ibm_2026
cd ~/terrabyte_ibm_2026
```

**1b. Python packages.** The VM runs Ubuntu with Python 3.12, which blocks `pip install` into the system Python (`externally-managed-environment`). So:
- install the heavy science packages with **apt**, because they're prebuilt for s390x and need no compiling
- create a **virtual environment** ("venv") that can see them, for everything else
```bash
sudo apt update
sudo apt install -y python3-venv python3-full python3-numpy python3-scipy python3-pandas \
                    python3-sklearn python3-matplotlib
python3 -m venv --system-site-packages ~/ttt-venv
source ~/ttt-venv/bin/activate
pip install fastapi uvicorn huggingface_hub gguf
pip install xgboost || echo "xgboost failed: fine, the second opinion is skipped"
```
**Every new terminal:** run `source ~/ttt-venv/bin/activate` first. Your prompt then starts with `(ttt-venv)`. All later steps assume this.

✅ `python3 -c "import numpy, scipy, pandas, sklearn, matplotlib, fastapi; print('ok')"` prints `ok`.

**1c. Data.** Download the CSV from https://archive.ics.uci.edu/dataset/880 and save it as `data/support2.csv`. From a laptop:
`scp support2.csv <user>@<vm-ip>:~/terrabyte_ibm_2026/data/`

✅ `ls data/support2.csv` shows the file.

## 2. Train the models
```bash
python3 ml/train.py --data data/support2.csv --out artifacts --bootstraps 20
```
This rebuilds `artifacts/`, **including `artifacts/bootstraps/`**, which isn't in git. Without it the uncertainty range collapses to a single number and the "Clinician judgement" tier never triggers.

✅ `ls artifacts/bootstraps | wc -l` shows 20, and `artifacts/metrics.json` exists.

Optional, for the AI accelerator: `python3 ml/accelerator_model.py --data data/support2.csv --out artifacts`

## 3. Get Granite running
Granite is IBM's open LLM. We run it with **llama.cpp**, and it never leaves the VM.

**3a. Build llama.cpp** (15–20 min). Use `-j2`, not `-j3`, because there's no swap.
```bash
# RHEL:   sudo dnf install -y git cmake gcc-c++ make python3-pip
# Ubuntu: sudo apt update && sudo apt install -y git cmake g++ make python3-pip
cd ~ && git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j2 --target llama-server llama-cli
```
✅ `ls ~/llama.cpp/build/bin/` lists `llama-server` and `llama-cli`.

**3b. Download Granite 3.3 2B** (5–10 min, about 2.7 GB)
```bash
source ~/ttt-venv/bin/activate          # huggingface_hub and gguf were installed here in step 1b
mkdir -p ~/models
hf download ibm-granite/granite-3.3-2b-instruct-GGUF --local-dir ~/models --include "*Q8_0*"
#   older huggingface_hub: use "huggingface-cli download" with the same arguments
ls -lh ~/models
```
If the file name differs, pick the **Q8_0** file from https://huggingface.co/ibm-granite/granite-3.3-2b-instruct-GGUF.

**3c. Convert the model to big-endian.** LinuxONE stores numbers in the opposite byte order to laptops, so the model file has to be converted.
```bash
cd ~/models
cp granite-3.3-2b-instruct-Q8_0.gguf granite-be.gguf       # the script edits the file in place
python3 ~/llama.cpp/gguf-py/gguf/scripts/gguf_convert_endian.py granite-be.gguf big
```

**3d. Quick test**
```bash
~/llama.cpp/build/bin/llama-cli -m ~/models/granite-be.gguf -p "In one sentence, what is advance care planning?" -n 60 -no-cnv
```
✅ A sensible sentence. An error mentioning "magic", "version" or "tensor type", or garbage text, means the byte order is wrong: redo 3c.

**3e. Run Granite as a local server** (leave it running)
```bash
nohup ~/llama.cpp/build/bin/llama-server -m ~/models/granite-be.gguf \
  -t 3 -c 2048 --host 127.0.0.1 --port 8080 > ~/llama.log 2>&1 &
sleep 20 && curl -s localhost:8080/health
```
- `--host 127.0.0.1` means only programs on this VM can reach Granite. **Don't open port 8080 in a firewall.**
- `-c 2048` keeps memory use down, and our prompts fit.

✅ `{"status":"ok"}`

## 4. Test the prep sheet (genai)
From the repo folder:
```bash
cd ~/terrabyte_ibm_2026
python3 genai/prep_sheet.py --no-llm                      # template only: checks scoring + retrieval
python3 genai/prep_sheet.py                               # with Granite: takes ~30-90 s
python3 genai/prep_sheet.py --language Chinese
python3 genai/prep_sheet.py --language Malay
python3 genai/prep_sheet.py --language Tamil
python3 genai/prep_sheet.py --why "Why is this patient flagged now?"
grep "tokens per second" ~/llama.log | tail -3            # speed, for the video
```
✅ `Granite used: True`, five headings, numbers matching the "Decision" line, and **no WARNING lines**.

Have a teammate who reads each language check that output. **Only claim languages that read correctly**, and update the README to match.

How it works: the score feeds `genai/retriever.py`, which finds matching passages in `genai/guidelines/*.md` (RAG). Then `genai/prompt.py` builds the prompt, and `genai/granite.py` sends it to llama-server. `genai/prep_sheet.py` checks the output for time-to-death phrases and for numbers the model didn't produce. If Granite is down, it falls back to a template so the demo never breaks.

## 5. Run the API in a container
```bash
cd ~/terrabyte_ibm_2026
podman build -t timetotalk-api -f api/Containerfile .
export TTT_API_TOKEN=$(openssl rand -hex 24)
podman run -d --name ttt --network host \
  -e TTT_API_TOKEN=$TTT_API_TOKEN -e TTT_ID_SALT=$(openssl rand -hex 16) timetotalk-api
```
`--network host` lets the container reach Granite at 127.0.0.1:8080.

```bash
curl -s localhost:8000/health                       # shows machine s390x and granite ok: true
curl -s -X POST localhost:8000/prep_sheet -H "Authorization: Bearer $TTT_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"patient_id\":\"demo-1\",\"language\":\"English\",\"patient\":$(cat genai/example_patient.json)}"
curl -s -X POST localhost:8000/prep_sheet -d '{}' -H "Content-Type: application/json"   # 401: no token
curl -s -X POST "localhost:8000/benchmark?n=1000" -H "Authorization: Bearer $TTT_API_TOKEN" \
  -H "Content-Type: application/json" -d "{\"patient_id\":\"b\",\"patient\":$(cat genai/example_patient.json)}"
podman exec ttt cat /app/logs/audit.log             # hashed IDs + metadata only, no clinical text
podman exec ttt whoami                              # appuser (not root)
```
✅ `/health` shows `"machine": "s390x"` and `"granite": {... "ok": true}`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | status, platform, Granite up/down |
| `POST /score` | risk, uncertainty, decision, top factors |
| `POST /prep_sheet` | score + Granite prep sheet (`language`: English / Chinese / Malay / Tamil) |
| `POST /ask_why` | short answer to a clinician's question (`question`, up to 500 characters) |
| `POST /benchmark?n=1000` | scoring speed |

## 6. Dashboard
`app/` (Streamlit) hasn't been built yet. When it is, open it from your laptop through an SSH tunnel so it isn't exposed publicly:
`ssh -L 8501:localhost:8501 <user>@<vm-ip>`, then browse to http://localhost:8501.

## 7. Evidence for judging (rubric: "use screenshots as evidence")
Take these screenshots for the video and README:
- [ ] `uname -m` → s390x, with `nproc` and `free -h`
- [ ] `train.py` finishing on the VM, with its runtime
- [ ] llama-server running (`curl :8080/health`) and the tokens-per-second line
- [ ] a Granite prep sheet in the terminal (English plus one other language)
- [ ] `podman ps` showing the container, `podman exec ttt whoami` → appuser
- [ ] the 401 response without a token
- [ ] the audit log showing hashed IDs only
- [ ] `/benchmark` patients-per-second
- [ ] the accelerator benchmark (`artifacts/accelerator_benchmark.json`), if step 2's optional command was run

## Troubleshooting
| Problem | Fix |
|---|---|
| `externally-managed-environment` from pip | You're outside the venv: `source ~/ttt-venv/bin/activate` (step 1b) |
| `ModuleNotFoundError` when running a script | Same: activate the venv first |
| Build killed or "out of memory" | Stop other jobs (`podman stop` the Jupyter container if not needed) and build with `-j1` |
| `hf` / `huggingface-cli: command not found` | Activate the venv; if it still fails, download the file on a laptop and `scp` it to `~/models/` |
| Endian script errors | `python3 .../gguf_convert_endian.py --help` to check the argument spelling; use the Q8_0 file |
| Prep sheet says "template used" | Granite isn't running: `curl localhost:8080/health`, check `~/llama.log` |
| Prep sheet is slow (>2 min) | Normal-ish on 3 vCPUs. Lower `MAX_TOKENS` in `genai/prep_sheet.py`, or set `TTT_LLM_TIMEOUT=300` |
| Container can't reach Granite | Make sure you used `--network host` |
| xgboost won't install in the container | Remove it from `api/requirements.txt`; the API runs without the second opinion |
| Stuck for >30 min | Ask in Discord **#ibm z help**, or reuse the IBM GenAI lab's Granite setup (set `TTT_LLM_URL` to wherever it serves) |
