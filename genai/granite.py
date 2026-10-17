"""
Talks to IBM Granite running locally on the LinuxONE VM. No patient data leaves the machine.

Two ways to reach Granite (pick with TTT_LLM_BACKEND):
  server  (default)  llama.cpp's llama-server on this VM, OpenAI-style chat API.
                     Start it with:  llama-server -m ~/models/granite-be.gguf -t 3 -c 2048 --host 127.0.0.1 --port 8080
  python             llama-cpp-python inside this process (if that's how the IBM lab loads Granite)

Env vars:
  TTT_LLM_URL       default http://127.0.0.1:8080/v1/chat/completions
  TTT_GRANITE_GGUF  model path for the python backend, default ~/models/granite-be.gguf
  TTT_LLM_TIMEOUT   seconds, default 300 (measured on our 3-vCPU z17: reads ~15 tokens/s, writes ~4.7 tokens/s,
                    so a prep sheet takes ~1.5-2 min)

Only the standard library is needed for the server backend, so the API container stays small.
"""
import json, os, urllib.error, urllib.request

BACKEND = os.environ.get("TTT_LLM_BACKEND", "server")
URL = os.environ.get("TTT_LLM_URL", "http://127.0.0.1:8080/v1/chat/completions")
GGUF = os.environ.get("TTT_GRANITE_GGUF", os.path.expanduser("~/models/granite-be.gguf"))
TIMEOUT = float(os.environ.get("TTT_LLM_TIMEOUT", "300"))

_llama = None


class GraniteUnavailable(RuntimeError):
    """Granite could not be reached; callers fall back to the template prep sheet."""


def _server_chat(messages, max_tokens, temperature):
    body = json.dumps({"messages": messages, "max_tokens": max_tokens,
                       "temperature": temperature}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.load(r)["choices"][0]["message"]["content"].strip()
    except (OSError, ValueError, KeyError, IndexError) as e:   # URLError/timeouts are OSError
        raise GraniteUnavailable(f"llama-server not reachable at {URL}: {e}") from e


def _python_chat(messages, max_tokens, temperature):
    global _llama
    try:
        if _llama is None:
            from llama_cpp import Llama
            _llama = Llama(model_path=GGUF, n_ctx=2048, n_threads=os.cpu_count() or 3, verbose=False)
        out = _llama.create_chat_completion(messages=messages, max_tokens=max_tokens,
                                            temperature=temperature)
        return out["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise GraniteUnavailable(f"llama-cpp-python failed with {GGUF}: {e}") from e


def chat(messages, max_tokens=350, temperature=0.2):
    """messages = [{"role": "system"|"user", "content": ...}], as built by genai/prompt.py."""
    if BACKEND == "python":
        return _python_chat(messages, max_tokens, temperature)
    return _server_chat(messages, max_tokens, temperature)


def health():
    """For /health and the dashboard: is Granite up?"""
    if BACKEND == "python":
        return {"backend": "python", "model": GGUF, "ok": os.path.exists(GGUF)}
    base = URL.split("/v1/")[0]
    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as r:
            return {"backend": "server", "url": base, "ok": json.load(r).get("status") == "ok"}
    except (OSError, ValueError):
        return {"backend": "server", "url": base, "ok": False}
