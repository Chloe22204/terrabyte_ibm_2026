"""
Turns a TimeToTalk score into a one-page conversation prep sheet for the doctor:
  score (ml/scoring.py) -> retrieve guideline passages (retriever.py) -> prompt (prompt.py) -> Granite (granite.py)

If Granite is unreachable, a template sheet with the same headings is returned instead, so the demo never breaks.

CLI (run from the repo root after train.py has produced artifacts/):
  python genai/prep_sheet.py --patient genai/example_patient.json
  python genai/prep_sheet.py --patient genai/example_patient.json --language Chinese
  python genai/prep_sheet.py --patient genai/example_patient.json --why "The family wants everything done. How do I respond?"
  python genai/prep_sheet.py --patient genai/example_patient.json --no-llm      # template only
"""
import argparse, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prompt import build_coach_prompt, build_prompt, factor_text, friendly  # noqa: E402
from retriever import retrieve  # noqa: E402
from granite import chat, GraniteUnavailable  # noqa: E402

# Whitelist: the language string goes into the prompt, so only these values are allowed.
LANGUAGES = {"English": "English", "Chinese": "Simplified Chinese", "Malay": "Bahasa Melayu", "Tamil": "Tamil"}
MAX_TOKENS = 300     # Granite writes ~4.7 tokens/s on our 3 vCPUs -> at most ~65 s of output
PASSAGE_SENTENCES = 2  # Granite reads ~15 tokens/s, so each passage is cut to its first 2 sentences

# One retrieval per prep-sheet heading, searched only within that heading's note, so every heading
# gets guidance (a broad search let documenting.md win the "questions" query). TF-IDF picks the best
# passage inside the note for this patient.
HEADING_QUERIES = [  # (query, how many passages, notes to search)
    ("prepare before the conversation who should be present family interpreter setting", 1, ["setting_up.md"]),
    ("open questions understanding illness goals fears what matters most", 1, ["questions_to_explore.md"]),
    ("phrases words to avoid say instead", 1, ["language_to_avoid.md"]),
    ("document record care plan advance care planning next step review", 1, ["documenting.md"]),
]
WHY_NOTES = ["when_to_start.md", "uncertainty.md"]
COACH_PASSAGES = 3       # 'Ask the guidelines': passages searched across all notes
COACH_MIN_SCORE = 0.05   # below this TF-IDF similarity a passage shares no real words with the question


def _norm(patient):
    return {k.lower().replace("_", "."): v for k, v in patient.items()}


def _num(p, key):
    try:
        return float(p.get(key))
    except (TypeError, ValueError):
        return None


def summarize(patient: dict) -> str:
    """De-identified summary for the prompt: age band instead of exact age, no IDs or names."""
    p = _norm(patient)
    parts = []
    age = _num(p, "age")
    if age is not None:
        parts.append(f"in their {int(age // 10) * 10}s")
    if p.get("sex"):
        parts.append(str(p["sex"]))
    if p.get("dzgroup"):
        parts.append(f"diagnosis group: {p['dzgroup']}")
    if str(p.get("ca", "")).lower() == "metastatic":
        parts.append("cancer that has spread")
    n_co = _num(p, "num.co")
    if n_co:
        parts.append(f"{int(n_co)} other illnesses")
    flags = []
    checks = [("alb", lambda v: v < 3.0), ("crea", lambda v: v > 2.0), ("bili", lambda v: v > 2.0),
              ("adlsc", lambda v: v >= 3), ("scoma", lambda v: v > 0), ("dementia", lambda v: v >= 1)]
    for key, bad in checks:
        v = _num(p, key)
        if v is not None and bad(v):
            flags.append(friendly(key))
    if flags:
        parts.append("notable: " + ", ".join(flags))
    return "Patient " + "; ".join(parts) + "." if parts else "No summary available."


def gather_passages(score, max_chunks=5):
    """Top 2 for 'why now' (based on the patient's factors), then the best chunk per remaining heading."""
    factors = " ".join(friendly(f) for f in score["top_factors"][:3])   # uses the readable label
    why_query = f"{score['decision']} when to start conversation {factors}"
    if score["decision"] == "Doctor judgement" or score.get("models_disagree"):
        why_query += " uncertainty prognosis range"
    picked, seen = [], set()
    for query, k, files in [(why_query, 1, WHY_NOTES)] + HEADING_QUERIES:
        for c in retrieve(query, k=k + 2, files=files):
            key = (c["file"], c["title"])
            if key not in seen:
                picked.append(c)
                seen.add(key)
                k -= 1
            if k == 0:
                break
    return picked[:max_chunks]


def shorten(text, n=PASSAGE_SENTENCES):
    return " ".join(re.split(r"(?<=[.!?])\s+", text)[:n])


PROGNOSIS = re.compile(r"\b(will die|going to die|(days?|weeks?|months?|years?) (left )?to live|"
                       r"life expectancy (of|is))\b", re.I)


def check_output(text, score):
    """Cheap guardrails on Granite's output. Returns warnings for the doctor, never edits the text."""
    warnings = []
    if PROGNOSIS.search(text):
        warnings.append("Output contains a time-to-death statement; the model should not give one. Review before use.")
    allowed = [round(100 * v) for v in [score["risk_6m"], *score["risk_6m_interval_80"]]] + [80]  # "80% range"
    for m in re.finditer(r"(\d{1,3}(?:\.\d+)?)\s?%", text):
        if all(abs(float(m.group(1)) - a) > 1 for a in allowed):
            warnings.append(f"Output mentions {m.group(0)}, which is not one of the model's numbers.")
    return warnings


def fallback_sheet(score, passages, patient=None):
    """Same five headings, filled from the model output and retrieved guidance, no LLM."""
    by_file = {}
    for c in passages:
        by_file.setdefault(c["file"], []).append(c["text"])

    def from_files(*files):
        texts = [t for f in files for t in by_file.get(f, [])]
        return texts[0] if texts else "(no guidance retrieved)"

    lo, hi = score["risk_6m_interval_80"]
    factors = ", ".join(factor_text(f, patient) for f in score["top_factors"][:3])
    lines = []
    if score["decision"] == "Doctor judgement":
        lines.append("NOTE: the model is uncertain for this patient. Your own clinical assessment should lead.\n")
    lines += [
        f"1. Why now\nRecommendation: {score['decision']}. Estimated 6-month mortality risk "
        f"{score['risk_6m']:.0%} (80% range {lo:.0%}-{hi:.0%}). Main factors: {factors}.",
        f"2. Before you go in\n{from_files('setting_up.md')}",
        f"3. Questions to explore\n{from_files('questions_to_explore.md')}",
        f"4. Phrases to avoid\n{from_files('language_to_avoid.md')}",
        f"5. Next step to record in the care plan\n{from_files('documenting.md')}",
        "\n(Template version: Granite was not available.)",
    ]
    return "\n\n".join(lines)


def make_prep_sheet(score, patient, language="English", use_llm=True):
    if language not in LANGUAGES:
        raise ValueError(f"language must be one of {list(LANGUAGES)}")
    t0 = time.perf_counter()
    passages = gather_passages(score)
    warnings, used_llm = [], False
    if use_llm:
        messages = build_prompt(score, summarize(patient),
                                [f"{shorten(c['text'])} [{c['file']}]" for c in passages], LANGUAGES[language],
                                patient=patient)
        try:
            text = chat(messages, max_tokens=MAX_TOKENS)
            used_llm = True
            warnings += check_output(text, score)
        except GraniteUnavailable as e:
            warnings.append(f"Granite unavailable, template used ({e})")
    if not used_llm:
        text = fallback_sheet(score, passages, patient)
        if language != "English":
            warnings.append("Template prep sheet is English only.")
    return {"text": text, "language": language, "used_llm": used_llm,
            "latency_s": round(time.perf_counter() - t0, 2), "warnings": warnings,
            "sources": sorted({c["source"] for c in passages}),
            "passages": [{"file": c["file"], "title": c["title"]} for c in passages]}


def ask_why(score, question, patient=None, use_llm=True):
    """'Ask the guidelines' conversation coach: answers a doctor's communication question from the
    guideline notes (retrieval over all notes + Granite). Off-topic questions get no Granite call."""
    question = question.strip()[:500]
    passages = [c for c in retrieve(question, k=COACH_PASSAGES) if c["score"] >= COACH_MIN_SCORE]
    out = {"used_llm": False, "warnings": [], "sources": sorted({c["source"] for c in passages}),
           "passages": [{"file": c["file"], "title": c["title"]} for c in passages]}
    if not passages:
        return dict(out, answer="The guideline notes don't cover this question. They cover when to start, setting "
                                "up, questions to explore, phrases to avoid, uncertainty and documenting. For why "
                                "this patient's risk is what it is, see Main factors above.")
    if use_llm:
        messages = build_coach_prompt(score, question,
                                      [f"{shorten(c['text'], 3)} [{c['file']}]" for c in passages],
                                      summarize(patient or {}))
        try:
            text = chat(messages, max_tokens=200)
            return dict(out, answer=text, used_llm=True, warnings=check_output(text, score))
        except GraniteUnavailable:
            pass
    return dict(out, answer="Granite is not available, so here is the most relevant guidance:\n\n"
                + "\n\n".join(f"{c['title']} [{c['file']}]\n{c['text']}" for c in passages))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default=os.path.join(HERE, "example_patient.json"))
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--language", default="English", choices=list(LANGUAGES))
    ap.add_argument("--why", help="ask the guidelines coach a question instead of making a prep sheet")
    ap.add_argument("--no-llm", action="store_true", help="skip Granite, use the template")
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(HERE, "..", "ml"))
    from scoring import TimeToTalkScorer
    patient = json.load(open(a.patient))
    score = TimeToTalkScorer(a.artifacts).score(patient)
    lo, hi = score["risk_6m_interval_80"]
    print(f"Decision: {score['decision']} | 6-month risk {score['risk_6m']:.0%} (80% range {lo:.0%}-{hi:.0%})\n")

    if a.why:
        out = ask_why(score, a.why, patient, use_llm=not a.no_llm)
        print(out["answer"])
    else:
        out = make_prep_sheet(score, patient, a.language, use_llm=not a.no_llm)
        print(out["text"])
        print(f"\n--- Granite used: {out['used_llm']} | {out['latency_s']} s")
        print("Guidance used:", ", ".join(f"{p['file']} > {p['title']}" for p in out["passages"]))
    for w in out["warnings"]:
        print("WARNING:", w)
