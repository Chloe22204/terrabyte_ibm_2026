"""
Prompt builder for IBM Granite (run locally on LinuxONE via llama.cpp, same way as the
IBM "Predictive AI + Generative AI" loan lab).

Plug-in point: reuse the lab's Granite loading code and its RAG retriever, then call
    prompt = build_prompt(score_result, patient_summary, retrieved_passages, language)
and send `prompt` to Granite. Keep max_tokens around 350 so it finishes quickly on 3 vCPUs.

Guardrails baked into the prompt:
  - output is for CLINICIANS only, never shown to patients
  - Granite must not change or invent the risk numbers or the decision; it only explains them
  - no prognosis statements like "will die in X months"
"""

SYSTEM = """You are a clinical communication assistant inside a hospital. You help doctors and nurses \
prepare for an advance care planning conversation with a seriously ill patient and their family.
Rules:
- Your reader is a clinician, never the patient.
- Use ONLY the numbers and decision given to you. Never change them or invent new ones.
- Never state how long the patient will live. Talk about uncertainty and priorities instead.
- Base suggestions on the guideline passages provided. If they do not cover something, say so.
- Be warm, plain and brief."""

TEMPLATE = """PATIENT SUMMARY (de-identified)
{summary}

MODEL OUTPUT (from the TimeToTalk survival model; do not alter)
- Recommendation: {decision}
- Estimated 6-month mortality risk: {risk:.0%} (80% range {lo:.0%}-{hi:.0%})
- Second model agrees: {agree}
- Main factors: {factors}

GUIDELINE PASSAGES (retrieved)
{passages}

TASK
Write a one-page conversation prep sheet in {language} with these headings:
1. Why now (2-3 sentences, refer to the main factors in plain words)
2. Before you go in (who should be present, what to check in the notes)
3. Questions to explore (4-5 open questions about understanding, goals, fears, what matters most)
4. Phrases to avoid
5. Next step to record in the care plan
If the recommendation is "Clinician judgement", start by saying the model is uncertain and the \
clinician's own assessment should lead."""

FRIENDLY = {
    "age": "age", "num.co": "number of other illnesses", "scoma": "level of consciousness",
    "alb": "low albumin (nutrition)", "bili": "bilirubin (liver)", "crea": "creatinine (kidneys)",
    "bun": "urea (kidneys)", "adlsc": "difficulty with daily activities", "aps": "overall physiology score",
    "sps": "overall physiology score", "meanbp": "blood pressure", "pafi": "oxygen levels",
    "hday": "days in hospital before assessment", "ca_metastatic": "cancer that has spread",
    "dementia": "dementia", "wblc": "white cell count", "resp": "breathing rate", "hrt": "heart rate",
}


def friendly(feature):
    if feature in FRIENDLY:
        return FRIENDLY[feature]
    if feature.startswith("dzgroup_"):
        return "diagnosis: " + feature.split("_", 1)[1]
    return feature.replace("_missing", " (not measured)")


def build_prompt(score, summary, passages, language="English"):
    factors = ", ".join(f"{friendly(f['feature'])} ({f['effect']})" for f in score["top_factors"][:3])
    agree = ("not available" if score.get("second_opinion_risk_6m") is None
             else ("no - models disagree" if score["models_disagree"] else "yes"))
    lo, hi = score["risk_6m_interval_80"]
    body = TEMPLATE.format(summary=summary, decision=score["decision"], risk=score["risk_6m"],
                           lo=lo, hi=hi, agree=agree, factors=factors,
                           passages="\n".join(f"- {p}" for p in passages) or "- (none retrieved)",
                           language=language)
    # Granite chat format; if the lab uses llama.cpp's chat API, pass SYSTEM and body as messages instead
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


def build_why_prompt(score, question):
    """For the 'Ask why' box in the app."""
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Model output: {score['decision']}, 6-month risk "
             f"{score['risk_6m']:.0%}, factors: {score['top_factors'][:5]}.\nClinician asks: {question}\n"
             "Answer in under 120 words using only the model output above."}]
