"""
Turn raw model contributions into plain-English factors a clinician can read.

Why: the model works on encoded columns ("ca_no", "ph_missing", "dzgroup_Coma"). Shown raw,
"ca_no (raises risk)" reads as "having no cancer raises risk" when it really means
"this patient is NOT in the no-cancer group". So we:
  1. add up all encoded columns that come from the same original question
     (ca_no + ca_yes -> "cancer"; alb + alb_missing -> "albumin"),
  2. label each factor with the patient's actual value ("Cancer: metastatic", "Albumin 2.1 g/dL").
Summing is exact for this model: the log-odds is a sum over columns.
"""
import math

CATEGORICAL = {"sex", "dzgroup", "dzclass", "ca"}

NAMES = {
    "age": ("Age", "{:.0f}"),
    "num.co": ("Number of other illnesses", "{:.0f}"),
    "scoma": ("Coma score (0 = fully awake)", "{:.0f}"),
    "sps": ("Acute physiology score (SUPPORT)", "{:.0f}"),
    "aps": ("Acute physiology score (APACHE III)", "{:.0f}"),
    "hday": ("Days in hospital before assessment", "{:.0f}"),
    "diabetes": ("Diabetes", None),
    "dementia": ("Dementia", None),
    "meanbp": ("Mean blood pressure", "{:.0f} mmHg"),
    "wblc": ("White cell count", "{:.1f}"),
    "hrt": ("Heart rate", "{:.0f} /min"),
    "resp": ("Breathing rate", "{:.0f} /min"),
    "temp": ("Temperature", "{:.1f} °C"),
    "pafi": ("Oxygen level (PaO2/FiO2)", "{:.0f}"),
    "alb": ("Albumin (nutrition)", "{:.1f} g/dL"),
    "bili": ("Bilirubin (liver)", "{:.1f} mg/dL"),
    "crea": ("Creatinine (kidneys)", "{:.1f} mg/dL"),
    "sod": ("Sodium", "{:.0f} mEq/L"),
    "ph": ("Blood pH", "{:.2f}"),
    "glucose": ("Glucose", "{:.0f} mg/dL"),
    "bun": ("Urea (kidneys)", "{:.0f} mg/dL"),
    "urine": ("Urine output", "{:.0f} mL/day"),
    "adlsc": ("Difficulty with daily activities (0-7)", "{:.1f}"),
    "sex": ("Sex", None),
    "dzgroup": ("Diagnosis", None),
    "dzclass": ("Diagnosis group", None),
    "ca": ("Cancer", None),
}
CANCER = {"no": "none", "yes": "yes, not spread", "metastatic": "spread (metastatic)"}


def _base(col):
    if col.endswith("_missing"):
        return col[: -len("_missing")]
    head = col.split("_", 1)[0]
    return head if head in CATEGORICAL and "_" in col else col


def _is_missing(v):
    return v is None or v == "" or (isinstance(v, float) and math.isnan(v)) or str(v).lower() == "nan"


def _label(base, raw):
    name, fmt = NAMES.get(base, (base, None))
    if _is_missing(raw):
        return f"{name}: not measured"
    if base == "ca":
        return f"{name}: {CANCER.get(str(raw), raw)}"
    if base in ("diabetes", "dementia"):
        return f"{name}: {'yes' if float(raw) >= 0.5 else 'no'}"
    if fmt is None:
        return f"{name}: {raw}"
    try:
        return f"{name} {fmt.format(float(raw))}"
    except (TypeError, ValueError):
        return f"{name}: {raw}"


def readable_factors(contributions: dict, raw_patient, k=5):
    """contributions: {encoded_column: log-odds contribution}; raw_patient: dict-like of original
    SUPPORT2 values (keys like 'ca', 'alb'). Returns the k largest grouped factors."""
    grouped = {}
    for col, v in contributions.items():
        b = _base(col)
        grouped[b] = grouped.get(b, 0.0) + float(v)
    raw = {str(key).lower().replace("_", "."): val for key, val in dict(raw_patient).items()}
    out = []
    for b, v in sorted(grouped.items(), key=lambda kv: -abs(kv[1]))[:k]:
        out.append({"feature": b, "label": _label(b, raw.get(b)),
                    "effect": "raises risk" if v > 0 else "lowers risk",
                    "log_odds_contribution": v})
    return out
