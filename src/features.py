"""
Single source of truth for the feature contract.

Before this module existed, the feature lists were written out by hand in three
places -- `scripts/baseline_model.py`, `scripts/lightgbm_model.py` and `app.py`.
They agreed by luck rather than by construction, and nothing checked that the
column order `app.py` built matched the order the model was trained on. Getting
that order wrong does not raise; LightGBM would just read the wrong column for
every feature and return a confident, meaningless number.

Everything that describes "what a row looks like" now lives here, and the tests
assert it against the saved model artifact.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------

#: Identifiers and raw columns replaced by engineered versions. Dropped before
#: training. `readmitted` is the original 3-class target and would leak.
DROP_FOR_MODEL = [
    "encounter_id",
    "patient_nbr",
    "readmitted",
    "age",  # replaced by age_midpoint
    "diag_1",
    "diag_2",
    "diag_3",  # replaced by diag_1_category
    "number_diagnoses",  # replaced by number_diagnoses_capped
]

#: The 23 per-drug columns. Summarised by `num_med_changes`; the raw versions are
#: dropped so the baseline model does not get 20+ near-constant dummies.
MED_COLS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "examide",
    "citoglipton",
]

TARGET = "readmitted_30d"
GROUP_COL = "patient_nbr"

#: Exact column order the LightGBM model was trained on. Order is load-bearing:
#: LightGBM matches features positionally, so a permutation here silently
#: produces garbage predictions rather than an error.
FEATURE_ORDER = [
    "race",
    "gender",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
    "total_prior_visits",
    "had_prior_inpatient",
    "age_midpoint",
    "number_diagnoses_capped",
    "med_changed",
    "on_diabetes_med",
    "num_med_changes",
    "diag_1_category",
]

#: Features LightGBM treats as native categoricals.
CATEGORICAL_COLS = [
    "race",
    "gender",
    "medical_specialty",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
    "diag_1_category",
]

# ---------------------------------------------------------------------------
# Categorical levels
# ---------------------------------------------------------------------------

#: Levels the shipped model actually saw during training, recovered from
#: `booster_.pandas_categorical`. Any value outside these sets is mapped to NaN
#: by LightGBM at predict time -- which is silent, so the UI must only ever offer
#: values from here. `tests/test_model_contract.py` enforces that.
#:
#: Note what is NOT here: `max_glu_serum` and `A1Cresult` have no "None" level,
#: even though "None" ("test not performed") is ~95% of the raw data. See
#: `src/io.py` for why it went missing.
TRAINING_LEVELS = {
    "race": ["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other", "Unknown"],
    "gender": ["Female", "Male", "Unknown/Invalid"],
    "medical_specialty": [
        "Cardiology",
        "Emergency/Trauma",
        "Family/GeneralPractice",
        "InternalMedicine",
        "Missing",
        "Nephrology",
        "Orthopedics",
        "Orthopedics-Reconstructive",
        "Other",
        "Radiologist",
        "Surgery-General",
    ],
    "max_glu_serum": [">200", ">300", "Norm"],
    "A1Cresult": [">7", ">8", "Norm"],
    "change": ["Ch", "No"],
    "diabetesMed": ["No", "Yes"],
    "diag_1_category": [
        "Circulatory",
        "Diabetes",
        "Digestive",
        "Genitourinary",
        "Injury",
        "Musculoskeletal",
        "Neoplasms",
        "Other",
        "Respiratory",
        "Unknown",
    ],
}

#: Raw dataset age buckets -> numeric midpoint, so models can use age ordinally.
AGE_MIDPOINTS = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}

#: The same buckets with UI-friendly labels ("[60-70)" -> "60-70"), for the
#: Streamlit selectbox.
AGE_DISPLAY_TO_MIDPOINT = {label.strip("[)"): midpoint for label, midpoint in AGE_MIDPOINTS.items()}

MAX_DIAGNOSES_CAP = 10


def categorize_diag(code) -> str:
    """Collapse an ICD-9 primary diagnosis code into a broad clinical category.

    Full ICD-9 has 700+ levels, far too many to learn from directly at this
    sample size. The buckets follow the grouping used in the Strack et al. paper
    that accompanies this dataset.
    """
    if code == "Unknown" or pd.isna(code):
        return "Unknown"
    try:
        code_num = float(code)
    except (ValueError, TypeError):
        # V codes (supplemental) and E codes (external causes of injury).
        return "Other"

    if 390 <= code_num <= 459 or code_num == 785:
        return "Circulatory"
    if 460 <= code_num <= 519 or code_num == 786:
        return "Respiratory"
    if 520 <= code_num <= 579 or code_num == 787:
        return "Digestive"
    if 250 <= code_num < 251:
        return "Diabetes"
    if 800 <= code_num <= 999:
        return "Injury"
    if 710 <= code_num <= 739:
        return "Musculoskeletal"
    if 580 <= code_num <= 629 or code_num == 788:
        return "Genitourinary"
    if 140 <= code_num <= 239:
        return "Neoplasms"
    return "Other"


def model_columns(df: pd.DataFrame) -> list[str]:
    """Return the modelling columns of `df`, dropping ids, raws and the target."""
    drop = set(DROP_FOR_MODEL) | set(MED_COLS) | {TARGET}
    return [c for c in df.columns if c not in drop]


def derive_features(raw: dict) -> dict:
    """Derive the engineered features from a dict of raw encounter inputs.

    `scripts/feature_engineering.py` does this column-wise over the full frame;
    this does it for a single row so the app and the pipeline cannot drift apart.
    """
    row = dict(raw)
    row["total_prior_visits"] = row["number_inpatient"] + row["number_emergency"] + row["number_outpatient"]
    row["had_prior_inpatient"] = int(row["number_inpatient"] > 0)
    row["number_diagnoses_capped"] = min(row.pop("number_diagnoses"), MAX_DIAGNOSES_CAP)
    row["med_changed"] = int(row["change"] == "Ch")
    row["on_diabetes_med"] = int(row["diabetesMed"] == "Yes")
    return row


def to_model_frame(row: dict) -> pd.DataFrame:
    """Build a single-row DataFrame in exactly the order the model expects.

    Categoricals are given the full training level set rather than only the one
    value present in the row. LightGBM does remap by label, so a single-value
    category happens to work -- but relying on that is fragile, and being
    explicit means an out-of-vocabulary value is visible as NaN here rather than
    silently absorbed inside the booster.
    """
    frame = pd.DataFrame([row])[FEATURE_ORDER]
    for col in CATEGORICAL_COLS:
        # Map out-of-vocabulary values to NaN explicitly. Passing them straight
        # to pd.Categorical would do the same thing today, but that is deprecated
        # and is scheduled to start raising, so do it ourselves.
        known = frame[col].where(frame[col].isin(TRAINING_LEVELS[col]))
        frame[col] = pd.Categorical(known, categories=TRAINING_LEVELS[col])
    return frame
