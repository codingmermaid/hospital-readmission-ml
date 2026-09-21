"""
Hospital Readmission Risk Estimator
===================================
A Streamlit app that estimates a patient's risk of being readmitted to hospital
within 30 days, using a LightGBM model trained on the UCI Diabetes 130-US
Hospitals dataset.

The displayed percentage is a *calibrated* probability. The underlying model is
trained with `scale_pos_weight`, whose raw output is inflated roughly 8x on the
odds scale and is not a probability of anything; see `src/calibration.py`.

This is a portfolio / educational project. It is NOT a validated clinical tool
and must not be used for real patient care decisions.

Run locally with: streamlit run app.py
"""

import joblib
import numpy as np
import streamlit as st

from src.calibration import calibrated_probability, load_calibrator, model_scale_pos_weight
from src.features import (
    AGE_DISPLAY_TO_MIDPOINT,
    TRAINING_LEVELS,
    derive_features,
    to_model_frame,
)

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------


@st.cache_resource
def load_artifacts():
    model = joblib.load("model/lightgbm_model.joblib")
    threshold = joblib.load("model/lightgbm_threshold.joblib")
    return model, threshold, model_scale_pos_weight(model), load_calibrator()


model, threshold, scale_pos_weight, calibrator = load_artifacts()

#: Observed 30-day readmission rate in the training data. Shown alongside the
#: estimate, because a risk number means very little without the base rate to
#: compare it against.
BASE_RATE = 0.112

#: The UI label for "this test was not performed".
#:
#: In the raw dataset that state is the literal string "None", and it is ~95% of
#: both test-result columns. It never reached the model as a level: `pd.read_csv`
#: counts "None" among its default NA strings, so the pipeline's CSV round-trip
#: turned it into NaN before training (see `src/io.py`). The shipped model
#: therefore learned "not tested" as *missingness*, and the honest way to ask it
#: about such a patient is to send NaN. Sending the string "None" would be
#: out-of-vocabulary -- LightGBM maps that to NaN as well, but silently and for
#: the wrong reason.
#:
#: `src/io.py` fixes the pipeline so a retrained model gets a real level; until
#: that retrain happens, this mapping is what matches the committed artifact.
NOT_TESTED = "Not tested"


def _test_result_options(column):
    return [NOT_TESTED] + TRAINING_LEVELS[column]


def _encode_test_result(value):
    return np.nan if value == NOT_TESTED else value


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Readmission Risk Estimator", page_icon="🏥", layout="centered")

st.title("Hospital Readmission Risk Estimator")
st.caption(
    "A portfolio project estimating 30-day hospital readmission risk, trained on the "
    "UCI Diabetes 130-US Hospitals dataset. This is an educational demo, not a validated "
    "clinical tool -- do not use it for real patient care decisions."
)

st.divider()

st.subheader("Patient history")
col1, col2 = st.columns(2)
with col1:
    age_group = st.selectbox("Age group", list(AGE_DISPLAY_TO_MIDPOINT.keys()), index=6)
    race = st.selectbox("Race", TRAINING_LEVELS["race"], index=2)
    gender = st.selectbox("Gender", ["Female", "Male"])
with col2:
    number_inpatient = st.number_input(
        "Prior inpatient visits (past year)", min_value=0, max_value=20, value=0
    )
    number_emergency = st.number_input(
        "Prior emergency visits (past year)", min_value=0, max_value=20, value=0
    )
    number_outpatient = st.number_input(
        "Prior outpatient visits (past year)", min_value=0, max_value=20, value=0
    )

st.subheader("Current encounter")
col3, col4 = st.columns(2)
with col3:
    time_in_hospital = st.slider("Length of stay (days)", 1, 14, 3)
    num_lab_procedures = st.number_input("Number of lab procedures", min_value=0, max_value=150, value=40)
    num_procedures = st.number_input("Number of procedures", min_value=0, max_value=10, value=1)
    num_medications = st.number_input("Number of medications", min_value=0, max_value=80, value=15)
with col4:
    number_diagnoses = st.number_input("Number of diagnoses recorded", min_value=1, max_value=16, value=7)
    diag_1_category = st.selectbox("Primary diagnosis category", TRAINING_LEVELS["diag_1_category"])
    medical_specialty = st.selectbox(
        "Attending physician specialty", TRAINING_LEVELS["medical_specialty"], index=4
    )

st.subheader("Diabetes-specific factors")
col5, col6 = st.columns(2)
with col5:
    diabetesMed = st.selectbox("On diabetes medication?", ["Yes", "No"])
    change = st.selectbox(
        "Was medication changed during this stay?",
        ["No", "Ch"],
        format_func=lambda x: "Yes" if x == "Ch" else "No",
    )
with col6:
    max_glu_serum = st.selectbox("Max glucose serum test result", _test_result_options("max_glu_serum"))
    A1Cresult = st.selectbox("A1C test result", _test_result_options("A1Cresult"))

with st.expander("Admission / discharge codes (advanced, optional)"):
    st.caption(
        "These come from the original dataset's coded fields. Defaults reflect a common "
        "emergency-admission, discharged-home scenario -- adjust only if you know the codes."
    )
    admission_type_id = st.number_input("Admission type ID", min_value=1, max_value=8, value=1)
    discharge_disposition_id = st.number_input("Discharge disposition ID", min_value=1, max_value=30, value=1)
    admission_source_id = st.number_input("Admission source ID", min_value=1, max_value=25, value=7)
    num_med_changes = st.number_input(
        "Number of individual medications up/down-dosed", min_value=0, max_value=10, value=0
    )

# ---------------------------------------------------------------------------
# Build feature row and predict
# ---------------------------------------------------------------------------

if st.button("Estimate readmission risk", type="primary"):
    row = derive_features(
        {
            "race": race,
            "gender": gender,
            "admission_type_id": admission_type_id,
            "discharge_disposition_id": discharge_disposition_id,
            "admission_source_id": admission_source_id,
            "time_in_hospital": time_in_hospital,
            "medical_specialty": medical_specialty,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "number_outpatient": number_outpatient,
            "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,
            "max_glu_serum": _encode_test_result(max_glu_serum),
            "A1Cresult": _encode_test_result(A1Cresult),
            "change": change,
            "diabetesMed": diabetesMed,
            "age_midpoint": AGE_DISPLAY_TO_MIDPOINT[age_group],
            "number_diagnoses": number_diagnoses,
            "num_med_changes": num_med_changes,
            "diag_1_category": diag_1_category,
        }
    )

    input_df = to_model_frame(row)
    raw_score = model.predict_proba(input_df)[:, 1][0]

    # Compare on the raw scale, where the saved threshold lives. Both the score
    # and the threshold are passed through the same monotonic calibration for
    # display, so the flag and the shown numbers can never disagree.
    is_high_risk = raw_score >= threshold
    risk = calibrated_probability(raw_score, scale_pos_weight, calibrator)
    threshold_display = calibrated_probability(threshold, scale_pos_weight, calibrator)

    st.divider()
    st.subheader("Result")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.metric(
            "Estimated 30-day readmission risk",
            f"{risk:.1%}",
            delta=f"{risk - BASE_RATE:+.1%} vs average patient",
            delta_color="inverse",
        )
    with col_b:
        if is_high_risk:
            st.warning(
                f"This profile scores above the model's decision threshold "
                f"({threshold_display:.1%}), flagged as higher risk. In a real hospital "
                "setting, this is where care teams might prioritise discharge planning "
                "or a follow-up call."
            )
        else:
            st.success(
                f"This profile scores below the model's decision threshold "
                f"({threshold_display:.1%}), flagged as lower risk."
            )

    st.caption(
        f"The average patient in this dataset has a {BASE_RATE:.1%} chance of 30-day "
        "readmission, so the threshold sits close to the base rate by construction -- it "
        "was chosen to hit ~54% recall, not to be a 50/50 cutoff. At that operating point "
        "precision is about 18%, meaning most flagged patients will not in fact be "
        "readmitted. That is genuine uncertainty in the problem, not a bug. Portfolio "
        "demo, not a clinical decision tool."
    )

    with st.expander("Why this number is not the model's raw output"):
        calibrator_name = (
            "fitted isotonic regression" if calibrator is not None else "analytic prior correction"
        )
        st.markdown(
            f"""
The model is trained with `scale_pos_weight = {scale_pos_weight:.2f}` to cope with the
~11% positive rate. That reweighting means `predict_proba` returns **{raw_score:.1%}**
for this patient -- a probability under a reweighted world where readmission is
{scale_pos_weight:.1f}x more common than it really is, not a probability of readmission.

Converting back to the true prior gives the **{risk:.1%}** shown above. The transform is
monotonic, so it changes the number without changing how patients rank against each
other; ROC-AUC is unaffected.

Calibrator in use: **{calibrator_name}**.
"""
        )

st.divider()
st.caption(
    "Built with LightGBM + Streamlit. Model trained on the UCI Diabetes 130-US Hospitals dataset (1999-2008)."
)
