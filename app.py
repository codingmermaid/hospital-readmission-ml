"""
Hospital Readmission Risk Estimator
=====================================
A Streamlit app that estimates a patient's risk of being readmitted
to hospital within 30 days, using a LightGBM model trained on the
UCI Diabetes 130-US Hospitals dataset.

This is a portfolio / educational project. It is NOT a validated
clinical tool and must not be used for real patient care decisions.

Run locally with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load("model/lightgbm_model.joblib")
    categorical_cols = joblib.load("model/categorical_cols.joblib")
    threshold = joblib.load("model/lightgbm_threshold.joblib")
    return model, categorical_cols, threshold


model, categorical_cols, threshold = load_artifacts()

FEATURE_ORDER = [
    'race', 'gender', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
    'time_in_hospital', 'medical_specialty', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient',
    'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed', 'total_prior_visits',
    'had_prior_inpatient', 'age_midpoint', 'number_diagnoses_capped', 'med_changed',
    'on_diabetes_med', 'num_med_changes', 'diag_1_category'
]

AGE_MIDPOINTS = {
    "0-10": 5, "10-20": 15, "20-30": 25, "30-40": 35, "40-50": 45,
    "50-60": 55, "60-70": 65, "70-80": 75, "80-90": 85, "90-100": 95
}

DIAG_CATEGORIES = [
    "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
    "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Unknown"
]

MEDICAL_SPECIALTIES = [
    "Missing", "InternalMedicine", "Family/GeneralPractice", "Cardiology",
    "Surgery-General", "Orthopedics", "Orthopedics-Reconstructive",
    "Emergency/Trauma", "Nephrology", "Radiologist", "Other"
]

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
    age_group = st.selectbox("Age group", list(AGE_MIDPOINTS.keys()), index=6)
    race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"])
    gender = st.selectbox("Gender", ["Female", "Male"])
with col2:
    number_inpatient = st.number_input("Prior inpatient visits (past year)", min_value=0, max_value=20, value=0)
    number_emergency = st.number_input("Prior emergency visits (past year)", min_value=0, max_value=20, value=0)
    number_outpatient = st.number_input("Prior outpatient visits (past year)", min_value=0, max_value=20, value=0)

st.subheader("Current encounter")
col3, col4 = st.columns(2)
with col3:
    time_in_hospital = st.slider("Length of stay (days)", 1, 14, 3)
    num_lab_procedures = st.number_input("Number of lab procedures", min_value=0, max_value=150, value=40)
    num_procedures = st.number_input("Number of procedures", min_value=0, max_value=10, value=1)
    num_medications = st.number_input("Number of medications", min_value=0, max_value=80, value=15)
with col4:
    number_diagnoses = st.number_input("Number of diagnoses recorded", min_value=1, max_value=16, value=7)
    diag_1_category = st.selectbox("Primary diagnosis category", DIAG_CATEGORIES)
    medical_specialty = st.selectbox("Attending physician specialty", MEDICAL_SPECIALTIES)

st.subheader("Diabetes-specific factors")
col5, col6 = st.columns(2)
with col5:
    diabetesMed = st.selectbox("On diabetes medication?", ["Yes", "No"])
    change = st.selectbox("Was medication changed during this stay?", ["No", "Ch"], format_func=lambda x: "Yes" if x == "Ch" else "No")
with col6:
    max_glu_serum = st.selectbox("Max glucose serum test result", ["None", "Norm", ">200", ">300"])
    A1Cresult = st.selectbox("A1C test result", ["None", "Norm", ">7", ">8"])

with st.expander("Admission / discharge codes (advanced, optional)"):
    st.caption(
        "These come from the original dataset's coded fields. Defaults reflect a common "
        "'emergency admission, discharged home' scenario -- adjust only if you know the codes."
    )
    admission_type_id = st.number_input("Admission type ID", min_value=1, max_value=8, value=1)
    discharge_disposition_id = st.number_input("Discharge disposition ID", min_value=1, max_value=30, value=1)
    admission_source_id = st.number_input("Admission source ID", min_value=1, max_value=25, value=7)
    num_med_changes = st.number_input("Number of individual medications up/down-dosed", min_value=0, max_value=10, value=0)

# ---------------------------------------------------------------------------
# Build feature row and predict
# ---------------------------------------------------------------------------

if st.button("Estimate readmission risk", type="primary"):
    age_midpoint = AGE_MIDPOINTS[age_group]
    total_prior_visits = number_inpatient + number_emergency + number_outpatient
    had_prior_inpatient = int(number_inpatient > 0)
    number_diagnoses_capped = min(number_diagnoses, 10)
    med_changed = int(change == "Ch")
    on_diabetes_med = int(diabetesMed == "Yes")

    row = {
        'race': race,
        'gender': gender,
        'admission_type_id': admission_type_id,
        'discharge_disposition_id': discharge_disposition_id,
        'admission_source_id': admission_source_id,
        'time_in_hospital': time_in_hospital,
        'medical_specialty': medical_specialty,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'max_glu_serum': max_glu_serum,
        'A1Cresult': A1Cresult,
        'change': change,
        'diabetesMed': diabetesMed,
        'total_prior_visits': total_prior_visits,
        'had_prior_inpatient': had_prior_inpatient,
        'age_midpoint': age_midpoint,
        'number_diagnoses_capped': number_diagnoses_capped,
        'med_changed': med_changed,
        'on_diabetes_med': on_diabetes_med,
        'num_med_changes': num_med_changes,
        'diag_1_category': diag_1_category,
    }

    input_df = pd.DataFrame([row])[FEATURE_ORDER]
    for col in categorical_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype("category")

    risk_score = model.predict_proba(input_df)[:, 1][0]
    is_high_risk = risk_score >= threshold

    st.divider()
    st.subheader("Result")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.metric("Estimated 30-day readmission risk", f"{risk_score:.1%}")
    with col_b:
        if is_high_risk:
            st.warning(
                f"This profile scores above the model's decision threshold ({threshold:.1%}), "
                "flagged as higher risk. In a real hospital setting, this is where care teams "
                "might prioritize discharge planning or a follow-up call."
            )
        else:
            st.success(
                f"This profile scores below the model's decision threshold ({threshold:.1%}), "
                "flagged as lower risk."
            )

    st.caption(
        "Reminder: this model's overall recall is about 54% and precision about 18% at this "
        "threshold, reflecting genuine uncertainty in readmission prediction. It is a portfolio "
        "demo, not a clinical decision tool."
    )

st.divider()
st.caption("Built with LightGBM + Streamlit. Model trained on the UCI Diabetes 130-US Hospitals dataset (1999-2008).")
