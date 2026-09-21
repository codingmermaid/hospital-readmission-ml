import os
import sys

import joblib
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join("model", "lightgbm_model.joblib")
THRESHOLD_PATH = os.path.join("model", "lightgbm_threshold.joblib")


@pytest.fixture(scope="session")
def model():
    """The committed LightGBM model.

    The model artifact *is* checked in, so these tests run in CI without needing
    the dataset (which is not committed).
    """
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"{MODEL_PATH} not present")
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="session")
def threshold():
    if not os.path.exists(THRESHOLD_PATH):
        pytest.skip(f"{THRESHOLD_PATH} not present")
    return joblib.load(THRESHOLD_PATH)


@pytest.fixture
def encounter():
    """A plausible baseline encounter, matching the app's default form state.

    Returned as a fresh dict per test so cases can mutate one field in isolation.
    """
    return {
        "race": "Caucasian",
        "gender": "Female",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": 3,
        "medical_specialty": "Missing",
        "num_lab_procedures": 40,
        "num_procedures": 1,
        "num_medications": 15,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "max_glu_serum": "Norm",
        "A1Cresult": "Norm",
        "change": "No",
        "diabetesMed": "Yes",
        "age_midpoint": 65,
        "number_diagnoses": 7,
        "num_med_changes": 0,
        "diag_1_category": "Circulatory",
    }
