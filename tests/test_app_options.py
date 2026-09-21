"""
Guards on what the Streamlit form is allowed to offer.

An option the model never saw is not an error at predict time -- LightGBM maps an
unknown level to NaN and returns a confident number anyway. That is how the app
came to offer "None" for two test-result fields that had no "None" level (see
src/io.py). So the check has to happen here.

`app.py` calls `st.set_page_config` at import, so importing it under test means
standing up Streamlit. These are source-level checks instead: cheaper, and they
target exactly the mistake that actually occurred -- a category list typed out by
hand next to a widget instead of taken from the model artifact.
"""

import ast
import os

import pytest

from src.features import CATEGORICAL_COLS, TRAINING_LEVELS

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


@pytest.fixture(scope="module")
def app_source():
    with open(APP_PATH, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def app_tree(app_source):
    return ast.parse(app_source)


def _selectbox_calls(tree):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "selectbox"
        ):
            yield node


def test_app_parses(app_tree):
    assert app_tree is not None


def test_every_selectbox_option_list_is_derived_not_hardcoded(app_tree):
    """Inline option lists are the failure mode; only small closed sets are allowed.

    "Yes"/"No"-style choices are fine to write inline -- they are not model
    categories with a vocabulary to drift from. Anything longer must come from
    TRAINING_LEVELS or another computed source.
    """
    offenders = []
    for call in _selectbox_calls(app_tree):
        if len(call.args) < 2:
            continue
        options = call.args[1]
        if isinstance(options, ast.List) and len(options.elts) > 2:
            label = ast.literal_eval(call.args[0]) if isinstance(call.args[0], ast.Constant) else "?"
            offenders.append(label)
    assert not offenders, f"selectboxes with hardcoded option lists: {offenders}"


def test_no_selectbox_offers_the_string_none(app_tree):
    """The specific regression: "None" was never a training level."""
    for call in _selectbox_calls(app_tree):
        if len(call.args) < 2 or not isinstance(call.args[1], ast.List):
            continue
        values = [e.value for e in call.args[1].elts if isinstance(e, ast.Constant)]
        assert "None" not in values, f"selectbox still offers 'None': {values}"


def test_app_sources_its_categories_from_the_shared_spec(app_source):
    """The category-valued widgets must read TRAINING_LEVELS."""
    assert "TRAINING_LEVELS" in app_source
    for col in ["race", "medical_specialty", "diag_1_category"]:
        assert f'TRAINING_LEVELS["{col}"]' in app_source, col


def test_app_applies_calibration_before_display(app_source):
    """The displayed percentage must not be the raw predict_proba output."""
    assert "calibrated_probability" in app_source
    assert "predict_proba" in app_source
    # The metric shown to the user is the calibrated value, not the raw score.
    assert 'f"{risk:.1%}"' in app_source
    assert 'f"{raw_score:.1%}"' not in app_source.split("st.metric")[1].split("st.caption")[0]


def test_training_levels_covers_every_categorical():
    assert set(TRAINING_LEVELS) == set(CATEGORICAL_COLS)
