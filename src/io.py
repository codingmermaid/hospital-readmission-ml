"""
CSV reading that does not quietly eat a clinical category.

The pipeline hands data between stages as CSV: ``clean.py`` writes
``diabetic_data_clean.csv``, ``feature_engineering.py`` reads it and writes
``diabetic_data_features.csv``, and the model scripts read that. Every one of
those reads used a bare ``pd.read_csv``.

``pd.read_csv`` treats the literal string ``"None"`` as missing. It is in
``pandas._libs.parsers.STR_NA_VALUES`` alongside ``"NA"``, ``"NULL"`` and
``"NaN"``. Two columns in this dataset use ``"None"`` as a real, meaningful
level:

* ``A1Cresult`` -- ``"None"`` means the A1C test was not performed
* ``max_glu_serum`` -- ``"None"`` means the glucose serum test was not performed

"Not tested" is roughly 95% of both columns, and it is clinically informative:
whether a clinician ordered the test is a signal about the encounter. On the
first CSV round-trip all of it silently became NaN.

You can see the damage in the shipped artifact. The levels LightGBM recorded at
training time are::

    max_glu_serum -> ['>200', '>300', 'Norm']
    A1Cresult     -> ['>7', '>8', 'Norm']

No ``"None"``, because by the time the model saw the data there was none left.

Two separate consequences, worth keeping apart:

1. **For LightGBM** the effect is mild. It handles NaN natively as its own split
   direction, so "not tested" was still learnable -- just as missingness rather
   than as a labelled level, and entirely by accident.
2. **For the logistic-regression baseline** it is worse. That pipeline one-hot
   encodes the categoricals, and the ``"None"`` level simply vanished from the
   design matrix, so the baseline could not represent "not tested" at all. Part
   of the LightGBM-vs-baseline gap reported in the README is this bug rather than
   a real modelling difference.

:func:`read_pipeline_csv` keeps the level. Note that adopting it changes the
feature space, so the numbers only move once the model is retrained -- see the
README section "Retraining after the None fix".
"""

from __future__ import annotations

import pandas as pd

#: Columns where the literal string "None" is a real level meaning "not tested",
#: rather than a missing value.
NONE_IS_A_LEVEL = ("A1Cresult", "max_glu_serum")

#: What the raw data actually uses for missing. The UCI export encodes missing as
#: "?", which `clean.py` converts explicitly; the empty string covers ragged rows.
REAL_NA_VALUES = ["", "?"]


def read_pipeline_csv(path: str, **kwargs) -> pd.DataFrame:
    """``pd.read_csv`` that does not reinterpret the string ``"None"`` as NaN.

    Disables pandas' default NA vocabulary entirely and substitutes the values
    this dataset genuinely uses for missing. Anything the caller passes for
    ``na_values`` or ``keep_default_na`` wins, so callers that want the old
    behaviour can still ask for it.
    """
    kwargs.setdefault("keep_default_na", False)
    kwargs.setdefault("na_values", REAL_NA_VALUES)
    return pd.read_csv(path, **kwargs)
