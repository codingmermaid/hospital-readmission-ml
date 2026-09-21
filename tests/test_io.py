"""
Tests for the CSV round-trip bug described in src/io.py.

The failure mode: `pd.read_csv` counts the literal string "None" among its
default NA values, so the "not tested" level of A1Cresult and max_glu_serum --
about 95% of both columns -- silently became NaN as data moved between pipeline
stages.
"""

import pandas as pd
import pytest

from src.io import NONE_IS_A_LEVEL, read_pipeline_csv

ROWS = "A1Cresult,max_glu_serum,race\nNone,None,Caucasian\nNorm,>200,?\n>7,>300,Asian\n"


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "sample.csv"
    p.write_text(ROWS, encoding="utf-8")
    return str(p)


def test_demonstrates_the_bug_in_plain_read_csv(csv_path):
    """Characterisation test: this is what the pipeline used to do.

    Kept deliberately. If a future pandas stops treating "None" as NA, this fails
    and tells us the workaround is no longer needed.
    """
    df = pd.read_csv(csv_path)
    assert df["A1Cresult"].isna().sum() == 1
    assert df["max_glu_serum"].isna().sum() == 1


def test_none_survives_as_a_real_level(csv_path):
    df = read_pipeline_csv(csv_path)
    for col in NONE_IS_A_LEVEL:
        assert df[col].isna().sum() == 0, col
        assert "None" in set(df[col]), col


def test_question_mark_is_still_missing(csv_path):
    """ "?" is this dataset's genuine missing marker and must keep mapping to NaN."""
    df = read_pipeline_csv(csv_path)
    assert df["race"].isna().sum() == 1


def test_other_values_are_unchanged(csv_path):
    df = read_pipeline_csv(csv_path)
    assert list(df["A1Cresult"]) == ["None", "Norm", ">7"]
    assert list(df["max_glu_serum"]) == ["None", ">200", ">300"]


def test_caller_can_override_the_defaults(csv_path):
    df = read_pipeline_csv(csv_path, keep_default_na=True, na_values=None)
    assert df["A1Cresult"].isna().sum() == 1


def test_empty_field_is_missing(tmp_path):
    p = tmp_path / "gap.csv"
    p.write_text("a,b\n1,\n2,x\n", encoding="utf-8")
    assert read_pipeline_csv(str(p))["b"].isna().sum() == 1
