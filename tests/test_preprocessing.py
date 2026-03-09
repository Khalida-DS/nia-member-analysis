"""
tests/test_preprocessing.py
============================
Unit tests for src/features/preprocessing.py.
Run: pytest tests/test_preprocessing.py -v
"""
import numpy as np
import pandas as pd
import pytest

from src.config import get_config
from src.features.preprocessing import (
    audit_missing,
    build_matrix,
    encode_flags,
    encode_ordinals,
    engineer_features,
    aggregate_household,
)

CFG  = get_config()
DATA = CFG.data


# ── Fixture: 4-row minimal DataFrame that covers all edge cases ───────────────

@pytest.fixture
def raw_df():
    """
    4 rows:
    - rows 0,1: same household (HH 100) — tests aggregation, max on products
    - row 2:    household 101 — primary member who used ERS
    - row 3:    household 102 — missing Income and Credit (tests NaN handling)
    """
    return pd.DataFrame({
        "Member Key":      [1, 2, 3, 4],
        "Household Key":   [100, 100, 101, 102],
        "FSV Credit Card Flag":    ["Y", "N", "Y", "N"],
        "INS Client Flag":         ["N", "N", "Y", "N"],
        "FSV CMSI Flag":           ["N", "Y", "N", "N"],
        "FSV Deposit Program Flag":["N", "N", "N", "N"],
        "FSV Home Equity Flag":    ["N", "N", "N", "N"],
        "FSV ID Theft Flag":       ["N", "N", "Y", "N"],
        "FSV Mortgage Flag":       ["N", "N", "N", "N"],
        "TRV Globalware Flag":     ["Y", "N", "N", "N"],
        "New Mover Flag":          ["N", "N", "N", "Y"],
        "Income":          ["50-59,999", "100-149,999", "Under 10K", None],
        "Credit Ranges":   ["700-749",   "800+",        "499 & Less", "Unknown"],
        "Number of Children": ["No children", "Two Children", "One Child", None],
        "Mail Responder":  ["Yes", "No", "Yes", "Yes"],
        "Email Available": ["Y",   "N",  "Y",   "Y"],
        "Do Not Direct Mail Solicit": [0, 1, 0, 0],
        "Member Type":     ["Primary", "Associate", "Primary", "Primary"],
        "Member Tenure Years": [5.0,   5.0,   12.0,   3.0],
        "Home Owner":      ["Home Owner", "Renter", "Home Owner", None],
        "Dwelling Type":   ["Single Family", "Apartment", "Single Family", "Single Family"],
        "Length Of Residence": [8.0, 3.0, 15.0, None],
        "ZIP":             ["02101", "02101", "06101", "10001"],
        "Mosaic Household":          ["A", "B", "A",  "C"],
        "Mosaic Global Household":   ["G1","G2","G1", "G3"],
        "kcl_B_IND_MosaicsGrouping": ["Golden Year", "Family", "Golden Year", "Boomers"],
        "ERS ENT Count Year 1": [2.0, 0.0, 1.0, 0.0],
        "ERS ENT Count Year 2": [1.0, 0.0, 0.0, 0.0],
        "ERS ENT Count Year 3": [0.0, 0.0, 2.0, 0.0],
        "ERS Member Cost Year 1": [45.0, 0.0, 30.0, 0.0],
        "ERS Member Cost Year 2": [20.0, 0.0,  0.0, 0.0],
        "ERS Member Cost Year 3": [ 0.0, 0.0, 80.0, 0.0],
        "Total Cost": [65.0, 0.0, 110.0, 0.0],
    })


# ── Helpers ────────────────────────────────────────────────────────────────────

def _encode(raw_df):
    df = encode_ordinals(raw_df, DATA)
    return encode_flags(df, DATA)


def _household(raw_df):
    df = _encode(raw_df)
    return aggregate_household(df, DATA.product_names)


def _engineered(raw_df):
    return engineer_features(_household(raw_df), DATA.product_names)


# ── audit_missing ──────────────────────────────────────────────────────────────

class TestAuditMissing:
    def test_returns_dataframe(self, raw_df):
        assert isinstance(audit_missing(raw_df), pd.DataFrame)

    def test_only_includes_null_columns(self, raw_df):
        report = audit_missing(raw_df)
        assert (report["missing_count"] > 0).all()

    def test_sorted_descending(self, raw_df):
        report = audit_missing(raw_df)
        assert report["missing_pct"].is_monotonic_decreasing

    def test_empty_when_no_nulls(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert len(audit_missing(df)) == 0


# ── encode_ordinals ────────────────────────────────────────────────────────────

class TestEncodeOrdinals:
    def test_income_becomes_numeric(self, raw_df):
        df = encode_ordinals(raw_df, DATA)
        assert pd.api.types.is_float_dtype(df["Income"]) or pd.api.types.is_integer_dtype(df["Income"])

    def test_income_values_in_valid_range(self, raw_df):
        df = encode_ordinals(raw_df, DATA)
        assert df["Income"].dropna().between(0, 300_000).all()

    def test_credit_becomes_numeric(self, raw_df):
        df = encode_ordinals(raw_df, DATA)
        assert pd.api.types.is_float_dtype(df["Credit Ranges"])

    def test_unknown_credit_becomes_nan(self, raw_df):
        df = encode_ordinals(raw_df, DATA)
        # Row 3 has "Unknown" credit — should be NaN
        assert pd.isna(df["Credit Ranges"].iloc[3])

    def test_null_income_stays_null(self, raw_df):
        df = encode_ordinals(raw_df, DATA)
        assert pd.isna(df["Income"].iloc[3])

    def test_children_mapped_to_int(self, raw_df):
        df = encode_ordinals(raw_df, DATA)
        vals = df["Number of Children"].dropna()
        assert (vals == vals.astype(int)).all()


# ── encode_flags ───────────────────────────────────────────────────────────────

class TestEncodeFlags:
    def test_product_flags_are_0_or_1(self, raw_df):
        df = _encode(raw_df)
        for prod in DATA.product_names:
            if prod in df.columns:
                assert df[prod].isin([0, 1]).all(), f"{prod} has unexpected value"

    def test_raw_product_cols_are_dropped(self, raw_df):
        df = _encode(raw_df)
        for col in DATA.product_cols:
            assert col not in df.columns

    def test_member_type_splits_into_two_binary_cols(self, raw_df):
        df = _encode(raw_df)
        assert "PrimaryMember"   in df.columns
        assert "AssociateMember" in df.columns
        assert "Member Type"     not in df.columns

    def test_mail_responder_is_0_or_1(self, raw_df):
        df = _encode(raw_df)
        assert df["Mail Responder"].isin([0, 1]).all()

    def test_primary_member_flag_correct(self, raw_df):
        df = _encode(raw_df)
        # Row 0: Primary → 1; Row 1: Associate → 0
        assert df["PrimaryMember"].iloc[0]   == 1
        assert df["AssociateMember"].iloc[1] == 1


# ── aggregate_household ────────────────────────────────────────────────────────

class TestAggregateHousehold:
    def test_one_row_per_unique_household(self, raw_df):
        df_hh = _household(raw_df)
        assert len(df_hh) == raw_df["Household Key"].nunique()

    def test_no_duplicate_household_keys(self, raw_df):
        df_hh = _household(raw_df)
        assert df_hh.index.is_unique

    def test_product_flag_max_is_correct(self, raw_df):
        """HH 100 has member 1 (Y) and member 2 (N) for Credit Card → max = 1"""
        df_hh = _household(raw_df)
        assert df_hh.loc[100, "FSV Credit Card"] == 1

    def test_product_flag_stays_zero_when_no_member_holds_it(self, raw_df):
        """HH 102 has no Credit Card → max = 0"""
        df_hh = _household(raw_df)
        assert df_hh.loc[102, "FSV Credit Card"] == 0

    def test_product_cols_are_0_or_1(self, raw_df):
        df_hh = _household(raw_df)
        for prod in DATA.product_names:
            if prod in df_hh.columns:
                assert df_hh[prod].isin([0, 1]).all()


# ── engineer_features ──────────────────────────────────────────────────────────

class TestEngineerFeatures:
    def test_total_ers_calls_is_created(self, raw_df):
        df = _engineered(raw_df)
        assert "total_ers_calls" in df.columns

    def test_total_ers_calls_nonnegative(self, raw_df):
        df = _engineered(raw_df)
        assert (df["total_ers_calls"] >= 0).all()

    def test_product_count_is_created(self, raw_df):
        df = _engineered(raw_df)
        assert "product_count" in df.columns
        assert (df["product_count"] >= 0).all()

    def test_is_multi_product_is_binary(self, raw_df):
        df = _engineered(raw_df)
        assert df["is_multi_product"].isin([0, 1]).all()

    def test_has_used_ers_is_binary(self, raw_df):
        df = _engineered(raw_df)
        assert df["has_used_ers"].isin([0, 1]).all()

    def test_is_long_term_member_is_binary(self, raw_df):
        df = _engineered(raw_df)
        if "is_long_term_member" in df.columns:
            assert df["is_long_term_member"].isin([0, 1]).all()

    def test_is_high_income_is_binary(self, raw_df):
        df = _engineered(raw_df)
        if "is_high_income" in df.columns:
            assert df["is_high_income"].isin([0, 1]).all()


# ── build_matrix ───────────────────────────────────────────────────────────────

class TestBuildMatrix:
    def test_output_is_fully_numeric(self, raw_df):
        df = _engineered(raw_df)
        X  = build_matrix(df, drop_cols=DATA.product_names)
        assert all(np.issubdtype(dt, np.number) for dt in X.dtypes)

    def test_no_null_values(self, raw_df):
        df = _engineered(raw_df)
        X  = build_matrix(df, drop_cols=DATA.product_names)
        assert X.isnull().sum().sum() == 0

    def test_product_cols_excluded(self, raw_df):
        df = _engineered(raw_df)
        X  = build_matrix(df, drop_cols=DATA.product_names)
        for prod in DATA.product_names:
            assert prod not in X.columns

    def test_row_count_preserved(self, raw_df):
        df = _engineered(raw_df)
        X  = build_matrix(df, drop_cols=DATA.product_names)
        assert len(X) == raw_df["Household Key"].nunique()