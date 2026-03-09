"""
src/features/preprocessing.py
==============================
Raw CSV → household-level feature matrix.

PIPELINE ORDER (order matters — do not change without reading the comments)
---------------------------------------------------------------------------
1.  load_raw()              Read CSV, drop auto-index col
2.  aggregate_ers_costs()   Pivot service-call rows → annual cost per member
                            *** MUST happen before deduplication ***
                            If you dedup first, you collapse call rows and
                            lose cost history for multi-call members.
3.  select_and_filter()     Keep relevant columns, remove CANCELLED members
4.  encode_ordinals()       Income / Credit / Children text → numeric
5.  encode_flags()          Y/N product flags → 1/0; Member Type → binary flags
6.  deduplicate()           One row per Member Key
7.  join_costs()            Attach aggregated cost columns
8.  aggregate_household()   Member rows → one row per Household Key
9.  engineer_features()     Create domain-driven derived features
10. build_matrix()          Median impute + one-hot encode → fully numeric X

Design rules
------------
* Every function is PURE: (DataFrame, ...) → DataFrame.  No global state.
* All settings flow through the ProjectConfig object (never hardcoded here).
* Every column exclusion has a documented reason.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from src.config import DataConfig, ProjectConfig, get_config

warnings.filterwarnings("ignore")


# ── 1. Load ──────────────────────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    """Read member_sample.csv and drop the auto-generated index column."""
    df = pd.read_csv(path, low_memory=False)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def audit_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame of missing counts/percentages sorted descending.
    Use this before any cleaning to document the starting state of the data.
    """
    n     = len(df)
    miss  = df.isnull().sum()
    report = pd.DataFrame({
        "missing_count": miss,
        "missing_pct":   (miss / n * 100).round(2),
    })
    return report[report["missing_count"] > 0].sort_values("missing_pct", ascending=False)


# ── 2. Aggregate ERS costs (BEFORE deduplication) ────────────────────────────

def aggregate_ers_costs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot service-call rows → one row per Member Key showing annual ERS cost.

    WHY THIS MUST RUN BEFORE DEDUPLICATION
    ---------------------------------------
    The raw CSV has one row per (member, service call).
    A member with 3 calls in 2017 has 3 rows, each with a Total Cost value.
    If you deduplicate first (keeping only 1 row per member), you lose
    2 of those 3 cost rows.  This function aggregates ALL rows first,
    then deduplication is safe.

    Returns a DataFrame indexed by Member Key with columns:
    Cost 2014, Cost 2015, ..., Cost 2019, Total Cost
    """
    df = df.copy()
    df["SC Date"]   = pd.to_datetime(df["SC Date"], errors="coerce")
    df["_cost_year"] = df["SC Date"].dt.year.fillna(0).astype(int)

    cost = (
        df[["Member Key", "Total Cost", "_cost_year"]]
        .groupby(["Member Key", "_cost_year"])["Total Cost"]
        .sum()
        .reset_index()
        .pivot(index="Member Key", columns="_cost_year", values="Total Cost")
        .fillna(0)
    )

    # Keep only real years (drop the 0-year bucket for rows with no SC Date)
    year_cols  = sorted([c for c in cost.columns if c > 0])
    cost       = cost[year_cols].copy()
    cost.columns = [f"Cost {y}" for y in year_cols]
    cost["Total Cost"] = cost.sum(axis=1)
    return cost


# ── 3. Select columns and filter population ───────────────────────────────────

def select_and_filter(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """
    Keep only columns relevant to the model; remove CANCELLED members.

    Exclusion reasons are documented in configs/settings.yaml under
    data.exclude_cols.  This function applies the same logic explicitly
    so the reasoning is visible here too.
    """
    keep = list(dict.fromkeys(cfg.product_cols + [
        "Member Key", "Household Key", "ZIP",
        "Number of Children", "Length Of Residence",
        "Mail Responder", "Home Owner", "Income",
        "Dwelling Type", "Credit Ranges",
        "Do Not Direct Mail Solicit", "Email Available",
        "ERS ENT Count Year 1", "ERS ENT Count Year 2", "ERS ENT Count Year 3",
        "ERS Member Cost Year 1", "ERS Member Cost Year 2", "ERS Member Cost Year 3",
        "Member Status", "Member Tenure Years", "Member Type",
        "Mosaic Household", "Mosaic Global Household", "kcl_B_IND_MosaicsGrouping",
        "SC Date", "Total Cost",
    ]))
    present = [c for c in keep if c in df.columns]
    df = df[present].copy()

    # Remove cancelled members — their behavioral patterns differ from active members
    # and including them would bias the model toward cancellation signals
    if "Member Status" in df.columns:
        df = df[df["Member Status"].isin(cfg.member_status_keep)]
        df = df.drop(columns=["Member Status"])

    return df


# ── 4. Encode ordinal columns ─────────────────────────────────────────────────

def encode_ordinals(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """
    Map Income / Credit Ranges / Number of Children text bands → numeric midpoints.

    Unknown or unmapped values become NaN, not 0.
    NaN is honest: "we don't know".  0 would mean "zero income / zero credit".
    Downstream median imputation handles the NaNs.
    """
    df = df.copy()
    if "Income" in df.columns:
        df["Income"] = df["Income"].map(cfg.income_map)
    if "Credit Ranges" in df.columns:
        df["Credit Ranges"] = df["Credit Ranges"].map(cfg.credit_map)
    if "Number of Children" in df.columns:
        df["Number of Children"] = df["Number of Children"].map(cfg.children_map)
    return df


# ── 5. Encode binary flags ────────────────────────────────────────────────────

def encode_flags(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """
    Convert Y/N product flags → 1/0.
    Convert Member Type → PrimaryMember / AssociateMember binary columns.
    """
    df  = df.copy()
    yn  = {"Y": 1, "N": 0, "Yes": 1, "No": 0}

    # Product flags
    for raw_col, clean_name in cfg.col_to_name.items():
        if raw_col in df.columns:
            df[clean_name] = df[raw_col].map(yn)
    df = df.drop(columns=[c for c in cfg.product_cols if c in df.columns])

    # Other binary flags
    for col in ("Mail Responder", "Email Available"):
        if col in df.columns:
            df[col] = df[col].map(yn)

    if "Do Not Direct Mail Solicit" in df.columns:
        df["Do Not Direct Mail Solicit"] = pd.to_numeric(
            df["Do Not Direct Mail Solicit"], errors="coerce"
        )

    # Member type → two binary columns (one-member households will be Primary=1, Associate=0)
    if "Member Type" in df.columns:
        df["PrimaryMember"]   = (df["Member Type"] == "Primary").astype(int)
        df["AssociateMember"] = (df["Member Type"] == "Associate").astype(int)
        df = df.drop(columns=["Member Type"])

    return df


# ── 6+7. Deduplicate and join costs ──────────────────────────────────────────

def deduplicate_and_join(
    df:       pd.DataFrame,
    df_cost:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Deduplicate to one row per Member Key, then attach aggregated cost columns.
    The Member Key becomes the index after this step.
    """
    df = df.set_index("Member Key")
    df = df[~df.index.duplicated(keep="first")]

    # Drop raw Total Cost and SC Date before joining — df_cost has the
    # correctly aggregated version. Keeping both causes a column name conflict.
    cols_to_drop = [c for c in ["Total Cost", "SC Date"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    df = df.join(df_cost, how="left")

    # Fill NaN costs with 0 — NaN here means no service calls (truly 0 cost)
    for col in df_cost.columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df


# ── 8. Aggregate to household level ──────────────────────────────────────────

def aggregate_household(
    df:            pd.DataFrame,
    product_names: List[str],
) -> pd.DataFrame:
    """
    Collapse individual member rows → one row per Household Key.

    Aggregation contract (every rule is a documented business decision):
    ─────────────────────────────────────────────────────────────────────
    Categorical columns (ZIP, Mosaic)  → mode  (most common in household)
    Continuous numeric                 → mean  (household average)
    ERS call counts / costs            → sum + mean
    Product flags                      → max   (1 if ANY member holds it)
    ─────────────────────────────────────────────────────────────────────
    """
    cat_cols = [
        "ZIP", "Home Owner", "Dwelling Type",
        "Mosaic Household", "Mosaic Global Household", "kcl_B_IND_MosaicsGrouping",
    ]
    df_filled = df.fillna("Not Set")
    df_cat    = (
        df_filled.groupby("Household Key")[[c for c in cat_cols if c in df_filled.columns]]
        .agg(lambda x: x.value_counts().index[0])
    )

    # Numeric aggregations
    num_agg: Dict = {"Member Key": "count"}
    mean_cols = [
        "Length Of Residence", "Member Tenure Years",
        "Income", "Credit Ranges", "Number of Children",
        "Mail Responder", "Email Available", "Do Not Direct Mail Solicit",
    ]
    for col in mean_cols:
        if col in df.columns:
            num_agg[col] = "mean"

    for yr in (1, 2, 3):
        for pfx in ("ERS ENT Count Year", "ERS Member Cost Year"):
            col = f"{pfx} {yr}"
            if col in df.columns:
                num_agg[col] = ["sum", "mean"]

    for col in [c for c in df.columns if c.startswith("Cost ")]:
        num_agg[col] = "sum"
    if "Total Cost" in df.columns:
        num_agg["Total Cost"] = "sum"

    df_num = df.groupby("Household Key").agg(num_agg)
    df_num.columns = [" ".join(map(str, c)).strip() for c in df_num.columns.values]

    # Products: max across household members
    prods_present = [p for p in product_names if p in df.columns]
    df_prod = df.groupby("Household Key")[prods_present].max()

    return df_cat.join(df_num, how="outer").join(df_prod, how="outer")


# ── 9. Feature engineering ────────────────────────────────────────────────────

def engineer_features(
    df:            pd.DataFrame,
    product_names: List[str],
) -> pd.DataFrame:
    """
    Create domain-driven derived features.

    Each feature encodes a specific piece of business knowledge:

    total_ers_calls      — engagement with the core product (roadside)
    avg_cost_per_call    — severity index: battery vs. major tow
    product_count        — depth of current relationship
    is_multi_product     — binary: engaged vs. shallow member
    is_long_term_member  — 10+ year tenure = brand-loyal, less price-sensitive
    has_used_ers         — has experienced the core product firsthand
    is_high_income       — eligible for premium products (mortgage, home equity)
    cost_trend           — rising vs. declining utilization trajectory
    """
    df = df.copy()

    # ERS engagement
    ers_sum_cols = [c for c in df.columns if "ERS ENT Count Year" in c and "sum" in c]
    df["total_ers_calls"] = df[ers_sum_cols].sum(axis=1) if ers_sum_cols else 0.0

    # Cost severity
    cost_col = next((c for c in df.columns if "Total Cost" in c), None)
    if cost_col:
        df["avg_cost_per_call"] = df[cost_col] / (df["total_ers_calls"] + 1e-3)

    # Product depth
    prods = [p for p in product_names if p in df.columns]
    df["product_count"]    = df[prods].sum(axis=1)
    df["is_multi_product"] = (df["product_count"] >= 2).astype(int)

    # Loyalty
    tenure_col = next((c for c in df.columns if "Member Tenure Years" in c), None)
    if tenure_col:
        df["is_long_term_member"] = (df[tenure_col] >= 10).astype(int)

    # Core product experience
    df["has_used_ers"] = (df["total_ers_calls"] > 0).astype(int)

    # Income tier
    income_col = next((c for c in df.columns if "Income" in c), None)
    if income_col:
        q75 = df[income_col].quantile(0.75)
        df["is_high_income"] = (df[income_col] >= q75).astype(int)

    # Cost trajectory
    year_sum_cols = sorted([c for c in df.columns if c.startswith("Cost 2")])
    if len(year_sum_cols) >= 2:
        df["cost_trend"] = df[year_sum_cols[-1]] - df[year_sum_cols[0]]

    return df


# ── 10. Build numeric feature matrix ─────────────────────────────────────────

def build_matrix(
    df:        pd.DataFrame,
    drop_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Final step: median-impute numerics + one-hot encode categoricals.
    Returns a fully numeric DataFrame with no nulls — ready for sklearn.
    """
    df = df.copy()
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Reset index so Household Key does not become a column accidentally
    if df.index.name == "Household Key":
        df = df.reset_index(drop=True)

    # Separate numeric and categorical AFTER dropping columns
    num_cols = df.select_dtypes(include=["float64", "int64", "int32", "float32"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()

    # Drop any numeric columns that are entirely NaN — imputer cannot handle them
    valid_num_cols = [c for c in num_cols if df[c].notna().any()]
    dropped = set(num_cols) - set(valid_num_cols)
    if dropped:
        import warnings
        warnings.warn(f"Dropping all-NaN numeric columns: {dropped}")
    num_cols = valid_num_cols

    # Median imputation
    if num_cols:
        imp    = SimpleImputer(strategy="median")
        arr    = imp.fit_transform(df[num_cols])
        df_num = pd.DataFrame(arr, columns=num_cols, index=df.index)
    else:
        df_num = pd.DataFrame(index=df.index)

    # One-hot encoding
    if cat_cols:
        enc    = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ohe    = enc.fit_transform(df[cat_cols].fillna("Not Set"))
        df_cat = pd.DataFrame(
            ohe,
            columns=enc.get_feature_names_out(cat_cols),
            index=df.index,
        )
    else:
        df_cat = pd.DataFrame(index=df.index)

    return df_num.join(df_cat)


# ── Full pipeline entry point ─────────────────────────────────────────────────

def run_preprocessing(
    raw_path: Optional[str] = None,
    cfg:      Optional[ProjectConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the complete 10-step preprocessing pipeline.

    Returns
    -------
    X      : fully numeric feature matrix  (households × features)
    df_hh  : household-level DataFrame with human-readable columns
             (use for EDA, regression target extraction, cluster profiling)

    Example
    -------
    >>> from src.features.preprocessing import run_preprocessing
    >>> X, df_hh = run_preprocessing()
    >>> print(X.shape)        # (~3500, ~120)
    >>> print(df_hh.shape)    # (~3500, ~60)
    """
    if cfg is None:
        cfg = get_config()

    path = raw_path or cfg.paths.raw_data

    # Steps 1–2: load and aggregate costs (BEFORE dedup)
    df      = load_raw(path)
    df_cost = aggregate_ers_costs(df)

    # Steps 3–5: filter, encode
    df = select_and_filter(df, cfg.data)
    df = encode_ordinals(df, cfg.data)
    df = encode_flags(df, cfg.data)

    # Steps 6–7: dedup and join costs
    df = deduplicate_and_join(df, df_cost)

    # Step 8: household aggregation
    df_hh = aggregate_household(df.reset_index(), cfg.data.product_names)

    # Step 9: feature engineering
    df_hh = engineer_features(df_hh, cfg.data.product_names)

    # Step 10: build numeric matrix
    exclude = cfg.data.product_names + ["Household Key"]
    X = build_matrix(df_hh, drop_cols=exclude)

    return X, df_hh