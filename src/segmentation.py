from __future__ import annotations

import pandas as pd


def rfm_segment_name(row: pd.Series) -> str:
    """Rule-based RFM segment label based on R/F/M scores (1..5)."""
    r, f, m = row["R_score"], row["F_score"], row["M_score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 4 and f >= 4:
        return "Loyal customers"
    if r >= 4 and f <= 2:
        return "New customers"
    if r <= 2 and f >= 4 and m >= 4:
        return "Cannot lose them"
    if r <= 2 and f >= 3:
        return "At risk"
    if r <= 2 and f <= 2:
        return "Hibernating"
    if r >= 3 and f >= 3:
        return "Potential loyalists"
    if r >= 4:
        return "Promising"
    return "Need attention"


def add_segment_labels(rfm: pd.DataFrame) -> pd.DataFrame:
    """Add `Segment` column to an RFM dataframe that already has R/F/M scores."""
    out = rfm.copy()
    out["Segment"] = out.apply(rfm_segment_name, axis=1)
    return out


def build_segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Return customer count + revenue summary per segment."""
    if "CustomerID" not in rfm.columns:
        tmp = rfm.reset_index()
    else:
        tmp = rfm

    return (
        tmp.groupby("Segment", as_index=False)
        .agg(customers=("CustomerID", "count"), revenue=("monetary", "sum"))
        .sort_values("revenue", ascending=False)
    )

