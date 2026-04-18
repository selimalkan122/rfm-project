from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def add_clv_tenure(
    rfm: pd.DataFrame,
    df: pd.DataFrame,
    *,
    horizon_months: int = 12,
    min_obs_days: int = 30,
    gross_margin_rate: Optional[float] = None,
) -> pd.DataFrame:
    """Add a simple CLV estimate using an observation-window purchase rate.

    Uses customer tenure (how long we've observed the customer) to estimate purchase cadence.
    This is typically more stable than a recency-based proxy like frequency/(recency+1).
    """
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    horizon_days = horizon_months * 30

    cust_first_purchase = (
        df.groupby("CustomerID")["InvoiceDate"].min().rename("first_purchase")
    )

    # Ensure CustomerID is a column for merge
    if "CustomerID" not in rfm.columns:
        out = rfm.reset_index()
    else:
        out = rfm.copy()

    out = (
        out.merge(cust_first_purchase, on="CustomerID", how="left")
        .set_index("CustomerID")
    )

    # Defensive: avoid divide-by-zero
    freq = out["frequency"].replace(0, np.nan)
    out["avg_order_value"] = (out["monetary"] / freq).fillna(0)

    obs_days = (reference_date - out["first_purchase"]).dt.days + 1
    obs_days = obs_days.clip(lower=min_obs_days).fillna(min_obs_days)
    out["obs_days"] = obs_days

    out["purchase_rate_daily"] = (out["frequency"] / obs_days).fillna(0)

    out["CLV_12M"] = (
        out["avg_order_value"] * out["purchase_rate_daily"] * horizon_days
    ).round(2)

    if gross_margin_rate is not None:
        out["CLV_12M_GM"] = (out["CLV_12M"] * gross_margin_rate).round(2)

    # region agent log
    import json
    import time
    from pathlib import Path

    _log_path = Path(__file__).resolve().parent.parent / "debug-70aa94.log"
    _payload = {
        "sessionId": "70aa94",
        "runId": "pre-fix",
        "hypothesisId": "A",
        "location": "clv.py:add_clv_tenure:before_return",
        "message": "CustomerID column vs index after CLV merge",
        "data": {
            "CustomerID_in_columns": bool("CustomerID" in out.columns),
            "index_name": out.index.name,
            "n_rows": int(len(out)),
        },
        "timestamp": int(time.time() * 1000),
    }
    with open(_log_path, "a", encoding="utf-8") as _f:
        _f.write(json.dumps(_payload) + "\n")
    # endregion

    return out

