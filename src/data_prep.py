from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class OnlineRetailSheets:
    year_2009_2010: str = "Year 2009-2010"
    year_2010_2011: str = "Year 2010-2011"


def load_online_retail_excel(
    path: str,
    *,
    sheets: OnlineRetailSheets = OnlineRetailSheets(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Online Retail II dataset (two sheets) from an Excel file."""
    df1 = pd.read_excel(path, sheet_name=sheets.year_2009_2010)
    df2 = pd.read_excel(path, sheet_name=sheets.year_2010_2011)
    return df1, df2


def standardize_year_df(
    df: pd.DataFrame,
    *,
    fill_description: bool = True,
) -> pd.DataFrame:
    """Standardize a single year DataFrame to a common schema used in the project."""
    out = df.rename(columns={"Customer ID": "CustomerID"}).copy()
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"], errors="coerce")

    # Drop rows where CustomerID or InvoiceDate is missing
    # CustomerID and the invoice date are the only columns that are required for the RFM table
    out = out.dropna(subset=["CustomerID", "InvoiceDate"]).copy()

    if fill_description and "Description" in out.columns:
        out["Description"] = out["Description"].fillna("Unknown")

    return out


def build_transactions(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    drop_duplicates: bool = True,
    duplicate_subset: Optional[list[str]] = None,
    remove_cancellations: bool = True,
    require_positive_qty_price: bool = True,
) -> pd.DataFrame:
    """Merge two year DataFrames and clean to a transaction-level table for RFM."""
    if duplicate_subset is None:
        duplicate_subset = ["Invoice", "StockCode", "Quantity", "InvoiceDate", "CustomerID"]

    df = pd.concat([df1, df2], ignore_index=True)

    if drop_duplicates:
        df = df.drop_duplicates(subset=duplicate_subset, keep="first").copy()
    # generally, cancelled invoices are marked with a C prefix
    if remove_cancellations:
        df = df[~df["Invoice"].astype(str).str.startswith("C", na=False)].copy()

    if require_positive_qty_price:
        df = df[(df["Quantity"] > 0) & (df["Price"] > 0)].copy()

    df["TotalPrice"] = df["Quantity"] * df["Price"]
    return df


def build_rfm_table(
    df: pd.DataFrame,
    *,
    reference_date: Optional[pd.Timestamp] = None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Build the base RFM table (recency/frequency/monetary) from transactions."""
    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            frequency=("Invoice", "nunique"),
            monetary=("TotalPrice", "sum"),
        )
        .reset_index()
    )

    return rfm, reference_date

