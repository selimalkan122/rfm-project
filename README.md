# RFM Customer Segmentation and CLV Analysis

A Jupyter project that analyses customer behaviour using the UCI **Online Retail II** dataset: RFM (Recency, Frequency, Monetary) metrics, rule-based segments, a simple CLV estimate, and K-Means clustering.

## Features

- **Data preparation**: Merge two yearly Excel sheets, remove cancellations and duplicates, and build line-level `TotalPrice` (`src/data_prep.py`).
- **RFM table**: Per-customer recency, frequency (unique invoice count), and monetary value.
- **Segmentation**: Rule-based segment labels from R/F/M scores and segment summaries (`src/segmentation.py`).
- **CLV**: Daily purchase rate from the observation window (tenure) and an approximate 12-month CLV (`src/clv.py`).
- **Clustering**: K-Means on log-transformed, scaled RFM features; choose *k* using inertia and silhouette (`src/clustering.py`).
- **Visualisation**: Segment overview and RFM distributions (`src/plots.py`).

Main notebook: **`Analysis_upgraded.ipynb`**.

## Dataset

The notebook expects the following file at the project root:

| File | Description |
|------|-------------|
| `online_retail_II.xlsx` | UCI Online Retail II (two sheets: `Year 2009-2010`, `Year 2010-2011`) |

Download the data from the [UCI Machine Learning Repository — Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place the Excel file in this project folder.

## Requirements

- Python 3.10+
- Main libraries: `pandas`, `numpy`, `openpyxl` (Excel), `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, Jupyter (`ipykernel` / notebook).

Example setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy openpyxl scikit-learn matplotlib seaborn scipy jupyter ipykernel
```

## Running

Open `Analysis_upgraded.ipynb` in VS Code / Cursor and run **Run All**. The working directory must be the project root so `from src....` imports resolve.

## Project layout

```
RFM project/
├── Analysis_upgraded.ipynb   # Main analysis
├── online_retail_II.xlsx     # Data (you must add this)
├── README.md
└── src/
    ├── data_prep.py          # Load Excel, cleaning, RFM base table
    ├── segmentation.py       # Segment rules and summaries
    ├── clv.py                # CLV columns
    ├── clustering.py         # K-Means prep and k diagnostics
    └── plots.py              # Charts
```

## License and source

- Dataset: UCI Online Retail II; see the UCI page for terms of use.
- Code and notes in this repo are for learning and analysis; validate results and add business rules before production use.
