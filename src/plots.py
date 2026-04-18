from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


def plot_segment_overview(segment_summary: pd.DataFrame, *, palette: list[str]) -> None:
    """3-panel segment overview: customer counts, revenue, revenue share."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("RFM Segment Overview", fontsize=15, fontweight="bold", y=1.02)

    seg = segment_summary.sort_values("customers", ascending=True)
    axes[0].barh(seg["Segment"], seg["customers"], color=palette[: len(seg)])
    axes[0].set_xlabel("Number of Customers")
    axes[0].set_title("Customers per Segment")
    for bar, val in zip(axes[0].patches, seg["customers"]):
        axes[0].text(
            bar.get_width() + 5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center",
            fontsize=8,
        )

    seg2 = segment_summary.sort_values("revenue", ascending=True)
    axes[1].barh(seg2["Segment"], seg2["revenue"] / 1e6, color=palette[: len(seg2)])
    axes[1].set_xlabel("Revenue (M £)")
    axes[1].set_title("Revenue per Segment")
    for bar, val in zip(axes[1].patches, seg2["revenue"]):
        axes[1].text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"£{val/1e6:.2f}M",
            va="center",
            fontsize=8,
        )

    axes[2].pie(
        segment_summary["revenue"],
        labels=segment_summary["Segment"],
        autopct="%1.1f%%",
        colors=palette[: len(segment_summary)],
        startangle=140,
        textprops={"fontsize": 8},
    )
    axes[2].set_title("Revenue Share")

    plt.tight_layout()
    plt.savefig("segment_overview.png", bbox_inches="tight")
    plt.show()


def plot_rfm_distributions(rfm: pd.DataFrame) -> None:
    """Histogram distributions for Recency, Frequency, Monetary."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("RFM Raw Metric Distributions", fontsize=13, fontweight="bold")

    cols = ["recency", "frequency", "monetary"]
    titles = ["Recency (days)", "Frequency (# orders)", "Monetary (£)"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, col, title, color in zip(axes, cols, titles, colors):
        data = rfm[col].clip(upper=rfm[col].quantile(0.99))
        ax.hist(data, bins=50, color=color, edgecolor="white", linewidth=0.4)
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.tight_layout()
    plt.savefig("rfm_distributions.png", bbox_inches="tight")
    plt.show()


def plot_rfm_heatmap(rfm: pd.DataFrame) -> None:
    """Heatmap of avg Monetary by R_score (y) and F_score (x)."""
    pivot = rfm.pivot_table(values="monetary", index="R_score", columns="F_score", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Avg Monetary (£)"},
    )
    ax.set_title("Avg Monetary by R-Score × F-Score", fontsize=13, fontweight="bold")
    ax.set_xlabel("Frequency Score")
    ax.set_ylabel("Recency Score")
    plt.tight_layout()
    plt.savefig("rfm_heatmap.png", bbox_inches="tight")
    plt.show()


def plot_scatter_rf(rfm: pd.DataFrame, *, palette: list[str]) -> None:
    """Scatter: Recency vs Frequency, coloured by Segment."""
    fig, ax = plt.subplots(figsize=(10, 6))
    segments = rfm["Segment"].unique()
    cmap = dict(zip(segments, palette[: len(segments)]))

    for seg, grp in rfm.groupby("Segment"):
        ax.scatter(
            grp["recency"],
            grp["frequency"],
            label=seg,
            alpha=0.55,
            s=18,
            color=cmap.get(seg, "#999"),
        )

    ax.set_xlabel("Recency (days since last purchase)")
    ax.set_ylabel("Frequency (# orders)")
    ax.set_title("Customer Map: Recency vs Frequency", fontsize=13, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig("scatter_rf.png", bbox_inches="tight")
    plt.show()


def plot_elbow_silhouette(results: dict) -> int:
    """Plot Elbow and Silhouette curves; return best k (max silhouette)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(results["k_values"], results["inertias"], marker="o", color="#4C72B0", linewidth=2)
    ax1.set_title("Elbow Method", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(1))

    best_k = results["k_values"][int(np.argmax(results["silhouettes"]))]
    ax2.plot(results["k_values"], results["silhouettes"], marker="s", color="#DD8452", linewidth=2)
    ax2.axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
    ax2.set_title("Silhouette Score", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.legend()
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(1))

    plt.tight_layout()
    plt.savefig("elbow_silhouette.png", bbox_inches="tight")
    plt.show()
    print(f"Recommended k = {best_k}  (highest silhouette)")
    return int(best_k)

