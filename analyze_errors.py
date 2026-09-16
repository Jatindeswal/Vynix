#!/usr/bin/env python3
"""
Project Vynix — Deep Error Analysis & Performance Visualization Suite
=====================================================================
Reads vynix_hico_results.csv (600 HOI classes, 9658 test images) and generates
8 publication-quality charts plus a comprehensive text summary.

Usage:
    python analyze_errors.py
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")

try:
    import seaborn as sns
except ImportError:
    sys.exit("seaborn is required: pip install seaborn")

# ── Style ───────────────────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    try:
        plt.style.use("seaborn-whitegrid")
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "vynix_hico_results.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "analysis_outputs")
DPI = 200


def load_data():
    df = pd.read_csv(CSV_PATH)
    df["ap_pct"] = df["ap"] * 100
    df["label"] = df["verb"] + " " + df["object"]
    return df


# ════════════════════════════════════════════════════════════════════════════
#  CHART 1 — Top 20 vs Bottom 20 Classes
# ════════════════════════════════════════════════════════════════════════════
def fig1_top_bottom(df):
    fig, (ax_top, ax_bot) = plt.subplots(1, 2, figsize=(18, 10))

    top20 = df.nlargest(20, "ap").sort_values("ap")
    ax_top.barh(top20["label"], top20["ap_pct"], color="#27ae60", edgecolor="black", linewidth=0.6)
    for i, (_, row) in enumerate(top20.iterrows()):
        ax_top.text(row["ap_pct"] + 0.8, i, f"{row['ap_pct']:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax_top.set_xlabel("Average Precision (%)", fontsize=12, fontweight="bold")
    ax_top.set_title("Top 20 Best-Performing HOI Classes", fontsize=14, fontweight="bold", color="#27ae60")
    ax_top.set_xlim(0, 100)

    bot20 = df.nsmallest(20, "ap").sort_values("ap", ascending=False)
    ax_bot.barh(bot20["label"], bot20["ap_pct"], color="#e74c3c", edgecolor="black", linewidth=0.6)
    for i, (_, row) in enumerate(bot20.iterrows()):
        ax_bot.text(row["ap_pct"] + 0.05, i, f"{row['ap_pct']:.2f}%", va="center", fontsize=9, fontweight="bold")
    ax_bot.set_xlabel("Average Precision (%)", fontsize=12, fontweight="bold")
    ax_bot.set_title("Bottom 20 Worst-Performing HOI Classes", fontsize=14, fontweight="bold", color="#e74c3c")
    ax_bot.set_xlim(0, max(bot20["ap_pct"]) * 3)

    fig.suptitle("Top 20 vs Bottom 20 HOI Interaction Classes by Average Precision",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_top20_bottom20_bar.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ════════════════════════════════════════════════════════════════════════════
#  CHART 2 — AP Distribution Histogram
# ════════════════════════════════════════════════════════════════════════════
def fig2_histogram(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Full distribution
    ax1.hist(df["ap_pct"], bins=40, color="#3498db", edgecolor="black", alpha=0.8, density=True)
    try:
        sns.kdeplot(df["ap_pct"], ax=ax1, color="#e74c3c", linewidth=2, label="KDE")
    except Exception:
        pass
    mean_val = df["ap_pct"].mean()
    median_val = df["ap_pct"].median()
    ax1.axvline(mean_val, color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}%")
    ax1.axvline(median_val, color="#f39c12", linestyle="-.", linewidth=2, label=f"Median: {median_val:.2f}%")
    ax1.set_xlabel("Average Precision (%)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax1.set_title("Distribution of AP Across 600 HOI Classes", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)

    # Rare vs Non-Rare
    rare_ap = df[df["category"] == "rare"]["ap_pct"]
    nonrare_ap = df[df["category"] == "non-rare"]["ap_pct"]
    ax2.hist(nonrare_ap, bins=30, color="#2ecc71", edgecolor="black", alpha=0.6, label=f"Non-Rare (n={len(nonrare_ap)})", density=True)
    ax2.hist(rare_ap, bins=30, color="#e74c3c", edgecolor="black", alpha=0.6, label=f"Rare (n={len(rare_ap)})", density=True)
    ax2.axvline(nonrare_ap.mean(), color="#27ae60", linestyle="--", linewidth=2, label=f"Non-Rare Mean: {nonrare_ap.mean():.1f}%")
    ax2.axvline(rare_ap.mean(), color="#c0392b", linestyle="--", linewidth=2, label=f"Rare Mean: {rare_ap.mean():.1f}%")
    ax2.set_xlabel("Average Precision (%)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax2.set_title("Rare vs Non-Rare AP Distributions", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_ap_distribution_histogram.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ════════════════════════════════════════════════════════════════════════════
#  CHART 3 — Verb × Object Heatmap
# ════════════════════════════════════════════════════════════════════════════
def fig3_heatmap(df):
    verb_counts = df["verb"].value_counts()
    obj_counts = df["object"].value_counts()
    top_verbs = verb_counts.head(25).index.tolist()
    top_objs = obj_counts.head(15).index.tolist()

    sub = df[df["verb"].isin(top_verbs) & df["object"].isin(top_objs)]
    pivot = sub.pivot_table(index="verb", columns="object", values="ap_pct", aggfunc="mean")
    pivot = pivot.reindex(index=top_verbs, columns=top_objs)

    fig, ax = plt.subplots(figsize=(20, 14))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", linewidths=0.5,
                ax=ax, cbar_kws={"label": "AP (%)"}, vmin=0, vmax=80,
                annot_kws={"fontsize": 8})
    ax.set_title("Verb × Object Performance Heatmap (AP %)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Object Category", fontsize=13, fontweight="bold")
    ax.set_ylabel("Verb Category", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_verb_performance_heatmap.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ════════════════════════════════════════════════════════════════════════════
#  CHART 4 — Average AP per Object Category
# ════════════════════════════════════════════════════════════════════════════
def fig4_object_ap(df):
    obj_ap = df.groupby("object")["ap_pct"].mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(14, 18))
    colors = ["#e74c3c" if v < 10 else "#f39c12" if v < 25 else "#27ae60" for v in obj_ap.values]
    ax.barh(obj_ap.index, obj_ap.values, color=colors, edgecolor="black", linewidth=0.5)
    for i, (name, val) in enumerate(obj_ap.items()):
        ax.text(val + 0.5, i, f"{val:.1f}%", va="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Mean Average Precision (%)", fontsize=12, fontweight="bold")
    ax.set_title("Average Precision by Object Category (80 Objects)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(obj_ap.values) * 1.15)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_object_avg_ap.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ════════════════════════════════════════════════════════════════════════════
#  CHART 5 — Average AP per Verb Category
# ════════════════════════════════════════════════════════════════════════════
def fig5_verb_ap(df):
    verb_ap = df.groupby("verb")["ap_pct"].mean().sort_values(ascending=True)
    top40 = verb_ap.tail(40)

    fig, ax = plt.subplots(figsize=(14, 14))
    colors = ["#e74c3c" if v < 10 else "#f39c12" if v < 25 else "#27ae60" for v in top40.values]
    ax.barh(top40.index, top40.values, color=colors, edgecolor="black", linewidth=0.5)
    for i, (name, val) in enumerate(top40.items()):
        ax.text(val + 0.5, i, f"{val:.1f}%", va="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Mean Average Precision (%)", fontsize=12, fontweight="bold")
    ax.set_title("Average Precision by Verb Category (Top 40 Verbs)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(top40.values) * 1.15)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig5_verb_avg_ap.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ════════════════════════════════════════════════════════════════════════════
#  CHART 6 — Rare vs Non-Rare Violin + Strip Plot
# ════════════════════════════════════════════════════════════════════════════
def fig6_violin(df):
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.violinplot(x="category", y="ap_pct", data=df, inner=None, palette={"rare": "#e74c3c", "non-rare": "#2ecc71"},
                   alpha=0.4, ax=ax, order=["rare", "non-rare"])
    sns.stripplot(x="category", y="ap_pct", data=df, jitter=0.25, alpha=0.35, size=3,
                  palette={"rare": "#c0392b", "non-rare": "#27ae60"}, ax=ax, order=["rare", "non-rare"])

    rare_mean = df[df["category"] == "rare"]["ap_pct"].mean()
    nonrare_mean = df[df["category"] == "non-rare"]["ap_pct"].mean()
    ax.axhline(rare_mean, color="#c0392b", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axhline(nonrare_mean, color="#27ae60", linestyle="--", linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Category Split", fontsize=13, fontweight="bold")
    ax.set_ylabel("Average Precision (%)", fontsize=13, fontweight="bold")
    ax.set_title(f"Rare (mean={rare_mean:.1f}%) vs Non-Rare (mean={nonrare_mean:.1f}%) AP Distribution",
                 fontsize=14, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig6_rare_vs_nonrare_boxplot.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ════════════════════════════════════════════════════════════════════════════
#  CHART 7 — Failure Mode Pie Chart
# ════════════════════════════════════════════════════════════════════════════
def fig7_pie(df):
    bins = [
        ("Strong (AP > 50%)", (df["ap_pct"] > 50).sum()),
        ("Moderate (20–50%)", ((df["ap_pct"] > 20) & (df["ap_pct"] <= 50)).sum()),
        ("Weak (5–20%)", ((df["ap_pct"] > 5) & (df["ap_pct"] <= 20)).sum()),
        ("Critical (< 5%)", (df["ap_pct"] <= 5).sum()),
    ]
    labels = [f"{name}\n({count} classes)" for name, count in bins]
    sizes = [count for _, count in bins]
    colors = ["#27ae60", "#f1c40f", "#e67e22", "#e74c3c"]
    explode = (0.03, 0.03, 0.03, 0.08)

    fig, ax = plt.subplots(figsize=(10, 10))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", startangle=140,
        colors=colors, explode=explode, textprops={"fontsize": 12},
        pctdistance=0.78, labeldistance=1.12
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(13)
    ax.set_title("Performance Tier Distribution (600 HOI Classes)", fontsize=15, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig7_failure_mode_pie.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ════════════════════════════════════════════════════════════════════════════
#  CHART 8 — Cumulative AP Curve
# ════════════════════════════════════════════════════════════════════════════
def fig8_cumulative(df):
    sorted_ap = np.sort(df["ap_pct"].values)
    cumulative = np.cumsum(sorted_ap)
    total = cumulative[-1]
    x = np.arange(1, len(sorted_ap) + 1)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(x, cumulative, alpha=0.3, color="#3498db")
    ax.plot(x, cumulative, color="#2c3e50", linewidth=2)

    # Mark 50th percentile
    half_idx = np.searchsorted(cumulative, total / 2)
    ax.axvline(half_idx, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"50% of total AP reached at class #{half_idx}")
    ax.axhline(total / 2, color="#e74c3c", linestyle=":", linewidth=1, alpha=0.5)

    ax.set_xlabel("HOI Classes (sorted by AP ascending)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative AP (%)", fontsize=12, fontweight="bold")
    ax.set_title("Cumulative Average Precision Curve (600 Classes)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(1, 600)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig8_cumulative_ap.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ════════════════════════════════════════════════════════════════════════════
#  TEXT SUMMARY
# ════════════════════════════════════════════════════════════════════════════
def print_summary(df):
    rare = df[df["category"] == "rare"]
    nonrare = df[df["category"] == "non-rare"]

    verb_mean = df.groupby("verb")["ap_pct"].mean().sort_values(ascending=False)
    obj_mean = df.groupby("object")["ap_pct"].mean().sort_values(ascending=False)

    print("\n" + "=" * 72)
    print("  📊 DEEP ERROR ANALYSIS SUMMARY — PROJECT VYNIX (HICO-DET)")
    print("=" * 72)

    print(f"\n  Total HOI Classes        : {len(df)}")
    print(f"  Mean AP                  : {df['ap_pct'].mean():.2f}%")
    print(f"  Median AP                : {df['ap_pct'].median():.2f}%")
    print(f"  Std AP                   : {df['ap_pct'].std():.2f}%")
    print(f"  Min AP                   : {df['ap_pct'].min():.3f}%")
    print(f"  Max AP                   : {df['ap_pct'].max():.2f}%")

    print(f"\n  Rare Mean AP             : {rare['ap_pct'].mean():.2f}%  ({len(rare)} classes)")
    print(f"  Non-Rare Mean AP         : {nonrare['ap_pct'].mean():.2f}%  ({len(nonrare)} classes)")

    print(f"\n  Classes with AP > 50%    : {(df['ap_pct'] > 50).sum()}")
    print(f"  Classes with AP > 30%    : {(df['ap_pct'] > 30).sum()}")
    print(f"  Classes with AP 5–20%    : {((df['ap_pct'] > 5) & (df['ap_pct'] <= 20)).sum()}")
    print(f"  Classes with AP < 5%     : {(df['ap_pct'] <= 5).sum()}")
    print(f"  Classes with AP < 1%     : {(df['ap_pct'] < 1).sum()}")

    print("\n  ── Top 5 Verbs by Mean AP ──")
    for v, ap in verb_mean.head(5).items():
        print(f"     {v:<20s} {ap:6.2f}%")

    print("\n  ── Bottom 5 Verbs by Mean AP ──")
    for v, ap in verb_mean.tail(5).items():
        print(f"     {v:<20s} {ap:6.2f}%")

    print("\n  ── Top 5 Objects by Mean AP ──")
    for o, ap in obj_mean.head(5).items():
        print(f"     {o:<20s} {ap:6.2f}%")

    print("\n  ── Bottom 5 Objects by Mean AP ──")
    for o, ap in obj_mean.tail(5).items():
        print(f"     {o:<20s} {ap:6.2f}%")

    print("\n" + "=" * 72)

    # Also write a text report file
    report_path = os.path.join(OUT_DIR, "error_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("PROJECT VYNIX — DEEP ERROR ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total HOI Classes: {len(df)}\n")
        f.write(f"Mean AP: {df['ap_pct'].mean():.2f}%\n")
        f.write(f"Median AP: {df['ap_pct'].median():.2f}%\n")
        f.write(f"Rare Mean AP: {rare['ap_pct'].mean():.2f}% ({len(rare)} classes)\n")
        f.write(f"Non-Rare Mean AP: {nonrare['ap_pct'].mean():.2f}% ({len(nonrare)} classes)\n\n")

        f.write("TOP 20 BEST CLASSES:\n")
        for _, row in df.nlargest(20, "ap").iterrows():
            f.write(f"  {row['verb']:<15s} {row['object']:<15s}  AP={row['ap_pct']:6.2f}%  ({row['category']})\n")

        f.write("\nBOTTOM 20 WORST CLASSES:\n")
        for _, row in df.nsmallest(20, "ap").iterrows():
            f.write(f"  {row['verb']:<15s} {row['object']:<15s}  AP={row['ap_pct']:6.3f}%  ({row['category']})\n")

        f.write("\nFAILURE MODE BREAKDOWN:\n")
        f.write(f"  Strong  (AP > 50%):  {(df['ap_pct'] > 50).sum()} classes\n")
        f.write(f"  Moderate (20-50%):   {((df['ap_pct'] > 20) & (df['ap_pct'] <= 50)).sum()} classes\n")
        f.write(f"  Weak     (5-20%):    {((df['ap_pct'] > 5) & (df['ap_pct'] <= 20)).sum()} classes\n")
        f.write(f"  Critical (< 5%):     {(df['ap_pct'] <= 5).sum()} classes\n")

    print(f"\n  ✓ Text report saved to {report_path}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading results from {CSV_PATH}...")
    df = load_data()
    print(f"✓ Loaded {len(df)} HOI classes.\n")
    print("Generating charts...")

    fig1_top_bottom(df)
    fig2_histogram(df)
    fig3_heatmap(df)
    fig4_object_ap(df)
    fig5_verb_ap(df)
    fig6_violin(df)
    fig7_pie(df)
    fig8_cumulative(df)

    print_summary(df)
    print(f"\n✓ All 8 charts saved to {OUT_DIR}/")
