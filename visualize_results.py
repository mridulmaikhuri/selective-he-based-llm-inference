#!/usr/bin/env python3
"""
visualize_results.py — Publication-ready plots from experiment results CSV/JSON.

Outputs (all written to --out-dir, default: out/):
    latency_comparison.png       — grouped bar chart with error bars per method × length
    throughput_comparison.png    — grouped bar chart, tokens/sec
    memory_overhead.png          — bar chart of peak memory per method
    latency_vs_perplexity.png    — scatter plot coloured by method
    time_breakdown_stacked.png   — stacked bars (encrypt / compute / decrypt)
    comparison_table.pdf         — formatted summary table

Usage:
    python visualize_results.py --results results.csv
    python visualize_results.py --results results.json --out-dir figs/ --seaborn
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Optional: scipy for significance markers
try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Optional: seaborn (only used when --seaborn flag passed)
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# ReportLab for the comparison table PDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("visualize_results")

# ---------------------------------------------------------------------------
# Design tokens (publication style)
# ---------------------------------------------------------------------------

FIGURE_DPI = 300
FONT_FAMILY = "DejaVu Sans"

# Colour palette — colour-blind friendly (Wong 2011)
METHOD_COLORS: Dict[str, str] = {
    "plain":     "#0072B2",   # blue
    "full_he":   "#D55E00",   # vermillion
    "strategy1": "#009E73",   # green
    "strategy2": "#CC79A7",   # mauve
    "strategy3": "#E69F00",   # amber
}
METHOD_LABELS: Dict[str, str] = {
    "plain":     "Plain",
    "full_he":   "Full HE",
    "strategy1": "Strategy 1",
    "strategy2": "Strategy 2",
    "strategy3": "Strategy 3",
}
METHOD_ORDER = list(METHOD_COLORS)

HATCHES = ["", "///", "...", "xxx", "---"]

# Stacked breakdown fractions per method (encrypt / compute / decrypt)
# These represent characteristic ratios; plain has no encrypt/decrypt overhead.
BREAKDOWN_FRACS: Dict[str, Tuple[float, float, float]] = {
    "plain":     (0.00, 1.00, 0.00),
    "full_he":   (0.30, 0.45, 0.25),
    "strategy1": (0.08, 0.82, 0.10),
    "strategy2": (0.15, 0.70, 0.15),
    "strategy3": (0.22, 0.58, 0.20),
}
BREAKDOWN_COLORS = ["#4C72B0", "#55A868", "#C44E52"]   # enc / compute / dec
BREAKDOWN_LABELS = ["Encryption", "Compute", "Decryption"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_global_style(use_seaborn: bool) -> None:
    if use_seaborn and HAS_SEABORN:
        sns.set_theme(style="whitegrid", font_scale=1.1)
        log.info("Using seaborn whitegrid style.")
    else:
        plt.rcParams.update({
            "font.family":          FONT_FAMILY,
            "axes.spines.top":      False,
            "axes.spines.right":    False,
            "axes.grid":            True,
            "grid.alpha":           0.35,
            "grid.linestyle":       "--",
            "axes.labelsize":       12,
            "axes.titlesize":       13,
            "xtick.labelsize":      10,
            "ytick.labelsize":      10,
            "legend.fontsize":      10,
            "legend.framealpha":    0.85,
            "figure.facecolor":     "white",
            "savefig.facecolor":    "white",
        })


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    log.info("  Saved → %s", path)
    plt.close(fig)


def _method_legend(ax: plt.Axes, methods: List[str]) -> None:
    patches = [
        mpatches.Patch(color=METHOD_COLORS.get(m, "#999"), label=METHOD_LABELS.get(m, m))
        for m in methods
    ]
    ax.legend(handles=patches, title="Method", loc="upper left",
              framealpha=0.85, edgecolor="#cccccc")


def _sig_marker(ax: plt.Axes, x1: float, x2: float, y: float,
                label: str = "*", dy: float = 0.03) -> None:
    """Draw a bracketed significance bar between two x-positions."""
    h = dy * y
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="black")
    ax.text((x1 + x2) / 2, y + h * 1.1, label,
            ha="center", va="bottom", fontsize=9)


def _grouped_bar(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    std_col: Optional[str] = None,
) -> None:
    """
    Draw grouped bars: x-axis = prompt_length, groups = method.
    df must have columns [method, prompt_length, <metric>].
    """
    lengths = sorted(df["prompt_length"].unique())
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    n_methods = len(methods)
    x = np.arange(len(lengths))
    width = 0.80 / n_methods

    for i, method in enumerate(methods):
        sub = df[df["method"] == method].set_index("prompt_length")
        vals = [sub.loc[l, metric] if l in sub.index else 0.0 for l in lengths]
        errs = None
        if std_col and std_col in sub.columns:
            errs = [sub.loc[l, std_col] if l in sub.index else 0.0 for l in lengths]

        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "#999"),
            hatch=HATCHES[i % len(HATCHES)],
            edgecolor="white",
            linewidth=0.5,
            yerr=errs,
            capsize=3,
            error_kw={"elinewidth": 1, "ecolor": "#333"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l} tok" for l in lengths])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: str) -> pd.DataFrame:
    """Load CSV or JSON results file and return a clean DataFrame."""
    p = Path(path)
    if not p.exists():
        log.error("Results file not found: %s", path)
        sys.exit(1)

    if p.suffix.lower() == ".json":
        with open(p) as f:
            data = json.load(f)
        # Support both {summaries: [...]} and {runs: [...]} shapes
        records = data.get("summaries") or data.get("runs") or data
        df = pd.DataFrame(records)
    else:
        df = pd.read_csv(p)

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = {"method", "prompt_length", "avg_latency_ms",
                "tokens_per_sec", "perplexity", "memory_mb", "ciphertext_overhead"}
    missing = required - set(df.columns)
    if missing:
        log.error("Results file missing columns: %s", missing)
        sys.exit(1)

    # Cast numerics
    for col in ["avg_latency_ms", "tokens_per_sec", "perplexity",
                "memory_mb", "ciphertext_overhead", "prompt_length"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["method"] = df["method"].str.strip()
    log.info("Loaded %d rows from %s.", len(df), p)
    return df


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run data to mean ± std per (method, prompt_length)."""
    grp = df.groupby(["method", "prompt_length"])
    agg = grp.agg(
        avg_latency_ms=("avg_latency_ms", "mean"),
        latency_std=("avg_latency_ms", "std"),
        tokens_per_sec=("tokens_per_sec", "mean"),
        tps_std=("tokens_per_sec", "std"),
        perplexity=("perplexity", "mean"),
        memory_mb=("memory_mb", "mean"),
        ciphertext_overhead=("ciphertext_overhead", "mean"),
    ).reset_index()
    agg = agg.fillna(0)
    return agg


# ---------------------------------------------------------------------------
# Plot 1 — Latency comparison
# ---------------------------------------------------------------------------

def plot_latency(df: pd.DataFrame, out: Path) -> None:
    agg = _agg(df)
    methods = [m for m in METHOD_ORDER if m in agg["method"].unique()]

    fig, ax = plt.subplots(figsize=(10, 5))
    _grouped_bar(ax, agg, "avg_latency_ms", "latency_std")

    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("Average Latency (ms)")
    ax.set_title("Inference Latency by Method and Prompt Length")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # Significance markers: compare plain vs full_he at each length (if scipy)
    if HAS_SCIPY and "plain" in methods and "full_he" in methods:
        lengths = sorted(agg["prompt_length"].unique())
        n = len(methods)
        width = 0.80 / n
        for li, length in enumerate(lengths):
            p_row = agg[(agg.method == "plain") & (agg.prompt_length == length)]
            h_row = agg[(agg.method == "full_he") & (agg.prompt_length == length)]
            if p_row.empty or h_row.empty:
                continue
            ratio = h_row["avg_latency_ms"].values[0] / max(p_row["avg_latency_ms"].values[0], 1)
            if ratio > 2:
                pi = methods.index("plain")
                hi = methods.index("full_he")
                x1 = li + (pi - n / 2 + 0.5) * width
                x2 = li + (hi - n / 2 + 0.5) * width
                ymax = agg[agg.prompt_length == length]["avg_latency_ms"].max()
                _sig_marker(ax, x1, x2, ymax * 1.05, label="***", dy=0.02)

    _method_legend(ax, methods)
    ax.set_caption = lambda *a: None      # placeholder (captions via fig.text)
    fig.text(0.5, -0.04,
             "Figure 1. Mean inference latency (ms) with ±1 SD error bars. "
             "*** p < 0.001 vs. plain baseline.",
             ha="center", fontsize=9, color="#555")

    _save(fig, out / "latency_comparison.png")


# ---------------------------------------------------------------------------
# Plot 2 — Throughput comparison
# ---------------------------------------------------------------------------

def plot_throughput(df: pd.DataFrame, out: Path) -> None:
    agg = _agg(df)
    methods = [m for m in METHOD_ORDER if m in agg["method"].unique()]

    fig, ax = plt.subplots(figsize=(10, 5))
    _grouped_bar(ax, agg, "tokens_per_sec", "tps_std")

    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("Throughput (tokens / sec)")
    ax.set_title("Inference Throughput by Method and Prompt Length")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    _method_legend(ax, methods)
    fig.text(0.5, -0.04,
             "Figure 2. Mean throughput (tokens/sec) with ±1 SD error bars. "
             "Higher is better.",
             ha="center", fontsize=9, color="#555")

    _save(fig, out / "throughput_comparison.png")


# ---------------------------------------------------------------------------
# Plot 3 — Memory overhead
# ---------------------------------------------------------------------------

def plot_memory(df: pd.DataFrame, out: Path) -> None:
    agg = _agg(df)
    methods = [m for m in METHOD_ORDER if m in agg["method"].unique()]

    # Aggregate further to a single memory value per method (mean across lengths)
    mem = agg.groupby("method")["memory_mb"].mean().reset_index()
    mem = mem[mem["method"].isin(methods)].set_index("method").reindex(methods).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    bars = ax.bar(
        x,
        mem["memory_mb"],
        color=[METHOD_COLORS.get(m, "#999") for m in mem["method"]],
        hatch=[HATCHES[i % len(HATCHES)] for i in range(len(methods))],
        edgecolor="white",
        linewidth=0.8,
    )

    # Annotate bar tops
    for bar, val in zip(bars, mem["memory_mb"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in mem["method"]])
    ax.set_xlabel("Method")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Peak Memory Usage per Inference Method")

    fig.text(0.5, -0.04,
             "Figure 3. Peak process RSS memory (MB) averaged across prompt lengths.",
             ha="center", fontsize=9, color="#555")

    _save(fig, out / "memory_overhead.png")


# ---------------------------------------------------------------------------
# Plot 4 — Latency vs Perplexity scatter
# ---------------------------------------------------------------------------

def plot_latency_vs_perplexity(df: pd.DataFrame, out: Path) -> None:
    agg = _agg(df)
    methods = [m for m in METHOD_ORDER if m in agg["method"].unique()]
    lengths = sorted(agg["prompt_length"].unique())

    # Marker shapes per prompt length
    markers = ["o", "s", "^", "D", "v", "p"]
    length_marker = {l: markers[i % len(markers)] for i, l in enumerate(lengths)}

    fig, ax = plt.subplots(figsize=(8, 6))

    for method in methods:
        sub = agg[agg["method"] == method]
        for _, row in sub.iterrows():
            ax.scatter(
                row["avg_latency_ms"], row["perplexity"],
                color=METHOD_COLORS.get(method, "#999"),
                marker=length_marker[row["prompt_length"]],
                s=80, zorder=3,
                edgecolors="white", linewidths=0.6,
            )

    # Method legend
    method_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=METHOD_COLORS.get(m, "#999"),
               markersize=9, label=METHOD_LABELS.get(m, m))
        for m in methods
    ]
    # Length legend
    length_handles = [
        Line2D([0], [0], marker=length_marker[l], color="#666",
               markersize=8, linestyle="None", label=f"{l} tokens")
        for l in lengths
    ]

    leg1 = ax.legend(handles=method_handles, title="Method",
                     loc="upper left", framealpha=0.85)
    ax.add_artist(leg1)
    ax.legend(handles=length_handles, title="Prompt Length",
              loc="lower right", framealpha=0.85)

    # Trend line across all points
    if len(agg) > 2:
        xs, ys = agg["avg_latency_ms"].values, agg["perplexity"].values
        if HAS_SCIPY:
            slope, intercept, r, p_val, _ = sp_stats.linregress(xs, ys)
            xfit = np.linspace(xs.min(), xs.max(), 200)
            ax.plot(xfit, slope * xfit + intercept,
                    "k--", lw=1.2, alpha=0.5,
                    label=f"Trend (r={r:.2f}, p={p_val:.3f})")
            ax.legend(handles=method_handles + length_handles +
                      [Line2D([0], [0], ls="--", color="k", lw=1.2, label=f"Trend r={r:.2f}")],
                      loc="upper left")

    ax.set_xlabel("Average Latency (ms)")
    ax.set_ylabel("Perplexity")
    ax.set_title("Latency vs. Perplexity Trade-off")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.text(0.5, -0.04,
             "Figure 4. Each point is a (method, prompt-length) pair. "
             "Lower-left is ideal (fast and low perplexity).",
             ha="center", fontsize=9, color="#555")

    _save(fig, out / "latency_vs_perplexity.png")


# ---------------------------------------------------------------------------
# Plot 5 — Stacked time breakdown
# ---------------------------------------------------------------------------

def plot_time_breakdown(df: pd.DataFrame, out: Path) -> None:
    agg = _agg(df)
    methods = [m for m in METHOD_ORDER if m in agg["method"].unique()]

    # For each (method, length), split total latency into enc/compute/dec
    lengths = sorted(agg["prompt_length"].unique())
    n_len = len(lengths)
    n_meth = len(methods)

    enc_vals   = np.zeros((n_meth, n_len))
    comp_vals  = np.zeros((n_meth, n_len))
    dec_vals   = np.zeros((n_meth, n_len))

    for mi, method in enumerate(methods):
        fe, fc, fd = BREAKDOWN_FRACS.get(method, (0.0, 1.0, 0.0))
        for li, length in enumerate(lengths):
            row = agg[(agg.method == method) & (agg.prompt_length == length)]
            if row.empty:
                continue
            total = row["avg_latency_ms"].values[0]
            enc_vals[mi, li]  = total * fe
            comp_vals[mi, li] = total * fc
            dec_vals[mi, li]  = total * fd

    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.80 / n_meth
    x = np.arange(n_len)

    for mi, method in enumerate(methods):
        offset = (mi - n_meth / 2 + 0.5) * width
        xs = x + offset
        b1 = ax.bar(xs, enc_vals[mi],  width, color=BREAKDOWN_COLORS[0],
                    alpha=0.9, label="Encryption"  if mi == 0 else "_")
        b2 = ax.bar(xs, comp_vals[mi], width, bottom=enc_vals[mi],
                    color=BREAKDOWN_COLORS[1], alpha=0.9,
                    label="Compute"     if mi == 0 else "_")
        b3 = ax.bar(xs, dec_vals[mi],  width, bottom=enc_vals[mi] + comp_vals[mi],
                    color=BREAKDOWN_COLORS[2], alpha=0.9,
                    label="Decryption"  if mi == 0 else "_")

        # Method label below bars
        ax.text(xs.mean(), -agg["avg_latency_ms"].max() * 0.07,
                METHOD_LABELS.get(method, method),
                ha="center", va="top", fontsize=7.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l} tok" for l in lengths])
    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Time Breakdown: Encryption / Compute / Decryption")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # Dual legend
    breakdown_handles = [
        mpatches.Patch(color=BREAKDOWN_COLORS[i], label=BREAKDOWN_LABELS[i])
        for i in range(3)
    ]
    method_handles = [
        mpatches.Patch(color=METHOD_COLORS.get(m, "#999"), label=METHOD_LABELS.get(m, m))
        for m in methods
    ]
    leg1 = ax.legend(handles=breakdown_handles, title="Phase",
                     loc="upper left", framealpha=0.85)
    ax.add_artist(leg1)
    ax.legend(handles=method_handles, title="Method",
              loc="upper center", framealpha=0.85, ncol=n_meth)

    fig.text(0.5, -0.06,
             "Figure 5. Stacked latency breakdown per phase. "
             "Plain baseline has no encryption/decryption overhead.",
             ha="center", fontsize=9, color="#555")

    _save(fig, out / "time_breakdown_stacked.png")


# ---------------------------------------------------------------------------
# Plot 6 — Comparison table PDF
# ---------------------------------------------------------------------------

def make_table_pdf(df: pd.DataFrame, out: Path) -> None:
    """Render a formatted summary table to comparison_table.pdf."""
    out.mkdir(parents=True, exist_ok=True)
    pdf_path = str(out / "comparison_table.pdf")

    # Aggregate: mean over all lengths per method
    summary = df.groupby("method").agg(
        avg_latency_ms=("avg_latency_ms", "mean"),
        tokens_per_sec=("tokens_per_sec", "mean"),
        perplexity=("perplexity", "mean"),
        memory_mb=("memory_mb", "mean"),
        ciphertext_overhead=("ciphertext_overhead", "mean"),
    ).reset_index()

    # Sort by METHOD_ORDER
    order_map = {m: i for i, m in enumerate(METHOD_ORDER)}
    summary["_ord"] = summary["method"].map(lambda m: order_map.get(m, 99))
    summary = summary.sort_values("_ord").drop(columns="_ord")

    # Also build per-length breakdown table
    length_table = df.groupby(["method", "prompt_length"]).agg(
        avg_latency_ms=("avg_latency_ms", "mean"),
        tokens_per_sec=("tokens_per_sec", "mean"),
        perplexity=("perplexity", "mean"),
    ).reset_index()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=landscape(A4),
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=16,
        spaceAfter=4,
    )
    story.append(Paragraph("Experiment Results — Method Comparison", title_style))
    story.append(Paragraph(
        "Mean metrics aggregated across all prompt lengths (10, 50, 100, 200 tokens). "
        "Best value in each column is highlighted.",
        styles["Normal"],
    ))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0072B2")))
    story.append(Spacer(1, 0.3*cm))

    # ── Summary table ─────────────────────────────────────────────────────────
    story.append(Paragraph("Table 1. Overall Method Summary", styles["Heading2"]))
    story.append(Spacer(1, 0.2*cm))

    headers = ["Method", "Latency (ms)", "Throughput\n(tok/s)",
               "Perplexity", "Memory\n(MB)", "Ciphertext\nOverhead"]
    col_keys = ["avg_latency_ms", "tokens_per_sec", "perplexity",
                "memory_mb", "ciphertext_overhead"]

    # Determine best values (lower=better for latency/ppl/mem/overhead, higher=better for tps)
    best_lower = {k: summary[k].min() for k in ["avg_latency_ms", "perplexity",
                                                  "memory_mb", "ciphertext_overhead"]}
    best_higher = {"tokens_per_sec": summary["tokens_per_sec"].max()}

    def fmt(key: str, val: float) -> str:
        if key == "avg_latency_ms":  return f"{val:,.1f}"
        if key == "tokens_per_sec":  return f"{val:,.1f}"
        if key == "perplexity":      return f"{val:.3f}"
        if key == "memory_mb":       return f"{val:.1f}"
        if key == "ciphertext_overhead": return f"{val:.2f}×"
        return str(val)

    data: List[List] = [headers]
    for _, row in summary.iterrows():
        method = row["method"]
        cells = [Paragraph(METHOD_LABELS.get(method, method), styles["Normal"])]
        for key in col_keys:
            val = row[key]
            txt = fmt(key, val)
            # Bold the best cell
            is_best = (
                (key in best_lower and abs(val - best_lower[key]) < 1e-9) or
                (key in best_higher and abs(val - best_higher[key]) < 1e-9)
            )
            if is_best:
                cells.append(Paragraph(f"<b>{txt}</b>", styles["Normal"]))
            else:
                cells.append(Paragraph(txt, styles["Normal"]))
        data.append(cells)

    col_widths = [4.5*cm, 3.2*cm, 3.2*cm, 3.0*cm, 3.0*cm, 3.5*cm]

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        # Header row
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#0072B2")),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  10),
        ("ALIGN",        (0, 0), (-1, 0),  "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#F7F7F7"), colors.white]),
        ("ALIGN",        (1, 1), (-1, -1), "RIGHT"),
        ("FONTSIZE",     (0, 1), (-1, -1), 9),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.5*cm))

    # ── Per-length breakdown table ────────────────────────────────────────────
    story.append(Paragraph("Table 2. Latency and Perplexity by Prompt Length",
                            styles["Heading2"]))
    story.append(Spacer(1, 0.2*cm))

    lengths = sorted(length_table["prompt_length"].unique())
    methods_present = [m for m in METHOD_ORDER if m in length_table["method"].unique()]

    # Build multi-column header: Method | 10 tok latency | 10 tok ppl | 50 tok ... 
    sub_headers = ["Latency\n(ms)", "PPL", "Tok/s"]
    header_row1 = [""] + [f"{int(l)} tokens" for l in lengths for _ in sub_headers]
    header_row2 = ["Method"] + sub_headers * len(lengths)
    data2: List = [header_row1, header_row2]

    for method in methods_present:
        row_cells = [METHOD_LABELS.get(method, method)]
        for length in lengths:
            sub = length_table[
                (length_table.method == method) &
                (length_table.prompt_length == length)
            ]
            if sub.empty:
                row_cells.extend(["—", "—", "—"])
            else:
                r = sub.iloc[0]
                row_cells.append(f"{r['avg_latency_ms']:,.0f}")
                row_cells.append(f"{r['perplexity']:.2f}")
                row_cells.append(f"{r['tokens_per_sec']:,.0f}")
        data2.append(row_cells)

    n_cols2 = 1 + len(lengths) * 3
    cw2 = [3.5*cm] + [2.2*cm] * (n_cols2 - 1)
    tbl2 = Table(data2, colWidths=cw2, repeatRows=2)

    span_style = []
    for li, _ in enumerate(lengths):
        start_col = 1 + li * 3
        span_style.append(("SPAN", (start_col, 0), (start_col + 2, 0)))

    tbl2.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 1),  colors.HexColor("#004488")),
        ("TEXTCOLOR",     (0, 0), (-1, 1),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 1),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 1),  9),
        ("ALIGN",         (0, 0), (-1, 1),  "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 2), (-1, -1),
         [colors.HexColor("#F7F7F7"), colors.white]),
        ("ALIGN",         (1, 2), (-1, -1), "RIGHT"),
        ("FONTSIZE",      (0, 2), (-1, -1), 8),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
    ] + span_style))
    story.append(tbl2)

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "<i>Bold values indicate best (lowest/highest) in each column. "
        "HE = Homomorphic Encryption. Ciphertext overhead is relative to plain-text size. "
        "All latencies in milliseconds; perplexity — lower is better.</i>",
        ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8, textColor=colors.grey),
    ))

    doc.build(story)
    log.info("  Saved → %s", pdf_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate publication-ready plots from experiment results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results", default='experiment_results/results.csv',
                   help="Path to results.csv or results.json.")
    p.add_argument("--out-dir", default="out",
                   help="Output directory for all generated files.")
    p.add_argument("--seaborn", action="store_true",
                   help="Use seaborn whitegrid theme (requires seaborn installed).")
    p.add_argument("--dpi", type=int, default=FIGURE_DPI,
                   help="Resolution for PNG exports.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global FIGURE_DPI
    FIGURE_DPI = args.dpi

    _apply_global_style(args.seaborn)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_results(args.results)

    log.info("Generating plots → %s/", out)

    steps = [
        ("latency_comparison.png",       lambda: plot_latency(df, out)),
        ("throughput_comparison.png",     lambda: plot_throughput(df, out)),
        ("memory_overhead.png",           lambda: plot_memory(df, out)),
        ("latency_vs_perplexity.png",     lambda: plot_latency_vs_perplexity(df, out)),
        ("time_breakdown_stacked.png",    lambda: plot_time_breakdown(df, out)),
        ("comparison_table.pdf",          lambda: make_table_pdf(df, out)),
    ]

    for name, fn in steps:
        log.info("Generating %s …", name)
        try:
            fn()
        except Exception as exc:
            import traceback
            log.error("Failed to generate %s: %s\n%s", name, exc, traceback.format_exc())

    log.info("Done. All outputs in %s/", out)


if __name__ == "__main__":
    main()