#!/usr/bin/env python3
"""
privacy_analysis.py — Privacy scoring and Pareto frontier analysis for SelectiveHE configs.

Scoring model
-------------
  privacy_score(config) -> float in [0, 1]

  Three equally-weighted sub-scores (1/3 each):
    • input_privacy    : fraction of input tokens processed under encryption
    • intermediate_privacy : fraction of intermediate activations kept encrypted
    • output_privacy   : whether final logits / outputs are kept encrypted

  Sub-score semantics
    0.0 — fully plaintext (maximum leakage)
    1.0 — fully encrypted (zero leakage)

  Composite = (input_privacy + intermediate_privacy + output_privacy) / 3

Leakage heuristics
------------------
  For each strategy a LeakageProfile is computed, capturing:
    • token_leakage_rate   : fraction of tokens visible to untrusted compute
    • layer_leakage_depth  : normalised depth of unencrypted layer range
    • side_channel_risk    : qualitative risk from timing / memory patterns (0–1)
    • re-identification_risk: probability heuristic of linking outputs to user

Usage
-----
  python privacy_analysis.py --results results.csv
  python privacy_analysis.py --results results.json --out pareto.png --no-plot
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("privacy_analysis")

# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class SelectiveHEConfig:
    """
    Describes how a selective homomorphic-encryption strategy processes a model.

    Attributes
    ----------
    name : str
        Human-readable identifier (must match method names in results CSV/JSON).
    input_enc_fraction : float
        Fraction of input tokens encrypted before leaving the client [0, 1].
    enc_layer_start : int
        First transformer layer executed under HE (0-indexed).
    enc_layer_end : int
        Last transformer layer executed under HE (inclusive, 0-indexed).
    total_layers : int
        Total number of transformer layers in the model.
    output_encrypted : bool
        Whether final logits are returned in ciphertext form.
    side_channel_risk : float
        Analyst-assigned side-channel risk score [0, 1].
    re_id_risk : float
        Analyst-assigned re-identification risk [0, 1].
    notes : str
        Free-text description of the strategy.
    """
    name: str
    input_enc_fraction: float       # [0, 1]
    enc_layer_start: int            # 0-indexed
    enc_layer_end: int              # inclusive
    total_layers: int
    output_encrypted: bool
    side_channel_risk: float = 0.0  # [0, 1]
    re_id_risk: float = 0.0         # [0, 1]
    notes: str = ""


@dataclass
class PrivacyScore:
    """Decomposed and composite privacy scores for a config."""
    config_name: str
    input_privacy: float        # [0, 1]
    intermediate_privacy: float # [0, 1]
    output_privacy: float       # [0, 1]
    composite: float            # arithmetic mean of the three, [0, 1]
    leakage_token_rate: float   # fraction of tokens leaked to server
    leakage_layer_depth: float  # normalised depth of plaintext execution
    side_channel_risk: float
    re_id_risk: float
    overall_leakage: float      # composite leakage (1 - privacy) with risk adjustments


# ---------------------------------------------------------------------------
# Predefined configurations for all methods
# ---------------------------------------------------------------------------
# Total transformer layers assumed for scoring (adjust to match your model)
_TOTAL_LAYERS = 32

CONFIGS: Dict[str, SelectiveHEConfig] = {
    "plain": SelectiveHEConfig(
        name="plain",
        input_enc_fraction=0.0,
        enc_layer_start=_TOTAL_LAYERS,   # no layer encrypted
        enc_layer_end=_TOTAL_LAYERS - 1,
        total_layers=_TOTAL_LAYERS,
        output_encrypted=False,
        side_channel_risk=0.05,
        re_id_risk=0.90,
        notes="No encryption; all computation in plaintext on server.",
    ),
    "full_he": SelectiveHEConfig(
        name="full_he",
        input_enc_fraction=1.0,
        enc_layer_start=0,
        enc_layer_end=_TOTAL_LAYERS - 1,
        total_layers=_TOTAL_LAYERS,
        output_encrypted=True,
        side_channel_risk=0.05,
        re_id_risk=0.02,
        notes="All computation under HE; maximum privacy, maximum latency.",
    ),
    "strategy1": SelectiveHEConfig(
        name="strategy1",
        input_enc_fraction=0.90,
        enc_layer_start=0,
        enc_layer_end=23,               # encrypt first 24 of 32 layers
        total_layers=_TOTAL_LAYERS,
        output_encrypted=False,
        side_channel_risk=0.15,
        re_id_risk=0.15,
        notes="Encrypt input + early layers; decode in final layers for speed.",
    ),
    "strategy2": SelectiveHEConfig(
        name="strategy2",
        input_enc_fraction=0.65,
        enc_layer_start=4,
        enc_layer_end=27,               # encrypt middle layers only
        total_layers=_TOTAL_LAYERS,
        output_encrypted=False,
        side_channel_risk=0.30,
        re_id_risk=0.35,
        notes="Selective mid-layer encryption; input and output partially exposed.",
    ),
    "strategy3": SelectiveHEConfig(
        name="strategy3",
        input_enc_fraction=0.80,
        enc_layer_start=0,
        enc_layer_end=15,               # encrypt first half
        total_layers=_TOTAL_LAYERS,
        output_encrypted=True,
        side_channel_risk=0.20,
        re_id_risk=0.20,
        notes="Encrypt input + first half of layers + output; skip mid-layers.",
    ),
}

# ---------------------------------------------------------------------------
# Privacy scoring
# ---------------------------------------------------------------------------

def _input_privacy(cfg: SelectiveHEConfig) -> float:
    """
    Input privacy sub-score.
    Equals the fraction of input tokens encrypted before leaving the client.
    """
    return float(np.clip(cfg.input_enc_fraction, 0.0, 1.0))


def _intermediate_privacy(cfg: SelectiveHEConfig) -> float:
    """
    Intermediate privacy sub-score.
    Measures what fraction of layers execute under HE.
    Accounts for layer ordering: gap before enc_layer_start also leaks.
    """
    n = cfg.total_layers
    if n == 0:
        return 0.0
    start = int(np.clip(cfg.enc_layer_start, 0, n))
    end   = int(np.clip(cfg.enc_layer_end,   0, n - 1))
    if start > end:
        return 0.0          # no layers encrypted
    encrypted_layers = end - start + 1
    # Penalise if there is a plaintext prefix (layers 0..start-1) or suffix
    prefix_penalty  = start / n              # fraction of leading plaintext layers
    suffix_penalty  = (n - 1 - end) / n     # fraction of trailing plaintext layers
    raw = encrypted_layers / n
    # Weighted: full coverage → 1.0, partial prefix/suffix → discounted
    score = raw * (1.0 - 0.5 * prefix_penalty) * (1.0 - 0.3 * suffix_penalty)
    return float(np.clip(score, 0.0, 1.0))


def _output_privacy(cfg: SelectiveHEConfig) -> float:
    """
    Output privacy sub-score.
    Binary: 1.0 if outputs are returned encrypted, else 0.0.
    A small partial credit is given when intermediate privacy is high
    (attacker cannot infer much even without output encryption).
    """
    if cfg.output_encrypted:
        return 1.0
    # Partial credit: if server never sees plaintext intermediates, output
    # leakage is bounded by what was already exposed in the middle layers.
    inter = _intermediate_privacy(cfg)
    partial = 0.30 * inter   # at most 0.30 credit for good intermediate privacy
    return float(np.clip(partial, 0.0, 1.0))


def privacy_score(config: SelectiveHEConfig) -> PrivacyScore:
    """
    Compute the privacy score for a SelectiveHEConfig.

    Returns a PrivacyScore dataclass with per-dimension and composite scores,
    plus leakage heuristics.

    Composite score
    ---------------
    composite = (input_privacy + intermediate_privacy + output_privacy) / 3

    All three dimensions are weighted equally (1/3 each) reflecting that
    exposure at *any* pipeline stage constitutes a privacy violation.

    Leakage heuristics
    ------------------
    leakage_token_rate   : fraction of tokens visible to server in plaintext
    leakage_layer_depth  : normalised depth of unencrypted layer execution
    overall_leakage      : adjusted composite leakage, accounting for
                           side-channel and re-identification risks
    """
    inp  = _input_privacy(config)
    intr = _intermediate_privacy(config)
    out  = _output_privacy(config)
    comp = (inp + intr + out) / 3.0

    # Leakage heuristics -------------------------------------------------------
    # Token leakage: fraction not encrypted at input
    token_leak = 1.0 - inp

    # Layer depth leakage: fraction of layers with plaintext access
    n = max(config.total_layers, 1)
    start = int(np.clip(config.enc_layer_start, 0, n))
    end   = int(np.clip(config.enc_layer_end,   0, n - 1))
    encrypted_layers = max(0, end - start + 1) if start <= end else 0
    plain_layers = n - encrypted_layers
    layer_depth_leak = plain_layers / n

    # Overall leakage: start from base (1 - composite), boost by side-channel
    # and re-id risks (each contributes up to 10% uplift)
    base_leak = 1.0 - comp
    overall_leak = base_leak + 0.10 * config.side_channel_risk \
                             + 0.10 * config.re_id_risk
    overall_leak = float(np.clip(overall_leak, 0.0, 1.0))

    return PrivacyScore(
        config_name=config.name,
        input_privacy=round(inp, 4),
        intermediate_privacy=round(intr, 4),
        output_privacy=round(out, 4),
        composite=round(comp, 4),
        leakage_token_rate=round(token_leak, 4),
        leakage_layer_depth=round(layer_depth_leak, 4),
        side_channel_risk=round(config.side_channel_risk, 4),
        re_id_risk=round(config.re_id_risk, 4),
        overall_leakage=round(overall_leak, 4),
    )


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def _is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """
    Given an (N, 2) array where column 0 = latency (lower better) and
    column 1 = privacy_score (higher better), return a boolean mask of
    Pareto-efficient points.

    A point is dominated if another point has both lower latency AND
    higher (or equal) privacy — or equal latency AND strictly higher privacy.
    """
    n = len(costs)
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        lat_i, priv_i = costs[i]
        for j in range(n):
            if i == j or not is_efficient[j]:
                continue
            lat_j, priv_j = costs[j]
            # j dominates i if j is at least as good in both dims and better in one
            if lat_j <= lat_i and priv_j >= priv_i and (lat_j < lat_i or priv_j > priv_i):
                is_efficient[i] = False
                break
    return is_efficient


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_results(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        log.error("Results file not found: %s", path)
        sys.exit(1)
    if p.suffix.lower() == ".json":
        with open(p) as f:
            data = json.load(f)
        records = data.get("summaries") or data.get("runs") or data
        df = pd.DataFrame(records)
    else:
        df = pd.read_csv(p)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for col in ["avg_latency_ms", "prompt_length"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _mean_latency_by_method(df: pd.DataFrame) -> Dict[str, float]:
    """Return mean avg_latency_ms per method, averaged over all prompt lengths."""
    return (
        df.groupby("method")["avg_latency_ms"]
        .mean()
        .to_dict()
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

# Design: dark-background "security dashboard" aesthetic
BG_COLOR     = "#0D1117"
GRID_COLOR   = "#21262D"
TEXT_COLOR   = "#E6EDF3"
ACCENT_COLOR = "#58A6FF"
PARETO_COLOR = "#F0883E"

METHOD_PALETTE: Dict[str, str] = {
    "plain":     "#6E7681",   # muted grey  — low privacy
    "strategy1": "#3FB950",   # green       — good balance
    "strategy2": "#D29922",   # amber       — moderate
    "strategy3": "#58A6FF",   # blue        — high privacy
    "full_he":   "#FF7B72",   # red-orange  — maximum privacy
}
METHOD_LABELS: Dict[str, str] = {
    "plain":     "Plain",
    "full_he":   "Full HE",
    "strategy1": "Strategy 1",
    "strategy2": "Strategy 2",
    "strategy3": "Strategy 3",
}


def _annotate_point(
    ax: plt.Axes,
    x: float, y: float,
    label: str,
    color: str,
    is_pareto: bool,
) -> None:
    """Draw annotated callout for a method point."""
    # Offset direction heuristic: push label away from centre
    cx, cy = 0.5, 0.5  # normalised centre (unused — use fixed offsets per method)
    offsets = {
        "plain":     (-55, -28),
        "full_he":   (+10, +12),
        "strategy1": (+10, -28),
        "strategy2": (-60, +10),
        "strategy3": (+10, +14),
    }
    key = label.lower().replace(" ", "")
    # Map label back to config name
    reverse_map = {v.lower().replace(" ", ""): k for k, v in METHOD_LABELS.items()}
    cfg_key = reverse_map.get(key, key)
    dx, dy = offsets.get(cfg_key, (10, 10))

    ax.annotate(
        label,
        xy=(x, y),
        xytext=(x + dx * (ax.get_xlim()[1] - ax.get_xlim()[0]) / 800,
                y + dy * (ax.get_ylim()[1] - ax.get_ylim()[0]) / 800),
        fontsize=9,
        color=color,
        fontweight="bold" if is_pareto else "normal",
        arrowprops=dict(
            arrowstyle="-",
            color=color,
            lw=0.8,
            alpha=0.6,
        ),
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor=BG_COLOR,
            edgecolor=color,
            alpha=0.85,
            linewidth=0.8,
        ),
    )


def _build_pareto_plot(
    scores: Dict[str, PrivacyScore],
    latencies: Dict[str, float],
    out_path: Path,
) -> None:
    methods = [m for m in scores if m in latencies]
    xs = np.array([latencies[m] for m in methods])       # latency (lower better)
    ys = np.array([scores[m].composite for m in methods]) # privacy (higher better)

    costs = np.column_stack([xs, ys])
    pareto_mask = _is_pareto_efficient(costs)

    # Sort Pareto points by latency for the frontier line
    pareto_indices = np.where(pareto_mask)[0]
    pareto_sorted  = pareto_indices[np.argsort(xs[pareto_indices])]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Grid
    ax.grid(True, color=GRID_COLOR, linewidth=0.7, linestyle="--", alpha=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Shaded regions
    xlims = (max(xs.min() * 0.6, 1), xs.max() * 1.25)
    ylims = (-0.05, 1.08)

    # Quadrant backgrounds
    xmid = np.median(xs)
    ymid = 0.5
    ax.axhspan(ymid, ylims[1], xmin=0, xmax=1,
               color="#1C2B1C", alpha=0.25, zorder=1)   # high-privacy band
    ax.axvspan(xlims[0], xmid, ymin=0, ymax=1,
               color="#1C1C2B", alpha=0.20, zorder=1)   # low-latency band

    # Quadrant labels
    ax.text(xlims[0] * 1.05, ylims[1] * 0.97,
            "← Fast & Private\n   (ideal)",
            fontsize=7.5, color="#3FB950", alpha=0.6, va="top")
    ax.text(xs.max() * 1.10, 0.02,
            "Slow & Exposed\n(avoid) →",
            fontsize=7.5, color="#FF7B72", alpha=0.55, va="bottom")

    # Pareto frontier line (step-style for dominance clarity)
    if len(pareto_sorted) > 1:
        px = xs[pareto_sorted]
        py = ys[pareto_sorted]
        ax.step(px, py, where="post",
                color=PARETO_COLOR, lw=2.0, linestyle="--",
                alpha=0.75, zorder=3, label="Pareto frontier")
        # Shaded area under Pareto front
        ax.fill_between(
            np.append(px, px[-1]),
            np.append(py, -0.1),
            step="post",
            color=PARETO_COLOR, alpha=0.07, zorder=2,
        )

    # Points
    for i, method in enumerate(methods):
        x, y   = xs[i], ys[i]
        color  = METHOD_PALETTE.get(method, ACCENT_COLOR)
        is_p   = pareto_mask[i]
        marker = "★" if is_p else "o"
        size   = 260 if is_p else 160

        ax.scatter(x, y, s=size, c=color, zorder=5,
                   edgecolors="white" if is_p else color,
                   linewidths=1.5 if is_p else 0,
                   marker="*" if is_p else "o")

        # Risk rings: outer ring sized by overall_leakage
        leakage = scores[method].overall_leakage
        ring_r  = 350 + 600 * leakage
        ax.scatter(x, y, s=ring_r, c=color, alpha=0.10,
                   edgecolors=color, linewidths=0.6, zorder=4)

    # Annotations (after setting limits)
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)

    for i, method in enumerate(methods):
        x, y  = xs[i], ys[i]
        color = METHOD_PALETTE.get(method, ACCENT_COLOR)
        is_p  = pareto_mask[i]
        _annotate_point(ax, x, y, METHOD_LABELS.get(method, method), color, is_p)

    # Privacy threshold line at 0.5
    ax.axhline(0.5, color="#8B949E", lw=0.8, linestyle=":", alpha=0.6)
    ax.text(xlims[1] * 0.995, 0.515, "privacy threshold = 0.5",
            fontsize=7.5, color="#8B949E", ha="right", alpha=0.7)

    # Axes styling
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)

    ax.set_xlabel("Mean Inference Latency (ms)  ←  lower is better",
                  fontsize=11, labelpad=10)
    ax.set_ylabel("Privacy Score  ↑  higher is better",
                  fontsize=11, labelpad=10)
    ax.set_title(
        "Privacy–Latency Trade-off: Pareto Frontier\n"
        "Selective Homomorphic Encryption Strategies",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", pad=16,
    )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # Legend
    handles = [
        Line2D([0], [0], marker="*", color="w", markersize=11,
               markerfacecolor=PARETO_COLOR, label="Pareto-optimal ★"),
        Line2D([0], [0], marker="o", color="w", markersize=9,
               markerfacecolor="#6E7681", label="Dominated"),
        Line2D([0], [0], linestyle="--", color=PARETO_COLOR, lw=1.8,
               label="Pareto frontier"),
    ] + [
        Line2D([0], [0], marker="o", color="w", markersize=9,
               markerfacecolor=METHOD_PALETTE.get(m, "#999"),
               label=METHOD_LABELS.get(m, m))
        for m in ["plain", "strategy2", "strategy1", "strategy3", "full_he"]
        if m in latencies
    ]
    leg = ax.legend(
        handles=handles,
        loc="lower right",
        framealpha=0.85,
        facecolor="#161B22",
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
        fontsize=8.5,
        title="Methods & Annotations",
        title_fontsize=8.5,
    )
    leg.get_title().set_color(TEXT_COLOR)

    # Caption
    fig.text(
        0.5, -0.03,
        "Figure. Privacy–Latency Pareto frontier. "
        "★ = Pareto-optimal (no method dominates in both dimensions). "
        "Ring radius ∝ overall leakage score (side-channel + re-id adjusted). "
        "Shaded region = dominated zone relative to Pareto front.",
        ha="center", fontsize=8, color="#8B949E", wrap=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    log.info("Saved Pareto plot → %s", out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Textual summary
# ---------------------------------------------------------------------------

def _print_summary(
    scores: Dict[str, PrivacyScore],
    latencies: Dict[str, float],
    pareto_mask: Dict[str, bool],
) -> str:
    lines: List[str] = []
    sep = "═" * 74

    lines.append(sep)
    lines.append("  PRIVACY ANALYSIS SUMMARY — Selective HE Strategies")
    lines.append(sep)

    # Header
    lines.append(f"\n{'Method':<14} {'Privacy':>8} {'Latency(ms)':>12} "
                 f"{'Leakage':>8} {'SideCh':>7} {'ReID':>6}  {'Pareto':>7}")
    lines.append("─" * 74)

    methods_sorted = sorted(
        scores.keys(),
        key=lambda m: scores[m].composite,
        reverse=True,
    )
    for m in methods_sorted:
        s   = scores[m]
        lat = latencies.get(m, float("nan"))
        p   = "★" if pareto_mask.get(m, False) else " "
        lines.append(
            f"{METHOD_LABELS.get(m, m):<14} "
            f"{s.composite:>8.4f} "
            f"{lat:>12,.1f} "
            f"{s.overall_leakage:>8.4f} "
            f"{s.side_channel_risk:>7.4f} "
            f"{s.re_id_risk:>6.4f}  "
            f"{p:>7}"
        )

    lines.append("\n" + sep)
    lines.append("  SCORE DECOMPOSITION  (input | intermediate | output)")
    lines.append(sep)
    lines.append(f"\n{'Method':<14} {'Input':>8} {'Interm.':>9} {'Output':>8}  Notes")
    lines.append("─" * 74)
    for m in methods_sorted:
        s = scores[m]
        cfg = CONFIGS.get(m)
        note = cfg.notes[:35] + "…" if cfg and len(cfg.notes) > 35 else (cfg.notes if cfg else "")
        lines.append(
            f"{METHOD_LABELS.get(m, m):<14} "
            f"{s.input_privacy:>8.4f} "
            f"{s.intermediate_privacy:>9.4f} "
            f"{s.output_privacy:>8.4f}  {note}"
        )

    lines.append("\n" + sep)
    lines.append("  LEAKAGE HEURISTICS")
    lines.append(sep)
    lines.append(f"\n{'Method':<14} {'TokenLeak':>10} {'LayerLeak':>10}  Risk-adjusted leakage")
    lines.append("─" * 74)
    for m in methods_sorted:
        s = scores[m]
        bar_len = int(s.overall_leakage * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        lines.append(
            f"{METHOD_LABELS.get(m, m):<14} "
            f"{s.leakage_token_rate:>10.4f} "
            f"{s.leakage_layer_depth:>10.4f}  "
            f"{bar}  {s.overall_leakage:.3f}"
        )

    lines.append("\n" + sep)
    lines.append("  PARETO-OPTIMAL METHODS  (not dominated in privacy AND latency)")
    lines.append(sep)
    pareto_methods = [m for m in methods_sorted if pareto_mask.get(m, False)]
    if pareto_methods:
        for m in pareto_methods:
            s   = scores[m]
            lat = latencies.get(m, float("nan"))
            lines.append(
                f"  ★ {METHOD_LABELS.get(m, m):<14} "
                f"privacy={s.composite:.4f}  latency={lat:,.1f}ms"
            )
    else:
        lines.append("  (none found — check data)")

    lines.append(sep + "\n")
    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute privacy scores and plot Privacy–Latency Pareto frontier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results", default='experiment_results/results.csv',
                   help="Path to results.csv or results.json from experiment_runner.")
    p.add_argument("--out", default="pareto.png",
                   help="Output path for the Pareto plot PNG.")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip generating the plot (print summary only).")
    p.add_argument("--json-scores", default=None,
                   help="Optional path to save privacy scores as JSON.")
    p.add_argument("--total-layers", type=int, default=_TOTAL_LAYERS,
                   help="Number of transformer layers in the model (used for scoring).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Allow overriding layer count at runtime
    if args.total_layers != _TOTAL_LAYERS:
        for cfg in CONFIGS.values():
            cfg.total_layers = args.total_layers
        log.info("Using total_layers=%d for all configs.", args.total_layers)

    # Load results
    df = _load_results(args.results)
    if "avg_latency_ms" not in df.columns or "method" not in df.columns:
        log.error("Results file must have 'method' and 'avg_latency_ms' columns.")
        sys.exit(1)

    latencies = _mean_latency_by_method(df)
    log.info("Methods found in results: %s", list(latencies.keys()))

    # Compute scores for all configs
    scores: Dict[str, PrivacyScore] = {}
    for method, cfg in CONFIGS.items():
        ps = privacy_score(cfg)
        scores[method] = ps
        log.debug(
            "%s: composite=%.4f  (inp=%.4f intr=%.4f out=%.4f)",
            method, ps.composite, ps.input_privacy,
            ps.intermediate_privacy, ps.output_privacy,
        )

    # Pareto mask (only for methods that appear in both scores and latency data)
    common = [m for m in scores if m in latencies]
    xs = np.array([latencies[m]     for m in common])
    ys = np.array([scores[m].composite for m in common])
    costs = np.column_stack([xs, ys])
    mask_arr = _is_pareto_efficient(costs)
    pareto_mask = {m: bool(mask_arr[i]) for i, m in enumerate(common)}

    # Summary
    _print_summary(scores, latencies, pareto_mask)

    # Optional: save scores as JSON
    if args.json_scores:
        out_j = Path(args.json_scores)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        payload = {m: asdict(s) for m, s in scores.items()}
        with open(out_j, "w") as f:
            json.dump(payload, f, indent=2)
        log.info("Saved privacy scores → %s", out_j)

    # Plot
    if not args.no_plot:
        _build_pareto_plot(scores, latencies, Path(args.out))
        log.info("Done. Pareto plot saved to %s", args.out)
    else:
        log.info("Skipped plot generation (--no-plot).")


if __name__ == "__main__":
    main()