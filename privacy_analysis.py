import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

from selective_he_config import (
    SelectiveHEConfig,
    STRATEGY_1_CONFIG,
    STRATEGY_2_CONFIG,
    STRATEGY_3_CONFIG,
)


def privacy_score(config: SelectiveHEConfig) -> float:
    """
    Compute a heuristic privacy score in [0, 1] for a SelectiveHEConfig.

    We consider three equally weighted dimensions:
      1) Input privacy   (e.g., embedding / first-block attention encrypted)
      2) Intermediate    (hidden states across transformer blocks)
      3) Output privacy  (final LM head / logits encrypted)

    Each dimension contributes 1/3 to the final score. Within each, we
    approximate coverage by the fraction of relevant layers that are
    present in `layers_to_encrypt`.
    """
    layers = set(config.layers_to_encrypt)

    # Heuristic buckets for TinyGPT-like models
    input_layers = {"embedding", "blocks.0.attention"}
    interm_layers = {
        "blocks.0.attention",
        "blocks.0.ffn",
        "blocks.1.attention",
        "blocks.1.ffn",
    }
    output_layers = {"lm_head"}

    def coverage(target: set) -> float:
        if not target:
            return 0.0
        return len(layers.intersection(target)) / len(target)

    input_cov = coverage(input_layers)
    interm_cov = coverage(interm_layers)
    output_cov = coverage(output_layers)

    score = (input_cov + interm_cov + output_cov) / 3.0
    return max(0.0, min(1.0, float(score)))


def base_privacy_scores() -> Dict[str, float]:
    """
    Privacy scores for named methods using example configs.
    """
    # Plaintext: no encryption
    plain_score = 0.0

    # Full HE: idealized full encryption on input/intermediate/output
    full_he_score = 1.0

    # Strategies using predefined configs
    strat1 = SelectiveHEConfig(
        layers_to_encrypt=STRATEGY_1_CONFIG["layers_to_encrypt"],
        operations_to_encrypt=STRATEGY_1_CONFIG["operations_to_encrypt"],
        encryption_granularity=STRATEGY_1_CONFIG["encryption_granularity"],
    )
    strat2 = SelectiveHEConfig(
        layers_to_encrypt=STRATEGY_2_CONFIG["layers_to_encrypt"],
        operations_to_encrypt=STRATEGY_2_CONFIG["operations_to_encrypt"],
        encryption_granularity=STRATEGY_2_CONFIG["encryption_granularity"],
    )
    strat3 = SelectiveHEConfig(
        layers_to_encrypt=STRATEGY_3_CONFIG["layers_to_encrypt"],
        operations_to_encrypt=STRATEGY_3_CONFIG["operations_to_encrypt"],
        encryption_granularity=STRATEGY_3_CONFIG["encryption_granularity"],
    )

    return {
        "plain": plain_score,
        "full_he": full_he_score,
        "strategy1": privacy_score(strat1),
        "strategy2": privacy_score(strat2),
        "strategy3": privacy_score(strat3),
    }


def load_results_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _to_float(row: Dict[str, Any], key: str) -> float:
    val = row.get(key)
    if val is None or val == "":
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def build_privacy_latency_points(
    rows: List[Dict[str, Any]],
    scores: Dict[str, float],
) -> List[Tuple[str, float, float]]:
    """
    Return list of (method, privacy_score, avg_latency_ms) for methods
    that have both a score and valid latency.
    """
    points: List[Tuple[str, float, float]] = []
    for r in rows:
        method = r.get("method", "")
        if method not in scores:
            continue
        if r.get("status") not in (None, "", "done"):
            # Skip errored methods
            continue

        latency = _to_float(r, "avg_latency_ms")
        if latency != latency:  # NaN
            continue

        points.append((method, scores[method], latency))
    return points


def pareto_front(points: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
    """
    Compute a simple Pareto front (max privacy, min latency).

    A point A dominates B if:
        privacy_A >= privacy_B and latency_A <= latency_B,
    with at least one strict inequality.
    """
    front: List[Tuple[str, float, float]] = []
    for i, (m_i, p_i, l_i) in enumerate(points):
        dominated = False
        for j, (m_j, p_j, l_j) in enumerate(points):
            if j == i:
                continue
            if (p_j >= p_i and l_j <= l_i) and (p_j > p_i or l_j < l_i):
                dominated = True
                break
        if not dominated:
            front.append((m_i, p_i, l_i))
    return front


def plot_pareto(
    points: List[Tuple[str, float, float]],
    front: List[Tuple[str, float, float]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))

    # All points
    for method, p, l in points:
        ax.scatter(l, p, c="tab:blue")
        ax.annotate(method, (l, p), textcoords="offset points", xytext=(5, 3), fontsize=8)

    # Pareto frontier (sorted by latency)
    if front:
        front_sorted = sorted(front, key=lambda x: x[2])
        lat = [x[2] for x in front_sorted]
        priv = [x[1] for x in front_sorted]
        ax.plot(lat, priv, "--", color="tab:red", label="Pareto front")

    ax.set_xlabel("Average latency (ms)")
    ax.set_ylabel("Privacy score (0–1)")
    ax.set_title("Privacy vs. latency Pareto frontier")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def summarize(points: List[Tuple[str, float, float]], front: List[Tuple[str, float, float]]) -> str:
    lines: List[str] = []
    lines.append("Privacy vs. latency summary:")
    for method, p, l in sorted(points, key=lambda x: x[2]):
        lines.append(f"  - {method:9s}: privacy={p:.3f}, latency={l:.2f} ms")

    lines.append("\nPareto-optimal methods (best privacy for given latency):")
    for method, p, l in sorted(front, key=lambda x: x[2]):
        lines.append(f"  * {method:9s}: privacy={p:.3f}, latency={l:.2f} ms")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Privacy analysis and Pareto frontier visualization.",
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results CSV file (e.g., results.csv).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="pareto.png",
        help="Output path for Pareto frontier plot (default: pareto.png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_path = Path(args.results).expanduser().resolve()
    if not results_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_path}")

    rows = load_results_csv(results_path)
    scores = base_privacy_scores()
    points = build_privacy_latency_points(rows, scores)

    if not points:
        raise RuntimeError("No valid methods with latency found in results.")

    front = pareto_front(points)
    out_path = Path(args.out).expanduser().resolve()
    plot_pareto(points, front, out_path)

    text = summarize(points, front)
    print(text)
    print(f"\nPareto plot saved to: {out_path}")


if __name__ == "__main__":
    main()

