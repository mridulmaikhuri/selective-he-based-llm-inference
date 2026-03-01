import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


DEGREES: List[int] = [3, 5, 7]

_GELU_COEFFS: Dict[int, np.ndarray] = {}
_EXP_COEFFS: Dict[int, np.ndarray] = {}


def _fit_gelu_polynomials(degrees: List[int] = DEGREES) -> None:
    """
    Fit least-squares polynomial approximations for GELU on [-5, 5].
    Stores coefficients in global _GELU_COEFFS (highest power first).
    """
    xs = np.linspace(-5.0, 5.0, 2001, dtype=np.float64)
    xs_t = torch.from_numpy(xs).float()
    ys = F.gelu(xs_t).numpy().astype(np.float64)

    for d in degrees:
        coeffs = np.polyfit(xs, ys, deg=d)
        _GELU_COEFFS[d] = coeffs


def _fit_exp_polynomials(degrees: List[int] = DEGREES) -> None:
    """
    Fit least-squares polynomial approximations for exp(z) on [-10, 0].
    Used inside a numerically stable softmax approximation.
    Stores coefficients in global _EXP_COEFFS.
    """
    zs = np.linspace(-10.0, 0.0, 2001, dtype=np.float64)
    ys = np.exp(zs)

    for d in degrees:
        coeffs = np.polyfit(zs, ys, deg=d)
        _EXP_COEFFS[d] = coeffs


def _ensure_fitted() -> None:
    if not _GELU_COEFFS:
        _fit_gelu_polynomials()
    if not _EXP_COEFFS:
        _fit_exp_polynomials()


def compute_mse_gelu(degrees: List[int] = DEGREES) -> Dict[int, float]:
    """
    Compute MSE between true GELU and polynomial approximations over [-5, 5].
    """
    _ensure_fitted()

    xs = np.linspace(-5.0, 5.0, 2001, dtype=np.float64)
    xs_t = torch.from_numpy(xs).float()
    ys_true = F.gelu(xs_t).numpy().astype(np.float64)

    mses: Dict[int, float] = {}
    for d in degrees:
        coeffs = _GELU_COEFFS[d]
        ys_approx = np.polyval(coeffs, xs)
        mse = float(np.mean((ys_true - ys_approx) ** 2))
        mses[d] = mse
    return mses


def compute_mse_softmax(
    degrees: List[int] = DEGREES,
    num_samples: int = 1000,
    dim: int = 4,
) -> Dict[int, float]:
    """
    Compute MSE between true softmax and approximate softmax for random logits.
    """
    _ensure_fitted()

    logits = torch.randn(num_samples, dim)
    probs_true = F.softmax(logits, dim=-1).numpy()

    mses: Dict[int, float] = {}
    for d in degrees:
        probs_approx = softmax_approx(logits, degree=d).numpy()
        mse = float(np.mean((probs_true - probs_approx) ** 2))
        mses[d] = mse
    return mses


def gelu_approx(x: torch.Tensor, degree: int = 5) -> torch.Tensor:
    """
    Polynomial approximation to GELU using pre-fitted coefficients.

    Args:
        x: input tensor
        degree: polynomial degree (3, 5, or 7 supported by default)
    """
    _ensure_fitted()
    if degree not in _GELU_COEFFS:
        _fit_gelu_polynomials([degree])

    coeffs = _GELU_COEFFS[degree]
    # Horner's method for numerical stability
    y = torch.zeros_like(x, dtype=torch.float32)
    for c in coeffs:
        y = y * x + float(c)
    return y


def softmax_approx(x: torch.Tensor, degree: int = 5, dim: int = -1) -> torch.Tensor:
    """
    Approximate softmax using a polynomial approximation to exp on a
    numerically-stable range plus standard max-subtraction.

    Steps:
      1. Shift logits by subtracting max along `dim` (stability).
      2. Clamp z in [-10, 0] and approximate exp(z) with a degree-d polynomial.
      3. Normalize approximate exp values to get a probability distribution.

    Numerical stability concerns:
      - For very large negative inputs (< -10), exp(z) underflows; clamping
        to -10 introduces approximation error but avoids overflow.
      - Polynomial exp approximation may become inaccurate outside the
        training range [-10, 0]; callers should clip inputs if needed.
    """
    _ensure_fitted()
    if degree not in _EXP_COEFFS:
        _fit_exp_polynomials([degree])

    coeffs = _EXP_COEFFS[degree]

    # 1. Max subtraction for numerical stability
    z = x - x.max(dim=dim, keepdim=True).values

    # 2. Clamp and apply polynomial approximation of exp
    z_clamped = z.clamp(min=-10.0, max=0.0)
    y = torch.zeros_like(z_clamped, dtype=torch.float32)
    for c in coeffs:
        y = y * z_clamped + float(c)

    # Avoid negative approximations and division by zero
    y = torch.clamp(y, min=0.0)
    denom = y.sum(dim=dim, keepdim=True) + 1e-8
    probs = y / denom
    return probs


def plot_approximations(out_dir: Path | None = None) -> None:
    """
    Save comparison plots for GELU and softmax approximations.
    """
    _ensure_fitted()
    if out_dir is None:
        out_dir = Path("activation_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # GELU plot
    xs = np.linspace(-5.0, 5.0, 1001, dtype=np.float64)
    xs_t = torch.from_numpy(xs).float()
    ys_true = F.gelu(xs_t).numpy().astype(np.float64)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys_true, label="GELU (true)", color="black", linewidth=2)
    for d in DEGREES:
        coeffs = _GELU_COEFFS[d]
        ys_approx = np.polyval(coeffs, xs)
        ax.plot(xs, ys_approx, label=f"deg {d} approx")
    ax.set_title("GELU polynomial approximations")
    ax.set_xlabel("x")
    ax.set_ylabel("GELU(x)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "gelu_approximation.png", dpi=300)
    plt.close(fig)

    # Softmax plot: 2-class case as function of logit difference
    deltas = np.linspace(-10.0, 10.0, 1001, dtype=np.float64)
    logits = torch.stack(
        [
            torch.zeros_like(torch.from_numpy(deltas).float()),
            torch.from_numpy(deltas).float(),
        ],
        dim=-1,
    )  # shape (N, 2)
    probs_true = F.softmax(logits, dim=-1)[:, 1].numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(deltas, probs_true, label="softmax true (p2)", color="black", linewidth=2)
    for d in DEGREES:
        probs_approx = softmax_approx(logits, degree=d, dim=-1)[:, 1].numpy()
        ax.plot(deltas, probs_approx, label=f"deg {d} approx")
    ax.set_title("Softmax approximations (2-class case)")
    ax.set_xlabel("logit difference (x2 - x1)")
    ax.set_ylabel("probability of class 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "softmax_approximation.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and visualize polynomial approximations for GELU and softmax.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="activation_plots",
        help="Directory to save approximation plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    _fit_gelu_polynomials()
    _fit_exp_polynomials()

    gelu_mses = compute_mse_gelu()
    softmax_mses = compute_mse_softmax()

    print("GELU approximation MSEs (on [-5, 5]):")
    for d, mse in gelu_mses.items():
        print(f"  degree {d}: MSE = {mse:.6e}")

    print("\nSoftmax approximation MSEs (random logits):")
    for d, mse in softmax_mses.items():
        print(f"  degree {d}: MSE = {mse:.6e}")

    plot_approximations(out_dir)
    print(f"\nApproximation plots written to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()

