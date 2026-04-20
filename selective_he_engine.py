"""
selective_he_engine.py
======================
Mixed plaintext / homomorphic-encryption inference engine.

Overview
--------
``selective_HE_inference`` walks a PyTorch model's named layers in forward-
execution order and applies one of two compute strategies per layer:

  • **Plaintext** — standard ``torch`` forward pass, no crypto overhead.
  • **HE**        — encrypt inputs → run the HE equivalent op from
                    ``he_layers.py`` → decrypt outputs back to int64 tensors.

The caller controls which layers get encrypted via ``SelectiveHEConfig``.

Design philosophy
-----------------
- Only ``nn.Linear`` is supported for HE compute in this file (matching
  ``he_linear`` from he_layers.py).  Passing a non-Linear layer in
  ``layers_to_encrypt`` raises a ``ConfigError``.
- Batch size must be 1 (element-wise BFV encryption).
- Activation layers (ReLU, GELU …) always run in plaintext; they cannot
  be expressed in BFV without polynomial approximations.
- The engine does **not** modify the model in-place; it shadows each layer
  with its own forward logic.

Return values
-------------
``selective_HE_inference`` returns a 3-tuple::

    (logits, timing_dict, encryption_state_log)

    logits               : torch.Tensor  — final model output
    timing_dict          : dict          — aggregated wall-clock seconds:
                             plain_time, encryption_time,
                             he_compute_time, decryption_time
    encryption_state_log : list[tuple]   — [(layer_name, was_encrypted), …]

Usage
-----
See the ``__main__`` block at the bottom for a runnable example.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# ── he_layers imports (must be on the path) ───────────────────────────────────
try:
    from Pyfhel import Pyfhel
    from he_layers import he_linear
    from he_utils import encrypt_tensor, decrypt_tensor, setup_HE_context, _TaggedList
    _PYFHEL_AVAILABLE = True
except ImportError as _e:
    _PYFHEL_AVAILABLE = False
    _PYFHEL_IMPORT_ERR = _e


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConfigError(ValueError):
    """Raised when the SelectiveHEConfig requests an impossible operation."""


class PyfhelUnavailableError(RuntimeError):
    """Raised when Pyfhel is not installed but HE compute is requested."""


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SelectiveHEConfig:
    """
    Configuration for mixed plaintext / HE inference.

    Attributes
    ----------
    layers_to_encrypt : list[str]
        Exact dot-separated module names (as returned by
        ``model.named_modules()``) that should be evaluated homomorphically.
        Example: ``["classifier.0", "classifier.2"]``.

    weight_scale : int
        Integer scale factor applied to weights **before** HE compute.
        Because BFV is integer-only, floating-point weights are rounded to
        ``round(W * weight_scale)``.  The decrypted output is then divided
        by ``weight_scale`` to recover approximate float values.
        Default: 1 (weights treated as already-integer; use larger values
        for float weight models, e.g. 100).

    bias_scale : int
        Scale for bias (should match ``weight_scale`` when input is also
        scaled, or ``weight_scale**2`` when both input and weight are scaled).
        Default: 1.

    input_scale : int
        Scale applied to the *input tensor* before encryption.
        Decrypted output is divided by ``input_scale * weight_scale``.
        Default: 1.

    batch_size_check : bool
        If True (default), raise ``ConfigError`` when batch size > 1.
    """
    layers_to_encrypt: list[str] = field(default_factory=list)
    weight_scale:      int  = 1
    bias_scale:        int  = 1
    input_scale:       int  = 1
    batch_size_check:  bool = True


# ---------------------------------------------------------------------------
# Timing accumulator
# ---------------------------------------------------------------------------

class _TimingAccumulator:
    """Simple container for per-phase wall-clock totals (seconds)."""

    def __init__(self):
        self.plain_time       = 0.0
        self.encryption_time  = 0.0
        self.he_compute_time  = 0.0
        self.decryption_time  = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "plain_time":       self.plain_time,
            "encryption_time":  self.encryption_time,
            "he_compute_time":  self.he_compute_time,
            "decryption_time":  self.decryption_time,
        }

    @property
    def total(self) -> float:
        return (self.plain_time + self.encryption_time +
                self.he_compute_time + self.decryption_time)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_named_layers(model: nn.Module) -> dict[str, nn.Module]:
    """
    Return an ordered dict mapping full dotted name → module for every
    *leaf* layer in the model (i.e. layers with no children of their own).
    """
    return {
        name: mod
        for name, mod in model.named_modules()
        if len(list(mod.children())) == 0 and name != ""
    }


def _validate_config(
    config: SelectiveHEConfig,
    named_layers: dict[str, nn.Module],
) -> None:
    """
    Validate config against the actual model topology.

    Raises ConfigError for:
      • A name in layers_to_encrypt that does not exist in the model.
      • A name in layers_to_encrypt whose layer is not nn.Linear.
      • HE requested but Pyfhel is not installed.
    """
    if config.layers_to_encrypt and not _PYFHEL_AVAILABLE:
        raise PyfhelUnavailableError(
            "HE layers requested but Pyfhel could not be imported.\n"
            f"Original error: {_PYFHEL_IMPORT_ERR}\n"
            "Install with:  pip install Pyfhel"
        )

    for name in config.layers_to_encrypt:
        if name not in named_layers:
            available = list(named_layers.keys())
            raise ConfigError(
                f"Layer '{name}' requested for HE compute was not found in "
                f"the model.\nAvailable leaf layers: {available}"
            )
        layer = named_layers[name]
        if not isinstance(layer, nn.Linear):
            raise ConfigError(
                f"Layer '{name}' is {type(layer).__name__}, but only "
                f"nn.Linear is supported for HE compute.  "
                f"Move activation / norm layers out of layers_to_encrypt."
            )


def _run_he_linear(
    x: torch.Tensor,
    layer: nn.Linear,
    HE: Pyfhel,
    config: SelectiveHEConfig,
    timing: _TimingAccumulator,
) -> torch.Tensor:
    """
    Run a single nn.Linear layer homomorphically.

    Steps
    -----
    1. Scale + cast input to int64.
    2. Encrypt (element-wise, batch=False).
    3. Call ``he_linear`` with scaled int weights + bias.
    4. Decrypt → float tensor, divide by combined scale.

    Parameters
    ----------
    x      : (1, in_features) float tensor  — plaintext activation.
    layer  : nn.Linear whose .weight and .bias we use.
    HE     : initialised Pyfhel context.
    config : holds scale factors.
    timing : mutable accumulator updated in-place.

    Returns
    -------
    torch.Tensor of shape (1, out_features), dtype float32.
    """
    # ── Input scaling & shape check ────────────────────────────────────────
    if x.dim() == 1:
        x = x.unsqueeze(0)                         # → (1, in_features)
    if x.shape[0] != 1:
        raise ConfigError(
            f"HE compute requires batch_size=1, got {x.shape[0]}. "
            "Set config.batch_size_check=True and pass a single sample."
        )

    # Scale input to integer domain
    x_scaled = (x * config.input_scale).round().to(torch.int64)  # (1, in_F)

    # ── Encryption ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    enc_x = encrypt_tensor(x_scaled, HE, batch=False)
    timing.encryption_time += time.perf_counter() - t0

    # ── Prepare plaintext weights / bias ─────────────────────────────────
    # nn.Linear stores weight as (out_features, in_features); he_linear
    # expects (in_features, out_features).
    W_np = (
        layer.weight.detach().cpu().float().numpy() * config.weight_scale
    ).round().astype(np.int64).T          # → (in_F, out_F)

    if layer.bias is not None:
        b_np = (
            layer.bias.detach().cpu().float().numpy() * config.bias_scale
        ).round().astype(np.int64)
    else:
        b_np = np.zeros(layer.out_features, dtype=np.int64)

    # ── HE compute ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    enc_y = he_linear(enc_x, W_np, b_np, HE)
    timing.he_compute_time += time.perf_counter() - t0

    # ── Decryption ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    y_int = decrypt_tensor(enc_y, HE)              # (1, out_F) int64 tensor
    timing.decryption_time += time.perf_counter() - t0

    # Un-scale: divide by the product of both scale factors
    combined_scale = float(config.input_scale * config.weight_scale)
    y_float = y_int.float() / combined_scale       # (1, out_F) float32

    return y_float


def _run_plain_layer(
    layer: nn.Module,
    x: torch.Tensor,
    timing: _TimingAccumulator,
) -> torch.Tensor:
    """Run layer in plaintext, accumulate wall-clock time."""
    t0 = time.perf_counter()
    with torch.no_grad():
        y = layer(x)
    timing.plain_time += time.perf_counter() - t0
    return y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def selective_HE_inference(
    model: nn.Module,
    input_ids: torch.Tensor,
    HE_context: "Pyfhel | None",
    config: SelectiveHEConfig,
) -> tuple[torch.Tensor, dict[str, float], list[tuple[str, bool]]]:
    """
    Run mixed plaintext / HE inference through ``model``.

    Parameters
    ----------
    model : nn.Module
        PyTorch model whose leaf layers are traversed in order.
        The model should be in eval mode; this function calls
        ``model.eval()`` automatically.

    input_ids : torch.Tensor
        Input tensor, shape ``(batch, *)``.  For HE layers batch must be 1.
        May be float or int; HE layers convert internally.

    HE_context : Pyfhel or None
        Initialised Pyfhel object (from ``setup_HE_context``).
        Pass ``None`` only if ``config.layers_to_encrypt`` is empty.

    config : SelectiveHEConfig
        Specifies which layers run under HE and scale factors.

    Returns
    -------
    logits : torch.Tensor
        Final output of the model (same shape as a normal forward pass).

    timing_dict : dict[str, float]
        Aggregated wall-clock seconds for each phase:
        ``plain_time``, ``encryption_time``, ``he_compute_time``,
        ``decryption_time``.  Sum = total wall time of inference.

    encryption_state_log : list[tuple[str, bool]]
        One entry per leaf layer: ``(layer_name, was_encrypted)``.

    Raises
    ------
    ConfigError          : impossible config (bad layer name, wrong type, batch>1).
    PyfhelUnavailableError : HE requested but Pyfhel not installed.
    """
    model.eval()

    # ── Collect leaf layers in model ───────────────────────────────────────
    named_layers = _get_named_layers(model)
    _validate_config(config, named_layers)

    he_set = set(config.layers_to_encrypt)

    # ── Batch size guard ───────────────────────────────────────────────────
    if config.batch_size_check and he_set:
        if input_ids.shape[0] != 1:
            raise ConfigError(
                f"HE compute requires batch_size=1, got {input_ids.shape[0]}. "
                "Slice a single sample: input_ids = input_ids[:1]"
            )

    # ── HE context guard ──────────────────────────────────────────────────
    if he_set and HE_context is None:
        raise ConfigError(
            "config.layers_to_encrypt is non-empty but HE_context is None.  "
            "Pass an initialised Pyfhel object."
        )

    # ── Inference loop ────────────────────────────────────────────────────
    timing = _TimingAccumulator()
    state_log: list[tuple[str, bool]] = []
    x = input_ids

    for layer_name, layer in named_layers.items():
        encrypted = layer_name in he_set

        if encrypted:
            y = _run_he_linear(x, layer, HE_context, config, timing)
        else:
            y = _run_plain_layer(layer, x, timing)

        state_log.append((layer_name, encrypted))
        x = y

    logits = x
    return logits, timing.to_dict(), state_log


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_timing_report(
    timing_dict: dict[str, float],
    encryption_state_log: list[tuple[str, bool]],
) -> None:
    """Print a formatted timing breakdown and encryption state list."""
    total = sum(timing_dict.values())

    print("\n" + "═" * 62)
    print("  Selective HE Inference — Timing Report")
    print("═" * 62)

    rows = [
        ("Plaintext compute",  "plain_time"),
        ("Encryption",         "encryption_time"),
        ("HE compute",         "he_compute_time"),
        ("Decryption",         "decryption_time"),
    ]
    for label, key in rows:
        t = timing_dict[key]
        pct = 100 * t / total if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {label:<22} {t*1e3:>8.2f} ms  [{bar:<50}] {pct:5.1f}%")
    print(f"  {'─'*60}")
    print(f"  {'Total':<22} {total*1e3:>8.2f} ms")

    print("\n  Layer Encryption State Log")
    print("  " + "─" * 44)
    he_count    = sum(1 for _, enc in encryption_state_log if enc)
    plain_count = len(encryption_state_log) - he_count
    for layer_name, encrypted in encryption_state_log:
        tag = "🔒 HE     " if encrypted else "🔓 PLAIN  "
        print(f"  {tag} {layer_name}")
    print("  " + "─" * 44)
    print(f"  HE layers   : {he_count}")
    print(f"  Plain layers: {plain_count}")
    print("═" * 62 + "\n")


# ---------------------------------------------------------------------------
# Demo / acceptance test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 62)
    print("  selective_he_engine.py — Mixed Plaintext / HE Inference Demo")
    print("=" * 62)

    # ── Build a tiny MLP: 8 → 16 → 8 → 4 ────────────────────────────────
    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.fc1  = nn.Linear(8,  16, bias=True)
            self.relu = nn.ReLU()
            self.fc2  = nn.Linear(16,  8, bias=True)
            self.act2 = nn.ReLU()
            self.fc3  = nn.Linear(8,   4, bias=True)

        def forward(self, x):
            return self.fc3(self.act2(self.fc2(self.relu(self.fc1(x)))))

    model = TinyMLP()

    # Integer-valued weights make BFV exact (no rounding error).
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # ── Input ─────────────────────────────────────────────────────────────
    torch.manual_seed(42)
    x = torch.randint(1, 5, (1, 8), dtype=torch.float32)
    print(f"\n  Input tensor (batch=1): {x.tolist()}")

    # ── strategy1: encrypt fc1 and fc3, fc2 in plaintext ─────────────────
    strategy1 = SelectiveHEConfig(
        layers_to_encrypt=["fc1", "fc3"],
        weight_scale=1,
        bias_scale=1,
        input_scale=1,
    )

    print("\n  Setting up HE context (n=2**13, t=65537) …", end=" ", flush=True)
    t0 = time.perf_counter()
    HE = setup_HE_context(n=2 ** 13, t=65537)
    print(f"done ({time.perf_counter()-t0:.2f}s)")

    print("\n  Running selective_HE_inference with strategy1 …")
    print("  (fc1=HE, relu=plain, fc2=plain, act2=plain, fc3=HE)\n")

    logits, timing_dict, state_log = selective_HE_inference(
        model, x, HE, strategy1
    )

    # ── Plaintext reference ───────────────────────────────────────────────
    with torch.no_grad():
        plain_logits = model(x)

    # ── Print results ─────────────────────────────────────────────────────
    print(f"  HE mixed logits     : {logits.flatten().tolist()}")
    print(f"  Plaintext logits    : {plain_logits.flatten().tolist()}")

    diff = (logits.float() - plain_logits.float()).abs().max().item()
    print(f"  Max absolute diff   : {diff:.4f}  "
          f"({'exact' if diff < 1e-3 else 'within float rounding'})")

    print_timing_report(timing_dict, state_log)

    # ── strategy2: all layers in plaintext ───────────────────────────────
    print("  strategy2: all plaintext (no HE) ───────────────────────────")
    strategy2 = SelectiveHEConfig(layers_to_encrypt=[])
    logits2, timing2, log2 = selective_HE_inference(model, x, None, strategy2)
    print(f"  Plaintext-only logits: {logits2.flatten().tolist()}")
    print_timing_report(timing2, log2)

    # ── strategy3: only fc2 in HE ─────────────────────────────────────────
    print("  strategy3: only fc2 in HE ───────────────────────────────────")
    strategy3 = SelectiveHEConfig(
        layers_to_encrypt=["fc2"],
        weight_scale=1,
        bias_scale=1,
        input_scale=1,
    )
    logits3, timing3, log3 = selective_HE_inference(model, x, HE, strategy3)
    print(f"  HE-fc2 logits: {logits3.flatten().tolist()}")
    print_timing_report(timing3, log3)

    # ── Error handling demos ──────────────────────────────────────────────
    print("  Error handling demos")
    print("  ─" * 30)

    # Bad layer name
    try:
        bad_cfg = SelectiveHEConfig(layers_to_encrypt=["nonexistent_layer"])
        selective_HE_inference(model, x, HE, bad_cfg)
    except ConfigError as e:
        print(f"  ✓ Bad layer name caught: {str(e)[:80]}…")

    # Non-Linear layer
    try:
        bad_cfg2 = SelectiveHEConfig(layers_to_encrypt=["relu"])
        selective_HE_inference(model, x, HE, bad_cfg2)
    except ConfigError as e:
        print(f"  ✓ Non-Linear HE caught: {str(e)[:80]}…")

    # Batch size > 1
    try:
        x_batch = torch.randint(1, 5, (3, 8), dtype=torch.float32)
        selective_HE_inference(model, x_batch, HE, strategy1)
    except ConfigError as e:
        print(f"  ✓ Batch > 1 caught: {str(e)[:80]}…")

    # HE_context=None with HE layers
    try:
        selective_HE_inference(model, x, None, strategy1)
    except ConfigError as e:
        print(f"  ✓ None HE context caught: {str(e)[:80]}…")

    print("\n  ✓ All demos and error checks passed")
    print("=" * 62)