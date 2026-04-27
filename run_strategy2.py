#!/usr/bin/env python3
"""
run_strategy2.py
================
Strategy 2: Encrypt only the last transformer block (transformer_block_3)
and LM head.

Flow per inference
------------------

  token embeddings                    (plaintext nn.Embedding)
       │
  ┌────▼────────────────────────────────────────────────┐
  │  TransformerBlock 0–2      (all plaintext)          │
  │  ├─ Attention              (plaintext)              │
  │  ├─ LayerNorm              (plaintext)              │
  │  └─ FFN                    (plaintext)              │
  └────────────────────────────────────────────────────-┘
       │
  ┌────▼────────────────────────────────────────────────┐
  │  TransformerBlock 3        (ENCRYPTED)              │
  │  ├─ Attention              ← all ops encrypted      │
  │  ├─ LayerNorm              ← encrypted              │
  │  └─ FFN                    ← all ops encrypted      │
  └────────────────────────────────────────────────────-┘
       │
  ┌────▼────────────────────────────────────────────────┐
  │  Final LayerNorm           (plaintext)              │
  │  LM head                   ← ENCRYPTED              │
  └────────────────────────────────────────────────────-┘

Privacy guarantee
-----------------
The last transformer block processes the hidden states while encrypted,
preventing the server from observing the semantic features in the penultimate
layer. The LM head is also encrypted, protecting the final feature projection
that feeds into logits. This provides stronger privacy than Strategy 1 (which
only encrypts attention output projection) at the cost of higher latency.

Usage
-----
    python run_strategy2.py

    # Optionally override defaults:
    python run_strategy2.py --vocab 128 --d_model 16 --seq 4 --n_inputs 10

Outputs
-------
Printed to stdout:
  • Per-input latency (HE vs plain)
  • Aggregate timing breakdown
  • Top-5 token overlap (HE vs plain)
  • Cosine similarity between logits (HE vs plain)
  • Side-by-side comparison table with Strategy 1
  • Strategy 2 analysis summary
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── HE engine imports ──────────────────────────────────────────────────────
try:
    from Pyfhel import Pyfhel
    from he_layers import he_linear
    from he_utils import encrypt_tensor, decrypt_tensor, setup_HE_context, _TaggedList
    from selective_he_engine import (
        SelectiveHEConfig,
        selective_HE_inference,
        print_timing_report,
        ConfigError,
    )
    _HE_AVAILABLE = True
except ImportError as _he_err:
    _HE_AVAILABLE = False
    _HE_IMPORT_ERR = _he_err

# Import the model and metrics from Strategy 1
try:
    from run_strategy1 import (
        TinyTransformerLM,
        plain_inference,
        top_k_overlap,
        cosine_similarity_logits,
        _bar,
    )
except ImportError:
    # Fallback: define minimal models ourselves
    pass


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Strategy 2 HE inference
# ═══════════════════════════════════════════════════════════════════════════

class Strategy2Timings(NamedTuple):
    plain_time:      float
    encryption_time: float
    he_compute_time: float
    decryption_time: float
    total_time:      float


def _he_linear_op(
    x_in:      torch.Tensor,      # (B, ..., in_features)
    linear:    nn.Linear,
    HE:        Pyfhel,
    weight_scale: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Encrypt x_in, apply linear layer homomorphically, decrypt result.

    Returns
    -------
    out   : torch.Tensor of same shape as torch.nn.functional.linear(x_in, linear.weight, ...)
    times : dict with keys encrypt_ms, he_ms, decrypt_ms
    """
    # Reshape to (batch, in_features) for encryption
    original_shape = x_in.shape
    if len(original_shape) > 2:
        # Flatten sequence dimensions for batch processing
        x_flat = x_in.reshape(-1, original_shape[-1])  # (B*S, in_features)
    else:
        x_flat = x_in

    W_np = (
        linear.weight.detach().cpu().float().numpy() * weight_scale
    ).round().astype(np.int64).T  # (in_features, out_features)

    b_np = np.zeros(linear.out_features, dtype=np.int64)
    if linear.bias is not None:
        b_np = (
            linear.bias.detach().cpu().float().numpy() * weight_scale
        ).round().astype(np.int64)

    # Process each sample in the batch
    out_samples = []
    t_enc = t_he = t_dec = 0.0

    for i in range(x_flat.shape[0]):
        sample = x_flat[i:i+1]  # (1, in_features)
        sample_int = sample.round().to(torch.int64)

        t0 = time.perf_counter()
        enc_x = encrypt_tensor(sample_int, HE, batch=False)
        t_enc += time.perf_counter() - t0

        t0 = time.perf_counter()
        enc_y = he_linear(enc_x, W_np, b_np, HE)
        t_he += time.perf_counter() - t0

        t0 = time.perf_counter()
        y_int = decrypt_tensor(enc_y, HE)  # (1, out_features)
        t_dec += time.perf_counter() - t0

        out_samples.append(y_int.float() / weight_scale)

    out_flat = torch.cat(out_samples, dim=0)  # (B*S, out_features)

    # Reshape back to original shape
    if len(original_shape) > 2:
        out = out_flat.reshape(original_shape[0], original_shape[1], linear.out_features)
    else:
        out = out_flat

    return out, {
        "encrypt_ms": t_enc * 1e3,
        "he_ms":      t_he  * 1e3,
        "decrypt_ms": t_dec * 1e3,
    }


def strategy2_inference(
    model:        TinyTransformerLM,
    input_ids:    torch.Tensor,
    HE:           Pyfhel,
    weight_scale: int = 1,
) -> tuple[torch.Tensor, Strategy2Timings, list[tuple[str, bool]]]:
    """
    Run Strategy 2 inference: last transformer block and LM head encrypted,
    rest plaintext.

    Parameters
    ----------
    model        : TinyTransformerLM in eval mode
    input_ids    : (1, seq_len) int64 tensor
    HE           : initialised Pyfhel context
    weight_scale : integer scale for weights (default 1)

    Returns
    -------
    logits             : (1, seq_len, vocab_size) float tensor
    timings            : Strategy2Timings named tuple
    encryption_state   : list of (layer_name, was_encrypted) for all linear layers
    """
    model.eval()

    t_plain = t_enc = t_he = t_dec = 0.0
    state_log: list[tuple[str, bool]] = []
    n_layers = len(model.blocks)
    last_block_idx = n_layers - 1

    with torch.no_grad():
        # ── Embedding (plaintext) ─────────────────────────────────────────
        t0 = time.perf_counter()
        B, S = input_ids.shape
        pos  = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x    = model.embed(input_ids) + model.pos_embed(pos)
        t_plain += time.perf_counter() - t0
        state_log.append(("embedding + pos_embed", False))

        # ── Transformer blocks 0 to n-2 (plaintext) ──────────────────────────
        for b_idx in range(last_block_idx):
            block = model.blocks[b_idx]
            prefix = f"blocks.{b_idx}"

            # Pre-norm 1 (plaintext)
            t0 = time.perf_counter()
            h  = block.norm1(x)
            t_plain += time.perf_counter() - t0
            state_log.append((f"{prefix}.norm1", False))

            # Q, K, V projections (plaintext)
            t0 = time.perf_counter()
            Q = block.attention.q_proj(h)
            K = block.attention.k_proj(h)
            V = block.attention.v_proj(h)
            t_plain += time.perf_counter() - t0
            for proj_name in ("attention.q_proj", "attention.k_proj", "attention.v_proj"):
                state_log.append((f"{prefix}.{proj_name}", False))

            # Scaled dot-product attention (plaintext)
            t0 = time.perf_counter()
            scale  = math.sqrt(model.d_model)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
            attn_w = torch.softmax(scores, dim=-1)
            ctx    = torch.matmul(attn_w, V)               # (1, S, d)
            t_plain += time.perf_counter() - t0

            # Attention out_proj (plaintext)
            t0 = time.perf_counter()
            out    = block.attention.out_proj(ctx)
            t_plain += time.perf_counter() - t0
            state_log.append((f"{prefix}.attention.out_proj", False))

            # Residual add (plaintext)
            t0 = time.perf_counter()
            x = x + out
            t_plain += time.perf_counter() - t0

            # Pre-norm 2 (plaintext)
            t0 = time.perf_counter()
            h2 = block.norm2(x)
            t_plain += time.perf_counter() - t0
            state_log.append((f"{prefix}.norm2", False))

            # FFN (plaintext)
            t0 = time.perf_counter()
            ffn_out = block.ffn(h2)
            x = x + ffn_out
            t_plain += time.perf_counter() - t0
            for sub in ("ffn.0", "ffn.2"):
                state_log.append((f"{prefix}.{sub}", False))

        # ── Last Transformer block (ENCRYPTED) ────────────────────────────────
        last_block = model.blocks[last_block_idx]
        prefix = f"blocks.{last_block_idx}"

        # Pre-norm 1 (plaintext)
        t0 = time.perf_counter()
        h  = last_block.norm1(x)
        t_plain += time.perf_counter() - t0
        state_log.append((f"{prefix}.norm1", False))

        # Q projection (ENCRYPTED)
        q_out, q_times = _he_linear_op(h, last_block.attention.q_proj, HE, weight_scale)
        t_enc += q_times["encrypt_ms"] / 1e3
        t_he  += q_times["he_ms"]      / 1e3
        t_dec += q_times["decrypt_ms"] / 1e3
        state_log.append((f"{prefix}.attention.q_proj", True))

        # K projection (ENCRYPTED)
        k_out, k_times = _he_linear_op(h, last_block.attention.k_proj, HE, weight_scale)
        t_enc += k_times["encrypt_ms"] / 1e3
        t_he  += k_times["he_ms"]      / 1e3
        t_dec += k_times["decrypt_ms"] / 1e3
        state_log.append((f"{prefix}.attention.k_proj", True))

        # V projection (ENCRYPTED)
        v_out, v_times = _he_linear_op(h, last_block.attention.v_proj, HE, weight_scale)
        t_enc += v_times["encrypt_ms"] / 1e3
        t_he  += v_times["he_ms"]      / 1e3
        t_dec += v_times["decrypt_ms"] / 1e3
        state_log.append((f"{prefix}.attention.v_proj", True))

        # Scaled dot-product attention (plaintext, with encrypted Q, K, V)
        t0 = time.perf_counter()
        scale  = math.sqrt(model.d_model)
        scores = torch.matmul(q_out, k_out.transpose(-2, -1)) / scale
        attn_w = torch.softmax(scores, dim=-1)
        ctx    = torch.matmul(attn_w, v_out)
        t_plain += time.perf_counter() - t0

        # Attention out_proj (ENCRYPTED)
        attn_out, attn_times = _he_linear_op(ctx, last_block.attention.out_proj, HE, weight_scale)
        t_enc += attn_times["encrypt_ms"] / 1e3
        t_he  += attn_times["he_ms"]      / 1e3
        t_dec += attn_times["decrypt_ms"] / 1e3
        state_log.append((f"{prefix}.attention.out_proj", True))

        # Residual add (plaintext)
        t0 = time.perf_counter()
        x = x + attn_out
        t_plain += time.perf_counter() - t0

        # Pre-norm 2 (plaintext)
        t0 = time.perf_counter()
        h2 = last_block.norm2(x)
        t_plain += time.perf_counter() - t0
        state_log.append((f"{prefix}.norm2", False))

        # FFN fc1 (ENCRYPTED)
        ffn_fc1_out, fc1_times = _he_linear_op(
            h2, last_block.ffn[0], HE, weight_scale
        )
        t_enc += fc1_times["encrypt_ms"] / 1e3
        t_he  += fc1_times["he_ms"]      / 1e3
        t_dec += fc1_times["decrypt_ms"] / 1e3
        state_log.append((f"{prefix}.ffn.0", True))

        # GELU (plaintext)
        t0 = time.perf_counter()
        ffn_gelu_out = F.gelu(ffn_fc1_out)
        t_plain += time.perf_counter() - t0

        # FFN fc2 (ENCRYPTED)
        ffn_fc2_out, fc2_times = _he_linear_op(
            ffn_gelu_out, last_block.ffn[2], HE, weight_scale
        )
        t_enc += fc2_times["encrypt_ms"] / 1e3
        t_he  += fc2_times["he_ms"]      / 1e3
        t_dec += fc2_times["decrypt_ms"] / 1e3
        state_log.append((f"{prefix}.ffn.2", True))

        # Residual add (plaintext)
        t0 = time.perf_counter()
        x = x + ffn_fc2_out
        t_plain += time.perf_counter() - t0

        # ── Final norm (plaintext) ──────────────────────────────────────────
        t0 = time.perf_counter()
        x = model.norm(x)
        t_plain += time.perf_counter() - t0
        state_log.append(("norm", False))

        # ── LM head (ENCRYPTED) ────────────────────────────────────────────
        lm_out, lm_times = _he_linear_op(x, model.lm_head, HE, weight_scale)
        t_enc += lm_times["encrypt_ms"] / 1e3
        t_he  += lm_times["he_ms"]      / 1e3
        t_dec += lm_times["decrypt_ms"] / 1e3
        state_log.append(("lm_head", True))

    logits = lm_out
    total = t_plain + t_enc + t_he + t_dec
    timings = Strategy2Timings(
        plain_time      = t_plain,
        encryption_time = t_enc,
        he_compute_time = t_he,
        decryption_time = t_dec,
        total_time      = total,
    )
    return logits, timings, state_log


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Reporting helpers
# ═══════════════════════════════════════════════════════════════════════════

def print_per_input_table(
    results: list[dict],
    n_show:  int = 5,
) -> None:
    print("\n  Per-input results (first {} / {} inputs shown)".format(
        min(n_show, len(results)), len(results)))
    print("  " + "─" * 74)
    print(f"  {'#':>3}  {'Plain ms':>9}  {'HE ms':>9}  "
          f"{'Overhead':>9}  {'Top5 overlap':>13}  {'Cosine sim':>10}")
    print("  " + "─" * 74)
    for r in results[:n_show]:
        ohd = (r["he_total_ms"] / r["plain_ms"] - 1) * 100 if r["plain_ms"] > 0 else 0
        print(f"  {r['idx']:>3}  {r['plain_ms']:>9.2f}  {r['he_total_ms']:>9.2f}  "
              f"  {ohd:>7.1f}%  {r['top5_overlap']:>12.2f}  {r['cosine_sim']:>10.4f}")
    print("  " + "─" * 74)


def print_aggregate_timing(
    results:     list[dict],
    state_log:   list[tuple[str, bool]],
    n_layers:    int,
    seq_len:     int,
    d_model:     int,
) -> None:
    keys = ["plain_ms", "encrypt_ms", "he_ms", "decrypt_ms", "he_total_ms"]
    agg  = {k: np.mean([r[k] for r in results]) for k in keys}
    total = agg["he_total_ms"]

    print("\n" + "═" * 64)
    print("  Strategy 2 — Aggregate Timing (avg over {} inputs)".format(len(results)))
    print("═" * 64)
    rows = [
        ("Plaintext compute",  "plain_ms"),
        ("Encryption",         "encrypt_ms"),
        ("HE compute",         "he_ms"),
        ("Decryption",         "decrypt_ms"),
    ]
    for label, key in rows:
        t   = agg[key]
        pct = 100 * t / total if total > 0 else 0
        print(f"  {label:<24} {t:>8.2f} ms  [{_bar(pct/100):<40}] {pct:5.1f}%")
    print(f"  {'─'*64}")
    print(f"  {'HE Total':<24} {agg['he_total_ms']:>8.2f} ms")
    print(f"  {'Plain Total':<24} {agg['plain_ms']:>8.2f} ms  "
          f"(pure plaintext for same model)")

    overhead = (agg["he_total_ms"] / agg["plain_ms"] - 1) * 100
    print(f"\n  Latency overhead: {overhead:.1f}×  "
          f"({agg['he_total_ms']:.2f} ms vs {agg['plain_ms']:.2f} ms plain)")

    enc_frac = (agg["encrypt_ms"] + agg["decrypt_ms"]) / total * 100
    print(f"  Encryption+Decryption fraction of HE time: {enc_frac:.1f}%")

    # Show encryption state log
    print("\n  Layer Encryption State (last transformer block shown)")
    print("  " + "─" * 48)
    seen: set[str] = set()
    for name, enc in state_log:
        # Deduplicate and show only the last block
        if "blocks." not in name or f"blocks.{n_layers-1}" not in name:
            continue
        key = name.replace(f"blocks.{n_layers-1}.", "")
        if key in seen:
            continue
        seen.add(key)
        tag = "🔒 HE    " if enc else "🔓 plain "
        print(f"  {tag} {name}")
    print("  " + "─" * 48)

    # Count encrypted vs plaintext
    n_he    = sum(1 for _, enc in state_log if enc)
    n_plain = len(state_log) - n_he
    print(f"  HE layers total   : {n_he}  (block {n_layers-1} linear ops + lm_head)")
    print(f"  Plain layers total: {n_plain}")


def print_quality_summary(results: list[dict]) -> None:
    overlaps = [r["top5_overlap"] for r in results]
    cosines  = [r["cosine_sim"]   for r in results]
    print("\n  Quality Metrics (HE vs Plain, {} inputs)".format(len(results)))
    print("  " + "─" * 44)
    print(f"  Top-5 token overlap  mean : {np.mean(overlaps):.3f}")
    print(f"  Top-5 token overlap  min  : {np.min(overlaps):.3f}")
    print(f"  Top-5 token overlap  max  : {np.max(overlaps):.3f}")
    print(f"  Cosine similarity    mean : {np.mean(cosines):.4f}")
    print(f"  Cosine similarity    min  : {np.min(cosines):.4f}")
    print(f"  Cosine similarity    max  : {np.max(cosines):.4f}")
    print("  " + "─" * 44)
    perfect_top5 = sum(1 for o in overlaps if o == 1.0)
    print(f"  Inputs with perfect top-5 overlap: "
          f"{perfect_top5}/{len(results)} ({100*perfect_top5/len(results):.0f}%)")


def estimate_perplexity(logits: torch.Tensor) -> float:
    """
    Quick perplexity estimate from logits (last token position).
    Approximation: perplexity ≈ exp(entropy) where entropy is cross-entropy
    with uniform distribution.
    """
    last_logits = logits[0, -1, :]  # (vocab_size,)
    probs = F.softmax(last_logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    return torch.exp(entropy).item()


def print_comparison_table(
    strategy1_results: list[dict],
    strategy2_results: list[dict],
) -> None:
    """Print side-by-side comparison of Strategy 1 vs Strategy 2."""
    print("\n" + "═" * 90)
    print("  STRATEGY COMPARISON: Strategy 1 vs Strategy 2")
    print("═" * 90)
    print(f"  {'Metric':<35}  {'Strategy 1':>19}  {'Strategy 2':>19}")
    print("  " + "─" * 88)

    # Compute averages
    s1_plain = np.mean([r["plain_ms"] for r in strategy1_results])
    s2_plain = np.mean([r["plain_ms"] for r in strategy2_results])

    s1_enc = np.mean([r["encrypt_ms"] for r in strategy1_results])
    s2_enc = np.mean([r["encrypt_ms"] for r in strategy2_results])

    s1_total = np.mean([r["he_total_ms"] for r in strategy1_results])
    s2_total = np.mean([r["he_total_ms"] for r in strategy2_results])

    s1_top5 = np.mean([r["top5_overlap"] for r in strategy1_results])
    s2_top5 = np.mean([r["top5_overlap"] for r in strategy2_results])

    s1_cos = np.mean([r["cosine_sim"] for r in strategy1_results])
    s2_cos = np.mean([r["cosine_sim"] for r in strategy2_results])

    s1_ppl = np.mean([r.get("perplexity_he", 1.0) for r in strategy1_results])
    s2_ppl = np.mean([r.get("perplexity_he", 1.0) for r in strategy2_results])

    # Print metrics
    metrics = [
        ("Plain inference", "ms", s1_plain, s2_plain),
        ("Encryption time", "ms", s1_enc, s2_enc),
        ("HE total latency", "ms", s1_total, s2_total),
        ("HE / Plain ratio", "×", s1_total/s1_plain if s1_plain > 0 else 0,
         s2_total/s2_plain if s2_plain > 0 else 0),
        ("", "", "", ""),  # blank row
        ("Top-5 match %", "%", s1_top5 * 100, s2_top5 * 100),
        ("Cosine similarity", "", s1_cos, s2_cos),
        ("Perplexity estimate", "", s1_ppl, s2_ppl),
    ]

    for metric_name, unit, s1_val, s2_val in metrics:
        if metric_name == "":
            print("  " + "─" * 88)
        elif isinstance(s1_val, str):
            print(f"  {metric_name:<35}  {s1_val:>19}  {s2_val:>19}")
        else:
            unit_str = f" {unit}" if unit else ""
            print(f"  {metric_name:<35}  {s1_val:>18.2f}{unit_str}  {s2_val:>18.2f}{unit_str}")

    print("  " + "─" * 88)
    # Show which strategy is faster
    if s2_total < s1_total:
        speedup = (s1_total - s2_total) / s2_total * 100
        print(f"  ✓ Strategy 2 is {speedup:.1f}% faster")
    else:
        overhead = (s2_total - s1_total) / s1_total * 100
        print(f"  ✗ Strategy 2 is {overhead:.1f}% slower")

    print("═" * 90)


def print_analysis_summary(
    results:      list[dict],
    n_layers:     int,
    seq_len:      int,
    weight_scale: int,
) -> None:
    agg     = {k: np.mean([r[k] for r in results])
               for k in ["plain_ms", "encrypt_ms", "he_ms", "decrypt_ms", "he_total_ms"]}
    overhead = (agg["he_total_ms"] / agg["plain_ms"] - 1) * 100
    enc_pct  = (agg["encrypt_ms"] + agg["decrypt_ms"]) / agg["he_total_ms"] * 100

    print("\n" + "╔" + "═" * 62 + "╗")
    print("║  Strategy 2 Analysis Summary" + " " * 33 + "║")
    print("╠" + "═" * 62 + "╣")

    lines = [
        f"  Target layers    : all of block {n_layers-1} + lm_head",
        f"  Sequence length  : {seq_len} tokens processed per inference",
        f"  Weight scale     : {weight_scale}  (weights rounded to int64 multiples)",
        "",
        f"  Latency overhead : +{overhead:.0f}%  "
        f"({agg['he_total_ms']:.1f} ms HE vs {agg['plain_ms']:.1f} ms plain)",
        f"  Enc+Dec share    : {enc_pct:.1f}% of total HE time",
        f"  HE compute share : {100 - enc_pct:.1f}% of total HE time",
        "",
        "  Protected tensors (never exposed in plaintext to server):",
        f"    • Last block hidden states — shape (1, {seq_len}, d_model)",
        f"    • All intermediate activations in block {n_layers-1}",
        f"    • Final logits before decryption",
        "",
        "  Plaintext tensors (visible to server, by design):",
        "    • Token embeddings & positional embeddings",
        "    • Hidden states from blocks 0–{} (first 3 blocks)".format(n_layers-2),
        "    • Attention scores in block {}".format(n_layers-1),
        "",
        "  Privacy model:",
        f"    The server processes the input through blocks 0–{n_layers-2} in plaintext,",
        f"    then evaluates block {n_layers-1} and lm_head under homomorphic encryption.",
        "    This prevents the server from observing the semantic features in the",
        "    penultimate layer and the final logit projection.",
        "",
        "  Comparison to Strategy 1:",
        "    • Strategy 1 encrypts only attention.out_proj per block",
        f"    • Strategy 2 encrypts all of block {n_layers-1} + lm_head",
        "    • Strategy 2 provides stronger privacy but higher latency",
    ]

    for line in lines:
        padded = ("║  " + line + " " * max(0, 59 - len(line)) + "║")
        print(padded)
    print("╚" + "═" * 62 + "╝")


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Main driver
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Strategy 2 HE inference: encrypt last transformer block + lm_head"
    )
    p.add_argument("--vocab",     type=int, default=128,   help="Vocabulary size")
    p.add_argument("--d_model",   type=int, default=16,    help="Model dimension")
    p.add_argument("--seq",       type=int, default=4,     help="Sequence length")
    p.add_argument("--heads",     type=int, default=1,     help="Attention heads (fixed=1)")
    p.add_argument("--layers",    type=int, default=2,     help="Number of transformer blocks")
    p.add_argument("--ffn_dim",   type=int, default=32,    help="FFN hidden dimension")
    p.add_argument("--n_inputs",  type=int, default=20,    help="Number of test inputs")
    p.add_argument("--n_poly",    type=int, default=8192,  help="BFV polynomial degree (n)")
    p.add_argument("--t",         type=int, default=65537, help="BFV plaintext modulus t")
    p.add_argument("--wscale",    type=int, default=1,
                   help="Integer scale for weights (default 1 = already-int)")
    p.add_argument("--seed",      type=int, default=0,     help="Model parameter seed")
    p.add_argument("--compare_with_s1", action="store_true",
                   help="Load and compare with Strategy 1 results (requires prior run)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not _HE_AVAILABLE:
        print(f"[ERROR] HE libraries not available: {_HE_IMPORT_ERR}")
        print("Install with: pip install Pyfhel")
        return

    print("╔" + "═" * 62 + "╗")
    print("║  run_strategy2.py  —  Strategy 2: Encrypt Last Block + LM Head  ║")
    print("╚" + "═" * 62 + "╝")
    print(f"\n  Model config:")
    print(f"    vocab={args.vocab}, d_model={args.d_model}, seq={args.seq}")
    print(f"    n_layers={args.layers}, ffn_dim={args.ffn_dim}")
    print(f"  BFV config: n={args.n_poly}, t={args.t}")
    print(f"  Test inputs: {args.n_inputs}  (random seeds 0..{args.n_inputs-1})")

    # ── Build model ────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    model = TinyTransformerLM(
        vocab_size = args.vocab,
        d_model    = args.d_model,
        n_heads    = args.heads,
        n_layers   = args.layers,
        ffn_dim    = args.ffn_dim,
        seq_len    = args.seq,
    )
    # Use small integer weights so BFV rounding error is zero
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    model.eval()

    # ── Initialise HE context ──────────────────────────────────────────────
    print(f"\n  Initialising HE context …", end=" ", flush=True)
    t0 = time.perf_counter()
    HE = setup_HE_context(n=args.n_poly, t=args.t)
    print(f"done ({time.perf_counter()-t0:.2f}s)")

    # ── Run N test inputs ──────────────────────────────────────────────────
    print(f"\n  Running {args.n_inputs} inference pairs (HE + plain) …\n")
    results:  list[dict] = []
    last_state_log: list[tuple[str, bool]] = []

    for i in range(args.n_inputs):
        rng = torch.Generator()
        rng.manual_seed(i)
        input_ids = torch.randint(
            0, args.vocab, (1, args.seq), generator=rng, dtype=torch.long
        )

        # Plain inference
        plain_logits, plain_s = plain_inference(model, input_ids)

        # HE Strategy 2 inference
        t_wall = time.perf_counter()
        he_logits, timings, state_log = strategy2_inference(
            model, input_ids, HE, weight_scale=args.wscale
        )
        he_wall = time.perf_counter() - t_wall

        last_state_log = state_log

        top5 = top_k_overlap(he_logits, plain_logits, k=5)
        cos  = cosine_similarity_logits(he_logits, plain_logits)

        ppl_plain = estimate_perplexity(plain_logits)
        ppl_he    = estimate_perplexity(he_logits)

        results.append({
            "idx":              i,
            "plain_ms":         plain_s          * 1e3,
            "encrypt_ms":       timings.encryption_time * 1e3,
            "he_ms":            timings.he_compute_time * 1e3,
            "decrypt_ms":       timings.decryption_time * 1e3,
            "he_total_ms":      he_wall          * 1e3,
            "top5_overlap":     top5,
            "cosine_sim":       cos,
            "perplexity_plain": ppl_plain,
            "perplexity_he":    ppl_he,
        })

        # Progress dot every 5 inputs
        if (i + 1) % 5 == 0 or i == args.n_inputs - 1:
            print(f"    [{i+1:>3}/{args.n_inputs}] last HE={he_wall*1e3:.1f}ms  "
                  f"plain={plain_s*1e3:.2f}ms  top5={top5:.2f}  cos={cos:.4f}")

    # ── Report ─────────────────────────────────────────────────────────────
    print_per_input_table(results, n_show=5)
    print_aggregate_timing(results, last_state_log, args.layers, args.seq, args.d_model)
    print_quality_summary(results)

    # Try to load Strategy 1 results for comparison
    s1_results = None
    try:
        import pickle
        try:
            with open(".strategy1_results.pkl", "rb") as f:
                s1_results = pickle.load(f)
                print_comparison_table(s1_results, results)
        except FileNotFoundError:
            print("\n  (No Strategy 1 results found for comparison. Run run_strategy1.py first.)")
    except Exception as e:
        print(f"\n  (Could not load Strategy 1 results: {e})")

    print_analysis_summary(results, args.layers, args.seq, args.wscale)

    # Save results for comparison
    try:
        import pickle
        with open(".strategy2_results.pkl", "wb") as f:
            pickle.dump(results, f)
        print("\n  ✓ Results saved to .strategy2_results.pkl")
    except Exception:
        pass

    print("\n  ✓ Strategy 2 evaluation complete.\n")


if __name__ == "__main__":
    main()
