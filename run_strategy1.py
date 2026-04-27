"""
run_strategy1.py
================
Strategy 1: Encrypt only the attention-output projection of every
transformer block; all other sub-layers run in plaintext PyTorch.

Flow per transformer block
--------------------------

  token embeddings          (plaintext nn.Embedding)
       │
  ┌────▼────────────────────────────────────────────────┐
  │  TransformerBlock                                    │
  │  ├─ LayerNorm           (plaintext)                 │
  │  ├─ Q projection        (plaintext nn.Linear)       │
  │  ├─ K projection        (plaintext nn.Linear)       │
  │  ├─ V projection        (plaintext nn.Linear)       │
  │  ├─ Scaled dot-product  (plaintext)                 │
  │  ├─ Attention out proj  ← ENCRYPTED (nn.Linear)     │
  │  ├─ Residual add        (plaintext)                 │
  │  ├─ LayerNorm           (plaintext)                 │
  │  ├─ FFN fc1             (plaintext nn.Linear)       │
  │  ├─ GELU                (plaintext)                 │
  │  └─ FFN fc2             (plaintext nn.Linear)       │
  └────────────────────────────────────────────────────-┘
       │
  LM head                   (plaintext nn.Linear)

Privacy guarantee
-----------------
The attention output projection is the linear map that mixes the value
vectors after the softmax weighting. Encrypting it means the server cannot
observe which semantic feature directions are being activated by the
attended tokens — concretely, the token representation *after* attention
mixing is never exposed in plaintext during that layer's computation.
Q, K, V weight matrices and their activations remain in plaintext (they
belong to the model server), but the client's post-attention state is
protected.

Usage
-----
    python run_strategy1.py

    # Optionally override defaults:
    python run_strategy1.py --vocab 256 --d_model 32 --seq 8 --heads 2 \
                            --layers 2 --n_inputs 20 --n_poly 8192 --t 65537

Outputs
-------
Printed to stdout:
  • Per-input latency (HE vs plain)
  • Aggregate timing breakdown
  • Top-5 token overlap (HE vs plain)
  • Cosine similarity between logits (HE vs plain)
  • Strategy 1 analysis summary
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
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


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Tiny Transformer model
# ═══════════════════════════════════════════════════════════════════════════

class SingleHeadAttention(nn.Module):
    """
    Single-head self-attention with separate Q/K/V projection modules.

    Each projection is a named nn.Linear so selective_he_engine can target
    them by name.  The output projection (``out_proj``) is the layer that
    Strategy 1 encrypts.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)   # ← encrypted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, seq_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        scale  = math.sqrt(self.d_model)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale    # (1,S,S)
        attn   = torch.softmax(scores, dim=-1)
        ctx    = torch.matmul(attn, V)                            # (1,S,d)
        # out_proj is run externally by the HE engine; we return ctx here
        # so the engine can intercept it.  In the standalone plain path,
        # out_proj is called in TransformerBlock.forward.
        return ctx


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block with single-head attention + 2-layer FFN.

    Named sub-modules (all leaf nn.Linear layers visible to named_modules):
      attention.q_proj    — Q projection
      attention.k_proj    — K projection
      attention.v_proj    — V projection
      attention.out_proj  — attention output projection  ← ENCRYPTED
      ffn.fc1             — FFN first layer
      ffn.fc2             — FFN second layer
    """

    def __init__(self, d_model: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1     = nn.LayerNorm(d_model)
        self.attention = SingleHeadAttention(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.ffn       = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        h    = self.norm1(x)
        ctx  = self.attention(h)                   # (1, S, d)
        out  = self.attention.out_proj(ctx)         # (1, S, d)  ← encrypted in HE path
        x    = x + out                             # residual
        # Pre-norm FFN
        x    = x + self.ffn(self.norm2(x))
        return x


class TinyTransformerLM(nn.Module):
    """
    Tiny transformer language model:
      Embedding → N × TransformerBlock → LayerNorm → LM head.

    The model is designed so that every nn.Linear whose name matches
    ``*.attention.out_proj`` is the attention output projection.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:    int,
        n_heads:    int,    # kept for API symmetry; model uses single-head
        n_layers:   int,
        ffn_dim:    int,
        seq_len:    int,
    ) -> None:
        super().__init__()
        self.d_model   = d_model
        self.seq_len   = seq_len
        self.embed     = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.blocks    = nn.ModuleList(
            [TransformerBlock(d_model, ffn_dim) for _ in range(n_layers)]
        )
        self.norm      = nn.LayerNorm(d_model)
        self.lm_head   = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (1, seq_len)
        B, S = input_ids.shape
        pos  = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x    = self.embed(input_ids) + self.pos_embed(pos)    # (1, S, d)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)                                # (1, S, vocab)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  HE-aware inference for Strategy 1
#     (we cannot use selective_he_engine directly because the model is NOT
#     a flat sequence of layers — the attention block has internal structure.
#     Instead we write a bespoke mixed-mode forward that calls he_linear
#     only for out_proj and runs everything else in torch.)
# ═══════════════════════════════════════════════════════════════════════════

class Strategy1Timings(NamedTuple):
    plain_time:      float
    encryption_time: float
    he_compute_time: float
    decryption_time: float
    total_time:      float


def _he_out_proj(
    ctx:      torch.Tensor,   # (1, seq_len, d_model)  — post-attention values
    out_proj: nn.Linear,
    HE:       Pyfhel,
    weight_scale: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Encrypt ctx, run out_proj homomorphically, decrypt result.

    We process one token position at a time (shape (1, d_model)) because
    he_linear expects (1, in_features).  The seq_len loop is cheap relative
    to the HE ops themselves.

    Returns
    -------
    out   : (1, seq_len, d_model) float tensor
    times : dict with keys encrypt_ms, he_ms, decrypt_ms
    """
    seq_len, d_model = ctx.shape[1], ctx.shape[2]
    W_np = (
        out_proj.weight.detach().cpu().float().numpy() * weight_scale
    ).round().astype(np.int64).T                    # (d_model, d_model)
    b_np = np.zeros(out_proj.out_features, dtype=np.int64)
    if out_proj.bias is not None:
        b_np = (
            out_proj.bias.detach().cpu().float().numpy() * weight_scale
        ).round().astype(np.int64)

    out_tokens = []
    t_enc = t_he = t_dec = 0.0

    for s in range(seq_len):
        token_vec = ctx[0, s, :].unsqueeze(0)          # (1, d_model) float
        x_int = token_vec.round().to(torch.int64)       # BFV needs ints

        t0 = time.perf_counter()
        enc_x = encrypt_tensor(x_int, HE, batch=False)
        t_enc += time.perf_counter() - t0

        t0 = time.perf_counter()
        enc_y = he_linear(enc_x, W_np, b_np, HE)
        t_he += time.perf_counter() - t0

        t0 = time.perf_counter()
        y_int = decrypt_tensor(enc_y, HE)               # (1, d_model)
        t_dec += time.perf_counter() - t0

        out_tokens.append(y_int.float() / weight_scale)

    out = torch.stack(out_tokens, dim=1)                # (1, S, d_model)
    return out, {
        "encrypt_ms": t_enc * 1e3,
        "he_ms":      t_he  * 1e3,
        "decrypt_ms": t_dec * 1e3,
    }


def strategy1_inference(
    model:        TinyTransformerLM,
    input_ids:    torch.Tensor,
    HE:           Pyfhel,
    weight_scale: int = 1,
) -> tuple[torch.Tensor, Strategy1Timings, list[tuple[str, bool]]]:
    """
    Run Strategy 1 inference: attention out_proj encrypted, rest plaintext.

    Parameters
    ----------
    model        : TinyTransformerLM in eval mode
    input_ids    : (1, seq_len) int64 tensor
    HE           : initialised Pyfhel context
    weight_scale : integer scale for out_proj weights (default 1)

    Returns
    -------
    logits             : (1, seq_len, vocab_size) float tensor
    timings            : Strategy1Timings named tuple
    encryption_state   : list of (layer_name, was_encrypted) for all linear layers
    """
    model.eval()

    t_plain = t_enc = t_he = t_dec = 0.0
    state_log: list[tuple[str, bool]] = []

    with torch.no_grad():
        # ── Embedding (plaintext) ─────────────────────────────────────────
        t0 = time.perf_counter()
        B, S = input_ids.shape
        pos  = torch.arange(S).unsqueeze(0)
        x    = model.embed(input_ids) + model.pos_embed(pos)
        t_plain += time.perf_counter() - t0
        state_log.append(("embedding + pos_embed", False))

        # ── Transformer blocks ────────────────────────────────────────────
        for b_idx, block in enumerate(model.blocks):
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

            # ── Attention out_proj (ENCRYPTED) ────────────────────────────
            out, he_times = _he_out_proj(
                ctx, block.attention.out_proj, HE, weight_scale
            )
            t_enc += he_times["encrypt_ms"] / 1e3
            t_he  += he_times["he_ms"]      / 1e3
            t_dec += he_times["decrypt_ms"] / 1e3
            state_log.append((f"{prefix}.attention.out_proj", True))

            # Residual add (plaintext) — note: out is already float from HE
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

        # ── Final norm + LM head (plaintext) ─────────────────────────────
        t0 = time.perf_counter()
        x      = model.norm(x)
        logits = model.lm_head(x)
        t_plain += time.perf_counter() - t0
        state_log.append(("norm",    False))
        state_log.append(("lm_head", False))

    total = t_plain + t_enc + t_he + t_dec
    timings = Strategy1Timings(
        plain_time      = t_plain,
        encryption_time = t_enc,
        he_compute_time = t_he,
        decryption_time = t_dec,
        total_time      = total,
    )
    return logits, timings, state_log


def plain_inference(
    model:     TinyTransformerLM,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Standard plaintext forward pass. Returns (logits, wall_seconds)."""
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(input_ids)
    return logits, time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Evaluation metrics
# ═══════════════════════════════════════════════════════════════════════════

def top_k_overlap(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    k:        int = 5,
) -> float:
    """
    Fraction of top-k token positions shared between two logit tensors.

    Both tensors are flattened over (seq_len,) for the last token only
    (standard LM "next-token" evaluation).
    """
    last_a = logits_a[0, -1, :]    # (vocab,)
    last_b = logits_b[0, -1, :]
    top_a  = set(torch.topk(last_a, k).indices.tolist())
    top_b  = set(torch.topk(last_b, k).indices.tolist())
    return len(top_a & top_b) / k


def cosine_similarity_logits(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
) -> float:
    """Cosine similarity between the last-token logit vectors."""
    a = logits_a[0, -1, :].float()
    b = logits_b[0, -1, :].float()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Reporting helpers
# ═══════════════════════════════════════════════════════════════════════════

def _bar(frac: float, width: int = 40) -> str:
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


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
    print("  Strategy 1 — Aggregate Timing (avg over {} inputs)".format(len(results)))
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
    print("\n  Layer Encryption State (first transformer block shown)")
    print("  " + "─" * 48)
    seen: set[str] = set()
    for name, enc in state_log:
        # Deduplicate across blocks; show block 0 as representative
        key = name.replace("blocks.0.", "").replace("blocks.1.", "block_N.")
        if key in seen:
            continue
        seen.add(key)
        tag = "🔒 HE    " if enc else "🔓 plain "
        print(f"  {tag} {name}")
    print("  " + "─" * 48)
    n_he    = sum(1 for _, enc in state_log if enc)
    n_plain = len(state_log) - n_he
    print(f"  HE layers total   : {n_he}  ({n_layers} blocks × 1 out_proj × seq_len={seq_len} tokens)")
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


def print_analysis_summary(
    results:     list[dict],
    n_layers:    int,
    seq_len:     int,
    weight_scale: int,
) -> None:
    agg     = {k: np.mean([r[k] for r in results])
               for k in ["plain_ms", "encrypt_ms", "he_ms", "decrypt_ms", "he_total_ms"]}
    overhead = (agg["he_total_ms"] / agg["plain_ms"] - 1) * 100
    enc_pct  = (agg["encrypt_ms"] + agg["decrypt_ms"]) / agg["he_total_ms"] * 100

    print("\n" + "╔" + "═" * 62 + "╗")
    print("║  Strategy 1 Analysis Summary" + " " * 33 + "║")
    print("╠" + "═" * 62 + "╣")

    lines = [
        f"  Target layer    : attention.out_proj (one per block × {n_layers} blocks)",
        f"  Sequence length : {seq_len} tokens processed per inference",
        f"  Weight scale    : {weight_scale}  (weights rounded to int64 multiples)",
        "",
        f"  Latency overhead: +{overhead:.0f}%  "
        f"({agg['he_total_ms']:.1f} ms HE vs {agg['plain_ms']:.1f} ms plain)",
        f"  Enc+Dec share   : {enc_pct:.1f}% of total HE time",
        f"  HE compute share: {100 - enc_pct:.1f}% of total HE time",
        "",
        "  Protected tensors (never exposed in plaintext to server):",
        f"    • Post-attention context vectors — shape (1, {seq_len}, d_model)",
        f"    • out_proj output (pre-residual) — shape (1, {seq_len}, d_model)",
        "",
        "  Plaintext tensors (visible to server, by design):",
        "    • Token embeddings & positional embeddings",
        "    • Q, K, V projection outputs (model's own computation)",
        "    • Attention score matrix  (softmax weights)",
        "    • FFN activations",
        "    • LM head logits",
        "",
        "  Privacy model:",
        "    The server evaluates out_proj without observing the",
        "    post-attention context. The context encodes WHICH tokens",
        "    the model attended to and with what mixture — keeping this",
        "    encrypted prevents the server from inferring attended token",
        "    semantics from the activation directions.",
        "",
        "  Limitations:",
        "    • BFV integer-only: weights are rounded (scale={})".format(weight_scale),
        "    • Rounding error accumulates across layers",
        "    • seq_len HE calls per block (no SIMD batching)",
        "    • Q/K/V + softmax remain plaintext (partial privacy)",
    ]

    for line in lines:
        padded = ("║  " + line + " " * max(0, 59 - len(line)) + "║")
        print(padded)
    print("╚" + "═" * 62 + "╝")


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Main driver
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Strategy 1 HE inference: encrypt attention output projection"
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
                   help="Integer scale for out_proj weights (default 1 = already-int)")
    p.add_argument("--seed",      type=int, default=0,     help="Model parameter seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not _HE_AVAILABLE:
        print(f"[ERROR] HE libraries not available: {_HE_IMPORT_ERR}")
        print("Install with: pip install Pyfhel")
        return

    print("╔" + "═" * 62 + "╗")
    print("║  run_strategy1.py  —  Strategy 1: Encrypt Attention Output  ║")
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

        # HE Strategy 1 inference
        t_wall = time.perf_counter()
        he_logits, timings, state_log = strategy1_inference(
            model, input_ids, HE, weight_scale=args.wscale
        )
        he_wall = time.perf_counter() - t_wall

        last_state_log = state_log

        top5 = top_k_overlap(he_logits, plain_logits, k=5)
        cos  = cosine_similarity_logits(he_logits, plain_logits)

        results.append({
            "idx":          i,
            "plain_ms":     plain_s          * 1e3,
            "encrypt_ms":   timings.encryption_time * 1e3,
            "he_ms":        timings.he_compute_time * 1e3,
            "decrypt_ms":   timings.decryption_time * 1e3,
            "he_total_ms":  he_wall          * 1e3,
            "top5_overlap": top5,
            "cosine_sim":   cos,
        })

        # Progress dot every 5 inputs
        if (i + 1) % 5 == 0 or i == args.n_inputs - 1:
            print(f"    [{i+1:>3}/{args.n_inputs}] last HE={he_wall*1e3:.1f}ms  "
                  f"plain={plain_s*1e3:.2f}ms  top5={top5:.2f}  cos={cos:.4f}")

    # ── Report ─────────────────────────────────────────────────────────────
    print_per_input_table(results, n_show=5)
    print_aggregate_timing(results, last_state_log, args.layers, args.seq, args.d_model)
    print_quality_summary(results)
    print_analysis_summary(results, args.layers, args.seq, args.wscale)

    # Save results for comparison with other strategies
    try:
        import pickle
        with open(".strategy1_results.pkl", "wb") as f:
            pickle.dump(results, f)
        print("  ✓ Results saved to .strategy1_results.pkl")
    except Exception:
        pass

    print("\n  ✓ Strategy 1 evaluation complete.\n")


if __name__ == "__main__":
    main()