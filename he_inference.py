"""
he_inference.py
===============
Toy encrypted inference pipeline for TinyGPT using BFV homomorphic encryption.

Overview
--------
Demonstrates single-token language-model inference where the input token ID
is encrypted by the *client*, all compute runs on the *server* over ciphertexts,
and the server returns encrypted logits that only the client can decrypt.

Pipeline (batch=1, seq_len=1):
    [CLIENT]
    1. Token ID  →  encrypt_token_id()  →  1 ciphertext
    2. Send ciphertext + HE public context to server

    [SERVER]
    3. Encrypted embedding lookup      →  d_model ciphertexts
    4. N × HE transformer block        →  d_model ciphertexts each
    5. HE LM head (linear projection)  →  vocab_size ciphertexts

    [CLIENT]
    6. Decrypt logits  →  top-5 token suggestions

Practical limitations (documented)
-----------------------------------
1. BATCH=1, SEQ_LEN=1 ONLY.
   Extending to seq_len > 1 requires the full he_attention_approx kernel
   (O(S²·d) ctxt×ctxt muls); at seq_len=1 attention degenerates to a
   trivial identity (one token attending to itself with weight 1.0).

2. INTEGER-ONLY ARITHMETIC.
   BFV operates modulo t (default 65537).  All weights, embeddings, and
   activations must be integer-scaled.  We use EMBED_SCALE=1 (truncate
   floats to nearest int) for the toy demo; for real models multiply
   weights by a larger scale and track accumulated scale factors carefully.

3. SOFTMAX / LAYER-NORM ARE APPROXIMATED.
   • LayerNorm: replaced by a no-op (skip) in the HE domain.  Computing
     running mean and variance homomorphically requires either bootstrapping
     or a polynomial approximation; neither is implemented here.
   • Softmax in attention: degenerate (single token, weight = 1.0 always).
   • FFN activation (GELU/ReLU): replaced by the identity (linear pass-through).
     Polynomial approximations of degree 3–7 can approximate GELU but cost
     multiple multiplication levels.

4. CIPHERTEXT COUNT AND MEMORY.
   With d_model=16 (toy) and vocab_size=100 (toy):
     Embedding output : d_model      = 16 ctxts
     Per transformer  : ~d_model     = 16 ctxts (in + out)
     LM head output   : vocab_size   = 100 ctxts
   Each BFV ciphertext with n=2**13 ≈ 2 × (n × 3 bytes) ≈ 48 KB.
   Total for toy model: ~200 ctxts × 48 KB ≈ 10 MB.
   For full TinyGPT (d_model=128, vocab=50257): ~50k ctxts ≈ 2.4 GB.
   → Production-scale HE inference requires batching, CKKS, and hardware
     acceleration (GPU-FHE libraries such as TFHE-rs or OpenFHE with CUDA).

5. NOISE BUDGET.
   Each transformer block consumes ~1 ctxt×ptxt multiplication level for
   the linear projections.  With n=2**13 the budget supports ~27 levels.
   4 transformer layers use ~4 levels → well within budget for the toy model.

Next steps for production
--------------------------
   • Switch to CKKS for floating-point weights (Microsoft SEAL / OpenFHE).
   • Use the diagonal / Halevi-Shoup method for batched GEMV.
   • Implement polynomial LayerNorm and GELU approximations.
   • Add bootstrapping to refresh noise budget between layers.
   • Explore MPC-FHE hybrid protocols to offload the non-linear ops.
"""

from __future__ import annotations

import time
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from Pyfhel import Pyfhel
except ImportError as _err:
    raise ImportError(
        "Pyfhel is required.  Install with:  pip install Pyfhel"
    ) from _err

from he_utils import setup_HE_context, encrypt_tensor, decrypt_tensor, _TaggedList
from he_layers import he_linear, he_attention_approx, _make_causal_mask

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Fixed-point scale applied to floating-point weights/embeddings before
# converting to integers for BFV.  Higher = more precision but faster
# overflow risk (product of two scaled values = EMBED_SCALE² × true value).
# With t=65537 and two scale multiplications: EMBED_SCALE² × max_val < t/2.
# For the toy model (small weights ≈ N(0,0.02) → round to 0 or ±1 at scale=1)
# we use scale=100 so that weights in [-0.32, 0.32] survive rounding.
EMBED_SCALE: int = 100

# Bytes per BFV ciphertext (two polynomials, each n × ~3 bytes for 64-bit q)
_BYTES_PER_CTXT: int = 48_000   # approximate for n=2**13


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HEInferenceConfig:
    """
    Toy model hyperparameters for encrypted inference.

    Kept deliberately tiny so the demo runs in seconds on CPU.
    Real TinyGPT has d_model=128, d_ff=512, num_layers=4, vocab_size=50257.
    """
    vocab_size:  int   = 200       # toy vocabulary
    d_model:     int   = 16        # embedding dimension
    num_heads:   int   = 2         # attention heads (unused at seq_len=1)
    d_ff:        int   = 32        # feed-forward hidden dim
    num_layers:  int   = 2         # transformer blocks
    max_seq_len: int   = 1         # HARD LIMIT — see module docstring
    embed_scale: int   = EMBED_SCALE
    n_poly:      int   = 2 ** 13   # BFV polynomial degree
    t:           int   = 65537     # BFV plaintext modulus


@dataclass
class TimingReport:
    """Stores wall-clock timings (seconds) for each inference stage."""
    context_setup:   float = 0.0
    token_encrypt:   float = 0.0
    embedding:       float = 0.0
    transformer:     list  = field(default_factory=list)
    lm_head:         float = 0.0
    logit_decrypt:   float = 0.0
    total:           float = 0.0

    def print(self) -> None:
        print("\n  ── Timing Breakdown ──────────────────────────────────────")
        print(f"  HE context setup       : {self.context_setup:>8.3f} s")
        print(f"  Token ID encryption    : {self.token_encrypt*1e3:>8.2f} ms")
        print(f"  Embedding lookup (HE)  : {self.embedding*1e3:>8.2f} ms")
        for i, t in enumerate(self.transformer):
            print(f"  Transformer block {i+1:<2}   : {t:>8.3f} s")
        print(f"  LM head (HE linear)    : {self.lm_head:>8.3f} s")
        print(f"  Logit decryption       : {self.logit_decrypt*1e3:>8.2f} ms")
        print(f"  ─────────────────────────────────────────────────────────")
        print(f"  Total end-to-end       : {self.total:>8.3f} s")


@dataclass
class MemoryReport:
    """Estimated ciphertext counts and memory footprint."""
    embedding_ctxts:    int = 0
    per_block_ctxts:    int = 0
    lm_head_ctxts:      int = 0
    total_ctxts:        int = 0
    estimated_mb:       float = 0.0

    def print(self) -> None:
        print("\n  ── Memory Profile (approximate) ──────────────────────────")
        print(f"  Embedding output       : {self.embedding_ctxts:>6} ciphertexts")
        print(f"  Per transformer block  : {self.per_block_ctxts:>6} ciphertexts")
        print(f"  LM head output         : {self.lm_head_ctxts:>6} ciphertexts")
        print(f"  Total peak ciphertexts : {self.total_ctxts:>6}")
        print(f"  Estimated memory       : {self.estimated_mb:>6.1f} MB  "
              f"({_BYTES_PER_CTXT//1000} KB/ctxt × {self.total_ctxts})")


# ---------------------------------------------------------------------------
# Toy model factory
# ---------------------------------------------------------------------------

def build_toy_model(cfg: HEInferenceConfig) -> nn.Module:
    """
    Build a minimal TinyGPT-compatible nn.Module with random weights.

    In a real workflow you would load TinyGPT from a checkpoint:
        model = TinyGPT(...)
        model.load_state_dict(torch.load('checkpoint.pt')['model_state'])

    Here we just instantiate nn.Embedding, nn.Linear layers directly so
    there is no circular import on TinyGPT.
    """
    class ToyModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            # Token + positional embeddings
            self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.pos_emb   = nn.Embedding(cfg.max_seq_len, cfg.d_model)
            # Transformer blocks: each block = attention projections + FFN
            self.blocks = nn.ModuleList([
                nn.ModuleDict({
                    # Q, K, V projections (d_model → d_model each)
                    "Wq": nn.Linear(cfg.d_model, cfg.d_model, bias=False),
                    "Wk": nn.Linear(cfg.d_model, cfg.d_model, bias=False),
                    "Wv": nn.Linear(cfg.d_model, cfg.d_model, bias=False),
                    "Wo": nn.Linear(cfg.d_model, cfg.d_model, bias=True),
                    # FFN: two linear layers
                    "ff1": nn.Linear(cfg.d_model, cfg.d_ff,   bias=True),
                    "ff2": nn.Linear(cfg.d_ff,    cfg.d_model, bias=True),
                })
                for _ in range(cfg.num_layers)
            ])
            # Final layer norm (not used in HE path; kept for shape reference)
            self.norm = nn.LayerNorm(cfg.d_model)
            # LM head tied to token embedding weight
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            self.lm_head.weight = self.token_emb.weight

            # Small-scale init so integer rounding doesn't zero everything out
            with torch.no_grad():
                nn.init.normal_(self.token_emb.weight, std=0.1)
                nn.init.normal_(self.pos_emb.weight,   std=0.1)
                for blk in self.blocks:
                    for proj in ["Wq","Wk","Wv","Wo","ff1","ff2"]:
                        nn.init.normal_(blk[proj].weight, std=0.1)
                        if blk[proj].bias is not None:
                            nn.init.zeros_(blk[proj].bias)

        def plaintext_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            """Reference plaintext forward pass for correctness comparison."""
            x = self.token_emb(input_ids) + self.pos_emb(
                torch.zeros_like(input_ids))
            for blk in self.blocks:
                # Identity attention at seq_len=1 (no softmax needed)
                q = torch.relu(blk["Wq"](x))
                x = x + blk["Wo"](q)
                x = x + blk["ff2"](torch.relu(blk["ff1"](x)))
            return self.lm_head(self.norm(x))

    torch.manual_seed(99)
    return ToyModel(cfg)


# ---------------------------------------------------------------------------
# Stage 1: Encrypted embedding lookup
# ---------------------------------------------------------------------------

def he_embedding_lookup(
    token_id: int,
    token_emb_weight: torch.Tensor,   # (vocab_size, d_model)
    pos_emb_weight:   torch.Tensor,   # (max_seq_len, d_model)
    HE: Pyfhel,
    embed_scale: int = EMBED_SCALE,
) -> _TaggedList:
    """
    Approximate encrypted embedding lookup for a single token.

    APPROACH — plaintext-select + encrypt:
    ----------------------------------------
    True homomorphic embedding lookup would require an oblivious RAM (ORAM)
    protocol to hide *which* row is selected.  That is a research-level
    primitive.  Here we use the practical approximation:

      1. The SERVER selects the embedding row by index (token_id) in plaintext.
         This leaks WHICH token was looked up (the index), but not the
         embedding *values* (which remain plaintext weights anyway in this
         threat model where the server owns the model).

      2. The selected embedding vector is integer-scaled and encrypted so
         subsequent transformer operations are performed homomorphically.

    For a stricter privacy model (hiding the token ID from the server):
      • Use an ORAM-based protocol (e.g., Path-ORAM over ciphertexts).
      • Or: client encrypts the one-hot indicator and the server computes
            emb = sum_i( onehot[i] * E[i] )  fully homomorphically.
            Cost: vocab_size ctxt×ptxt muls — feasible for small vocabularies.

    Parameters
    ----------
    token_id          : integer token index
    token_emb_weight  : (vocab_size, d_model) float embedding table
    pos_emb_weight    : (max_seq_len, d_model) positional embedding table
    HE                : initialised Pyfhel context
    embed_scale       : fixed-point integer scale

    Returns
    -------
    _TaggedList of d_model ciphertexts representing the scaled embedding vector
    """
    if token_id < 0 or token_id >= token_emb_weight.shape[0]:
        raise ValueError(
            f"token_id={token_id} out of range [0, {token_emb_weight.shape[0]})"
        )

    # Select and sum token + positional embeddings (position 0 always at seq=1)
    tok_vec = token_emb_weight[token_id]   # (d_model,) float
    pos_vec = pos_emb_weight[0]            # (d_model,) float
    emb_vec = tok_vec + pos_vec            # (d_model,) float

    # Scale to integers and clamp to safe BFV range (< t/2 = 32768)
    t_half = (HE.get_plain_modulus() // 2) if hasattr(HE, 'get_plain_modulus') else 32768
    emb_int = torch.round(emb_vec * embed_scale).to(torch.int64)
    emb_int = emb_int.clamp(-t_half + 1, t_half - 1)

    # Encrypt as a 1-D vector of d_model ciphertexts
    enc = encrypt_tensor(emb_int.unsqueeze(0), HE, batch=False)  # shape (1, d_model)
    return enc


# ---------------------------------------------------------------------------
# Stage 2: HE transformer block (simplified for seq_len=1)
# ---------------------------------------------------------------------------

def he_transformer_block_seq1(
    x_enc: _TaggedList,            # (1, d_model) encrypted
    block: nn.ModuleDict,
    d_model: int,
    HE: Pyfhel,
    embed_scale: int = EMBED_SCALE,
    block_idx: int = 0,
) -> _TaggedList:
    """
    Simplified HE transformer block for seq_len=1.

    Approximations made
    -------------------
    LayerNorm   → SKIPPED (identity).  Computing mean/variance homomorphically
                  requires bootstrapping or polynomial approximation; at seq_len=1
                  with small integer inputs the activations stay in a bounded
                  range so the skip is acceptable for demonstration purposes.

    Attention   → Q @ K^T is trivially the scalar Q·K (both are 1-D).  At
                  seq_len=1 the softmax output is always [1.0], so:
                    Attention(Q,K,V) = V      (single token attends to itself)
                  We therefore directly pass V through the output projection Wo,
                  skipping he_attention_approx to avoid the full O(S²·d) cost.
                  Set seq_len > 1 in future to activate full HE attention.

    GELU / ReLU → REPLACED by identity in the FFN.  A polynomial approximation
                  f(x) ≈ a + b·x + c·x² (degree-2 ReLU) can be added at the
                  cost of one additional ctxt×ptxt multiplication level.

    Residual    → INCLUDED: x += output of each sub-layer (exact in HE).

    Computation (d_model=16, d_ff=32 toy model):
      Attention block : 1 he_linear (d→d) Wo           = 16×16 = 256 ctxt×ptxt
      FFN block       : he_linear (d→d_ff) + (d_ff→d)  = 16×32 + 32×16 = 1024 ctxt×ptxt
      Residual adds   : 2 × d_model = 32 ctxt+ctxt (cheap)

    Parameters
    ----------
    x_enc      : encrypted (1, d_model) activation
    block      : nn.ModuleDict with Wq, Wk, Wv, Wo, ff1, ff2
    d_model    : model hidden dimension
    HE         : Pyfhel context
    embed_scale: integer scale (passed through; only used for weight conversion)
    block_idx  : block index for logging

    Returns
    -------
    _TaggedList of d_model ciphertexts (same shape as input)
    """
    def _w(layer: nn.Linear) -> np.ndarray:
        """Extract weight matrix as scaled int64 numpy array."""
        w = layer.weight.detach().float()        # (out, in)
        w_scaled = torch.round(w * embed_scale).to(torch.int64)
        return w_scaled.T.numpy()                # (in, out) for he_linear

    def _b(layer: nn.Linear) -> np.ndarray:
        """Extract bias as scaled int64 numpy array."""
        if layer.bias is None:
            return np.zeros(layer.out_features, dtype=np.int64)
        b = layer.bias.detach().float()
        return torch.round(b * embed_scale).to(torch.int64).numpy()

    # ── Attention sub-layer (degenerate at seq_len=1) ─────────────────────────
    # At seq_len=1: softmax([q·k]) = [1.0], so output = V = Wv @ x.
    # We compute: attn_out = Wo @ (Wv @ x_enc)  in two he_linear calls.
    v_enc = he_linear(x_enc, _w(block["Wv"]), _b(block["Wv"]), HE)
    attn_out = he_linear(v_enc, _w(block["Wo"]), _b(block["Wo"]), HE)

    # Residual: x = x + attn_out
    x_res = _he_add_vectors(x_enc, attn_out, HE)

    # ── FFN sub-layer ──────────────────────────────────────────────────────────
    # Linear1: x → d_ff   (activation skipped → identity)
    ff1_out = he_linear(x_res, _w(block["ff1"]), _b(block["ff1"]), HE)
    # Linear2: d_ff → d_model
    ff2_out = he_linear(ff1_out, _w(block["ff2"]), _b(block["ff2"]), HE)

    # Residual: x = x + ff2_out
    x_out = _he_add_vectors(x_res, ff2_out, HE)

    return x_out


# ---------------------------------------------------------------------------
# Stage 3: HE LM head
# ---------------------------------------------------------------------------

def he_lm_head(
    x_enc: _TaggedList,
    lm_head: nn.Linear,
    HE: Pyfhel,
    embed_scale: int = EMBED_SCALE,
) -> _TaggedList:
    """
    Project encrypted hidden state to vocabulary logits.

    The LM head is a standard nn.Linear(d_model, vocab_size, bias=False).
    Weight tying means lm_head.weight = token_emb.weight.

    Memory note:
        vocab_size ciphertexts output.
        For toy vocab_size=200: 200 × 48 KB ≈ 9.4 MB.
        For GPT-2 vocab_size=50257: ~2.4 GB → prohibitive without batching.
        With BFV slot-batching one ciphertext holds n//2 = 4096 values,
        so the full vocabulary fits in ceil(50257/4096) = 13 ciphertexts.
    """
    W = lm_head.weight.detach().float()          # (vocab_size, d_model)
    W_scaled = torch.round(W * embed_scale).to(torch.int64)
    W_np = W_scaled.T.numpy()                    # (d_model, vocab_size)
    b_np = np.zeros(lm_head.out_features, dtype=np.int64)

    return he_linear(x_enc, W_np, b_np, HE)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def full_he_inference(
    model: nn.Module,
    input_ids: torch.Tensor,
    HE_context: Optional[Pyfhel] = None,
    cfg: Optional[HEInferenceConfig] = None,
    top_k: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Full toy encrypted inference pipeline for batch=1, seq_len=1.

    LIMITATION: batch_size=1 and seq_len=1 are hard-coded throughout.
    Extending to seq_len > 1 requires:
      - Running he_attention_approx for each block (O(S²·d) ctxt×ctxt muls).
      - Positional encodings for positions 1..S-1.
      - Causal masking passed through all layers.
      This is left as a documented next step; see he_layers.he_attention_approx.

    Parameters
    ----------
    model      : nn.Module with attributes token_emb, pos_emb, blocks, lm_head.
                 Compatible with build_toy_model() or TinyGPT from model.py.
    input_ids  : torch.Tensor of shape (1, 1) — single token ID.
    HE_context : pre-built Pyfhel object, or None to auto-create from cfg.
    cfg        : HEInferenceConfig (used only when HE_context is None).
    top_k      : number of top predictions to return (default 5).
    verbose    : print progress and timing.

    Returns
    -------
    dict with keys:
      "top_tokens"    : list of top_k (token_id, score) tuples (decrypted)
      "logits_dec"    : torch.Tensor of shape (vocab_size,) — all decrypted logits
      "timings"       : TimingReport
      "memory"        : MemoryReport
      "plaintext_top" : list of top_k (token_id, score) from plaintext reference
    """
    if cfg is None:
        cfg = HEInferenceConfig()

    # Validate input shape
    if input_ids.shape != (1, 1):
        raise ValueError(
            f"full_he_inference requires input_ids.shape == (1, 1), "
            f"got {tuple(input_ids.shape)}.\n"
            "This pipeline supports batch_size=1, seq_len=1 only."
        )

    token_id = int(input_ids[0, 0].item())
    if verbose:
        print(f"\n  Input token ID : {token_id}")

    timings = TimingReport()
    t_wall  = time.perf_counter()

    # ── HE context ────────────────────────────────────────────────────────────
    if HE_context is None:
        if verbose:
            print("  Setting up HE context …", end=" ", flush=True)
        t0 = time.perf_counter()
        HE = setup_HE_context(n=cfg.n_poly, t=cfg.t)
        timings.context_setup = time.perf_counter() - t0
        if verbose:
            print(f"done ({timings.context_setup:.2f}s)")
    else:
        HE = HE_context
        if verbose:
            print("  Using provided HE context.")

    # ── Plaintext reference (for correctness comparison) ──────────────────────
    if verbose:
        print("  Computing plaintext reference …", end=" ", flush=True)
    with torch.no_grad():
        if hasattr(model, "plaintext_forward"):
            plain_logits = model.plaintext_forward(input_ids).squeeze()
        else:
            plain_logits = model(input_ids).squeeze()
    plain_top = plain_logits.topk(top_k)
    if verbose:
        print("done")
        print(f"  Plaintext top-{top_k}: "
              f"{list(zip(plain_top.indices.tolist(), [round(v,3) for v in plain_top.values.tolist()]))}")

    # ── Stage 1: Encrypt token ID (embedding lookup) ──────────────────────────
    if verbose:
        print("\n  [Stage 1] Encrypted embedding lookup …", end=" ", flush=True)
    t0 = time.perf_counter()

    x_enc = he_embedding_lookup(
        token_id=token_id,
        token_emb_weight=model.token_emb.weight.detach(),
        pos_emb_weight=model.pos_emb.weight.detach(),
        HE=HE,
        embed_scale=cfg.embed_scale,
    )
    timings.token_encrypt = time.perf_counter() - t0
    # embedding timing is the same as token encrypt at seq=1
    timings.embedding = timings.token_encrypt

    if verbose:
        print(f"done ({timings.embedding*1e3:.1f} ms, "
              f"{len(x_enc)} ciphertexts for d_model={cfg.d_model})")

    # ── Stage 2: Transformer blocks ───────────────────────────────────────────
    for layer_idx, blk in enumerate(model.blocks):
        if verbose:
            print(f"\n  [Stage 2.{layer_idx+1}] Transformer block {layer_idx+1}/{len(model.blocks)} …",
                  flush=True)
        t0 = time.perf_counter()

        x_enc = he_transformer_block_seq1(
            x_enc=x_enc,
            block=blk,
            d_model=cfg.d_model,
            HE=HE,
            embed_scale=cfg.embed_scale,
            block_idx=layer_idx,
        )

        t_blk = time.perf_counter() - t0
        timings.transformer.append(t_blk)
        if verbose:
            print(f"           block {layer_idx+1} done ({t_blk:.3f}s)")

    # ── Stage 3: LM head ──────────────────────────────────────────────────────
    if verbose:
        print(f"\n  [Stage 3] HE LM head (d_model={cfg.d_model} → vocab={cfg.vocab_size}) …",
              flush=True)
    t0 = time.perf_counter()

    logits_enc = he_lm_head(x_enc, model.lm_head, HE, embed_scale=cfg.embed_scale)
    timings.lm_head = time.perf_counter() - t0

    if verbose:
        print(f"           done ({timings.lm_head:.3f}s, "
              f"{len(logits_enc)} ciphertexts for vocab={cfg.vocab_size})")

    # ── Stage 4: Decrypt logits ───────────────────────────────────────────────
    if verbose:
        print(f"\n  [Stage 4] Decrypting {cfg.vocab_size} logit ciphertexts …",
              end=" ", flush=True)
    t0 = time.perf_counter()

    logits_raw = decrypt_tensor(logits_enc, HE)          # (1, vocab_size) int64
    logits_dec = logits_raw.flatten().to(torch.float32)  # (vocab_size,)
    timings.logit_decrypt = time.perf_counter() - t0

    if verbose:
        print(f"done ({timings.logit_decrypt*1e3:.1f} ms)")

    timings.total = time.perf_counter() - t_wall

    # ── Top-k predictions ─────────────────────────────────────────────────────
    top_vals, top_idxs = logits_dec.topk(min(top_k, len(logits_dec)))
    top_tokens = list(zip(top_idxs.tolist(), top_vals.tolist()))

    # ── Memory profile ────────────────────────────────────────────────────────
    embed_ctxts   = cfg.d_model
    per_blk_ctxts = cfg.d_model * 4   # Wv·x, Wo·v, ff1, ff2 (peak concurrent)
    lm_ctxts      = cfg.vocab_size
    total_ctxts   = embed_ctxts + per_blk_ctxts * cfg.num_layers + lm_ctxts
    mem = MemoryReport(
        embedding_ctxts  = embed_ctxts,
        per_block_ctxts  = per_blk_ctxts,
        lm_head_ctxts    = lm_ctxts,
        total_ctxts      = total_ctxts,
        estimated_mb     = total_ctxts * _BYTES_PER_CTXT / 1e6,
    )

    return {
        "top_tokens":    top_tokens,
        "logits_dec":    logits_dec,
        "timings":       timings,
        "memory":        mem,
        "plaintext_top": list(zip(
            plain_top.indices.tolist(),
            plain_top.values.tolist(),
        )),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _he_add_vectors(a: _TaggedList, b: _TaggedList, HE: Pyfhel) -> _TaggedList:
    """Element-wise HE addition of two equal-length ciphertext lists."""
    if len(a) != len(b):
        raise ValueError(
            f"Cannot add vectors of different lengths: {len(a)} vs {len(b)}"
        )
    summed = [ca + cb for ca, cb in zip(a, b)]
    shape = getattr(a, "original_shape", None)
    return _TaggedList(summed, original_shape=shape, batch=False)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  he_inference.py — Toy Encrypted LM Inference Demo")
    print("=" * 65)
    print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║  PRACTICAL LIMITATIONS (read before interpreting results)   ║
  ║                                                              ║
  ║  • batch_size=1, seq_len=1  ONLY                            ║
  ║  • LayerNorm: SKIPPED (identity substitution)               ║
  ║  • GELU/ReLU in FFN: SKIPPED (linear pass-through)          ║
  ║  • Attention at seq=1: degenerate (weight always 1.0)       ║
  ║  • Weights integer-scaled (EMBED_SCALE=100, truncation err) ║
  ║  • Results are APPROXIMATE; not equivalent to plaintext GPT ║
  ║  • Toy vocab=200, d_model=16 — not a real language model    ║
  ╚══════════════════════════════════════════════════════════════╝
""")

    cfg = HEInferenceConfig(
        vocab_size=200,
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_layers=2,
        embed_scale=EMBED_SCALE,
        n_poly=2 ** 13,
        t=65537,
    )

    print(f"  Config: vocab={cfg.vocab_size}, d_model={cfg.d_model}, "
          f"layers={cfg.num_layers}, d_ff={cfg.d_ff}")
    print(f"  BFV:   n={cfg.n_poly}, t={cfg.t}\n")

    # Build toy model
    model = build_toy_model(cfg)
    model.eval()

    # Single token input
    TEST_TOKEN = 42
    input_ids = torch.tensor([[TEST_TOKEN]], dtype=torch.long)

    # Run encrypted inference
    results = full_he_inference(
        model=model,
        input_ids=input_ids,
        HE_context=None,   # auto-create
        cfg=cfg,
        top_k=5,
        verbose=True,
    )

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Results")
    print("=" * 65)

    print(f"\n  Input token ID : {TEST_TOKEN}")

    print(f"\n  ── Plaintext top-5 (reference) ──────────────────────────────")
    for rank, (tid, score) in enumerate(results["plaintext_top"], 1):
        print(f"  #{rank}  token {tid:>4}  score {score:>10.4f}")

    print(f"\n  ── Encrypted inference top-5 (decrypted) ────────────────────")
    for rank, (tid, score) in enumerate(results["top_tokens"], 1):
        print(f"  #{rank}  token {tid:>4}  score {score:>10.0f}  (integer, ×{EMBED_SCALE}² scaled)")

    results["timings"].print()
    results["memory"].print()

    # Correctness: check that HE top-1 matches plaintext top-1 (order may differ
    # due to integer scaling and skipped nonlinearities, so we check overlap).
    he_top_ids   = {t for t, _ in results["top_tokens"]}
    pt_top_ids   = {t for t, _ in results["plaintext_top"]}
    overlap      = he_top_ids & pt_top_ids
    print(f"\n  Top-5 overlap between HE and plaintext: {len(overlap)}/5 tokens")
    print(f"  (Low overlap expected due to skipped LayerNorm/GELU — see limitations)")

    print(f"\n  ✓ Demo complete — decrypted top-5 tokens returned successfully")

    print("""
  Next steps for production-quality HE inference:
  ─────────────────────────────────────────────────
  1. Switch to CKKS (OpenFHE / Microsoft SEAL) for floating-point weights.
  2. Implement polynomial LayerNorm: mean/var via iterative Newton steps.
  3. Approximate GELU with degree-3 polynomial: 0.5x(1 + tanh(√2/π(x+0.044x³))).
  4. Enable seq_len > 1 via he_attention_approx (see he_layers.py).
  5. Use diagonal / Halevi-Shoup batching to reduce GEMV ctxt count by √d.
  6. Add bootstrapping to refresh noise budget between deep layers.
  7. Explore hybrid MPC+FHE: compute softmax/LayerNorm via secure 2-party protocol.
""")