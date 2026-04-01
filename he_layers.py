"""
he_layers.py
============
Homomorphic linear (fully-connected) layer using Pyfhel BFV scheme.

Overview
--------
Implements  y = x @ W + b  entirely in the encrypted domain, where:

    x  — encrypted input vector of shape (1, in_features)
    W  — plaintext weight matrix      of shape (in_features, out_features)
    b  — plaintext bias vector         of shape (out_features,)
    y  — encrypted output vector      of shape (1, out_features)

Because BFV supports plaintext × ciphertext multiplication (much cheaper than
ciphertext × ciphertext), we keep W and b in the clear and only encrypt x.
This is the standard "client encrypts activations, server evaluates model"
paradigm for privacy-preserving inference.

Computational cost
------------------
For an (in_features → out_features) linear layer:

  Operations per output neuron j:
    • in_features  ctxt × ptxt multiplications   (x_i * W[i,j])
    • in_features-1 ctxt + ctxt additions         (sum the products)
    • 1 ctxt + ptxt addition                      (+ bias_j)

  Total:
    • out_features * in_features  multiply-and-add pairs
    • Complexity: O(in_features * out_features)  — same as plaintext GEMV

Noise budget
-----------
BFV addition consumes negligible noise; plaintext multiplication consumes a
moderate amount (less than ciphertext multiplication).  For this single-layer
implementation:

  Budget consumed ≈  1 × (ptxt-mul cost)  per output neuron
                   + log2(in_features) × (add cost)  ← negligible

A single layer is well within the budget for n=2**13 or n=2**14.

Batching limits
---------------
We use element-wise (non-batched) encryption here for clarity.  Each scalar
x_i is its own ciphertext.  Batched BFV could pack the entire input vector
into ONE ciphertext using slot rotations (the "diagonal method"), but that
requires rotation keys and a more complex kernel.  The element-wise approach
is simpler and sufficient for small in_features (≤ ~256).

For large in_features, consider:
  • BFV diagonal / Halevi-Shoup algorithm (O(sqrt(n)) rotations)
  • CKKS scheme (approximate arithmetic, supports floating-point weights)
"""

import time
from typing import Union

import numpy as np
import torch

try:
    from Pyfhel import Pyfhel, PyCtxt
except ImportError as _err:  # pragma: no cover
    raise ImportError(
        "Pyfhel is required.  Install with:  pip install Pyfhel"
    ) from _err

from he_utils import setup_HE_context, encrypt_tensor, decrypt_tensor, _TaggedList


# ---------------------------------------------------------------------------
# Core: homomorphic linear layer
# ---------------------------------------------------------------------------

def he_linear(
    encrypted_input: "_TaggedList",
    plain_weights: Union[np.ndarray, torch.Tensor],
    plain_bias: Union[np.ndarray, torch.Tensor],
    HE: Pyfhel,
) -> "_TaggedList":
    """
    Homomorphic matrix-vector multiplication: y = x @ W + b.

    This function evaluates a single fully-connected layer in the encrypted
    domain.  Weights and bias remain in plaintext; only the input is encrypted.

    Parameters
    ----------
    encrypted_input : _TaggedList of PyCtxt
        Element-wise encrypted input vector produced by
        ``encrypt_tensor(x, HE, batch=False)`` where x has shape (1, in_features)
        or (in_features,).  Must contain exactly ``in_features`` ciphertexts.

    plain_weights : np.ndarray or torch.Tensor, shape (in_features, out_features)
        Weight matrix in plaintext.  Values must fit in [-(t-1)/2, (t-1)/2]
        after rounding to integers (BFV operates modulo t).
        For floating-point weights, multiply by a fixed scale before passing
        and divide the decrypted output by the same scale.

    plain_bias : np.ndarray or torch.Tensor, shape (out_features,)
        Bias vector in plaintext.  Same integer constraint applies.

    HE : Pyfhel
        Initialised Pyfhel object (from setup_HE_context).

    Returns
    -------
    _TaggedList of PyCtxt
        Encrypted output vector of length out_features.
        Decrypt with ``decrypt_tensor(result, HE)`` to recover a
        (1, out_features) int64 tensor.

    Raises
    ------
    TypeError  : wrong types for any argument.
    ValueError : shape mismatches or unsupported batch size.

    Notes
    -----
    Algorithm (column-wise dot product):
        For each output neuron j in [0, out_features):
            acc_j  = encrypted_input[0] * W[0, j]
            acc_j += encrypted_input[1] * W[1, j]
            ...
            acc_j += encrypted_input[in_features-1] * W[in_features-1, j]
            acc_j += encode(bias[j])        ← plaintext addition, free noise-wise

    Ciphertext × plaintext multiply (HE.multiply_plain) is used throughout;
    this is ~10× cheaper (in noise and time) than ciphertext × ciphertext.
    """
    # ── Type / shape validation ───────────────────────────────────────────────
    if not isinstance(HE, Pyfhel):
        raise TypeError(f"HE must be a Pyfhel instance, got {type(HE)}")
    if not isinstance(encrypted_input, (list, _TaggedList)):
        raise TypeError("encrypted_input must be a list of PyCtxt objects")

    # Normalise weights and bias to numpy int64
    W = _to_numpy_int(plain_weights)
    b = _to_numpy_int(plain_bias)

    if W.ndim != 2:
        raise ValueError(f"plain_weights must be 2-D (in_features, out_features), got shape {W.shape}")
    if b.ndim != 1:
        raise ValueError(f"plain_bias must be 1-D (out_features,), got shape {b.shape}")

    in_features, out_features = W.shape

    if b.shape[0] != out_features:
        raise ValueError(
            f"Bias length {b.shape[0]} does not match out_features={out_features}"
        )

    # Flatten encrypted_input and check length
    enc_list = list(encrypted_input)
    orig_shape = getattr(encrypted_input, "original_shape", None)

    # Accept shape (1, in_features) or (in_features,)
    if orig_shape is not None:
        n_enc = int(np.prod(orig_shape))
    else:
        n_enc = len(enc_list)

    if n_enc != in_features:
        raise ValueError(
            f"encrypted_input contains {n_enc} ciphertexts but "
            f"plain_weights has in_features={in_features}.  "
            f"Ensure you encrypted a vector of length in_features."
        )

    if len(enc_list) != in_features:
        raise ValueError(
            f"Internal mismatch: {len(enc_list)} ciphertexts vs "
            f"in_features={in_features}"
        )

    # ── Homomorphic GEMV ───────────────────────────────────────────────────────
    # Computational note:
    #   • _encode_scalar encodes a Python int into a BFV plaintext polynomial.
    #   • HE.multiply_plain(ctxt, ptxt) computes ctxt * ptxt in-place (modifies
    #     a copy; Pyfhel returns a new object when using the * operator).
    #   • We accumulate using += (homomorphic addition in-place) for efficiency.
    #   • Adding a plaintext bias at the end is essentially free (no noise cost).

    output_ciphertexts: list[PyCtxt] = []

    for j in range(out_features):
        # Weighted sum: acc_j = sum_i( x_i * W[i, j] )
        acc = None
        for i in range(in_features):
            w_ij = int(W[i, j])

            if w_ij == 0:
                # Multiplying by zero wastes a ciphertext; just skip.
                # The zero contribution is implicitly handled by the
                # initialisation of acc below.
                continue

            # Encode scalar weight as a plaintext constant
            ptxt_w = HE.encode([w_ij])

            # Ciphertext × plaintext  (returns NEW ciphertext, src unchanged)
            term = enc_list[i] * ptxt_w     # multiply_plain under the hood

            if acc is None:
                acc = term
            else:
                acc += term                 # homomorphic addition (in-place)

        if acc is None:
            # All weights for this neuron were zero → output is just bias
            acc = HE.encrypt(HE.encode([0]))

        # Add plaintext bias  (ctxt + ptxt, negligible noise cost)
        bias_j = int(b[j])
        if bias_j != 0:
            ptxt_b = HE.encode([bias_j])
            acc += ptxt_b

        output_ciphertexts.append(acc)

    return _TaggedList(
        output_ciphertexts,
        original_shape=(1, out_features),
        batch=False,
    )


# ---------------------------------------------------------------------------
# Timing comparison helper
# ---------------------------------------------------------------------------

def _plain_matmul(
    x: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Standard PyTorch matmul: y = x @ W + b."""
    return torch.matmul(x.float(), W.float()) + b.float()


def run_timing_comparison(
    in_features: int = 8,
    out_features: int = 4,
    n_poly: int = 2 ** 13,
    t: int = 65537,
    n_repeats_plain: int = 1000,
) -> dict:
    """
    Benchmark he_linear vs torch.matmul on the same (1, in_features) input.

    Parameters
    ----------
    in_features, out_features : int
        Layer dimensions.
    n_poly : int
        BFV polynomial modulus degree for this benchmark.
    t : int
        BFV plaintext modulus.
    n_repeats_plain : int
        How many times to repeat the plaintext matmul for a stable average.

    Returns
    -------
    dict with keys: t_he, t_plain, ratio, he_result, plain_result, max_error
    """
    print(f"\n{'─'*60}")
    print(f"  Layer: ({in_features},) → ({out_features},)")
    print(f"  BFV params: n={n_poly}, t={t}")
    print(f"{'─'*60}")

    # ── Setup ─────────────────────────────────────────────────────────────────
    print("  Setting up HE context …", end=" ", flush=True)
    t0 = time.perf_counter()
    HE = setup_HE_context(n=n_poly, t=t)
    print(f"done ({time.perf_counter()-t0:.2f}s)")

    # ── Generate random integer tensors ───────────────────────────────────────
    # Keep values small to avoid overflow modulo t and to stay in a range
    # where the integer ≈ float approximation is reasonable.
    torch.manual_seed(42)
    x_pt = torch.randint(1, 10, (1, in_features), dtype=torch.int64)
    W_pt = torch.randint(1, 10, (in_features, out_features), dtype=torch.int64)
    b_pt = torch.randint(0, 5,  (out_features,),             dtype=torch.int64)

    # ── Plaintext baseline ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    for _ in range(n_repeats_plain):
        plain_out = _plain_matmul(x_pt, W_pt, b_pt)
    t_plain_total = time.perf_counter() - t0
    t_plain = t_plain_total / n_repeats_plain

    plain_out_int = plain_out.to(torch.int64)
    print(f"  Plaintext matmul (avg over {n_repeats_plain} runs): {t_plain*1e6:.2f} µs")

    # ── Homomorphic linear ─────────────────────────────────────────────────────
    print("  Encrypting input …", end=" ", flush=True)
    t0 = time.perf_counter()
    enc_x = encrypt_tensor(x_pt, HE, batch=False)
    t_enc = time.perf_counter() - t0
    print(f"done ({t_enc*1e3:.1f} ms, {in_features} ciphertexts)")

    print("  Running he_linear …", end=" ", flush=True)
    t0 = time.perf_counter()
    enc_y = he_linear(enc_x, W_pt.numpy(), b_pt.numpy(), HE)
    t_he_compute = time.perf_counter() - t0
    print(f"done ({t_he_compute*1e3:.1f} ms)")

    print("  Decrypting output …", end=" ", flush=True)
    t0 = time.perf_counter()
    he_out = decrypt_tensor(enc_y, HE)
    t_dec = time.perf_counter() - t0
    print(f"done ({t_dec*1e3:.1f} ms)")

    t_he_total = t_enc + t_he_compute + t_dec

    # ── Correctness check ─────────────────────────────────────────────────────
    he_out_flat    = he_out.flatten().to(torch.int64)
    plain_out_flat = plain_out_int.flatten()
    max_err = (he_out_flat - plain_out_flat).abs().max().item()

    print(f"\n  ── Results ──")
    print(f"  Plaintext output  : {plain_out_flat.tolist()}")
    print(f"  Decrypted HE out  : {he_out_flat.tolist()}")
    print(f"  Max absolute error: {max_err}  (0 = exact for BFV integers)")

    ratio = t_he_total / t_plain if t_plain > 0 else float("inf")
    print(f"\n  ── Timing ──")
    print(f"  Plaintext matmul  : {t_plain*1e6:>10.2f} µs  (avg)")
    print(f"  HE encrypt        : {t_enc*1e3:>10.2f} ms")
    print(f"  HE compute        : {t_he_compute*1e3:>10.2f} ms")
    print(f"  HE decrypt        : {t_dec*1e3:>10.2f} ms")
    print(f"  HE total          : {t_he_total*1e3:>10.2f} ms")
    print(f"  Slowdown ratio    : {ratio:>10.1f}×  (HE / plaintext)")
    print(f"\n  Note: The HE slowdown ({ratio:.0f}×) is expected and normal.")
    print(f"  FHE trades speed for data privacy — the server performs the")
    print(f"  matrix multiply without ever seeing the plaintext input x.")

    return {
        "t_he":       t_he_total,
        "t_plain":    t_plain,
        "ratio":      ratio,
        "he_result":  he_out_flat,
        "plain_result": plain_out_flat,
        "max_error":  max_err,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy_int(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert torch.Tensor or numpy array to int64 numpy array."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy().astype(np.int64)
    if isinstance(arr, np.ndarray):
        return arr.astype(np.int64)
    raise TypeError(
        f"Expected np.ndarray or torch.Tensor, got {type(arr)}"
    )


# ---------------------------------------------------------------------------
# Approximate Homomorphic Attention
# ---------------------------------------------------------------------------
#
# Softmax approximation strategy — "plaintext-side top-k re-weighting"
# ---------------------------------------------------------------------
# True softmax requires exp(), which cannot be computed exactly in BFV
# (a purely integer scheme).  We use a two-phase approximation:
#
#   Phase 1 — compute raw dot-product scores Q @ K^T homomorphically.
#             This is the only step performed in the encrypted domain.
#             The server decrypts ONLY the score matrix (not Q, K, or V
#             individually) — a controlled leakage chosen for tractability.
#
#   Phase 2 — the decrypted integer scores are softmax-approximated in
#             plaintext via integer-scaled fixed-point softmax:
#               a) subtract max for numerical stability,
#               b) apply floor(exp(score - max) * SCALE) to get integers,
#               c) divide by sum to obtain attention weights in [0, 1],
#               d) re-scale to integers: w_i = round(w_i * WEIGHT_SCALE).
#
#   Phase 3 — re-encrypt the integer attention weights and multiply with
#             encrypted V to produce the encrypted output.
#
# Why this approach?
#   • Alternative 1: polynomial approximation of softmax inside HE — requires
#     CKKS (not BFV), many multiplication levels, and a carefully tuned
#     low-degree polynomial.  Error is hard to bound.
#   • Alternative 2: share only the score indices (argmax / top-k) — leaks
#     attention pattern but not values; useful when pattern privacy < value
#     privacy.
#   • Chosen approach (phase 2 in plaintext): cleanest for BFV since BFV
#     cannot represent floats.  The privacy model is: the server learns the
#     attention *score magnitudes* (integers, not the raw Q/K values), which
#     is a documented, bounded leakage acceptable in many threat models
#     (e.g., outsourced inference where the model weights are the secret).
#
# Causal masking strategy
# -----------------------
# Standard causal masking sets upper-triangle scores to -inf before softmax.
# In our scheme masking is applied in Phase 2 (plaintext softmax):
#   • After decrypting the integer score matrix, set masked positions to
#     a large negative integer (−BIG) before the exp step so they map to
#     ≈ 0 weight after normalisation.
#   • This is equivalent to the standard additive -inf mask and requires
#     zero extra HE operations.
#   • The causal_mask parameter is a boolean (seq_len, seq_len) array where
#     True = keep, False = mask out (or None for full attention).
#
# Computational cost  O(seq_len^2 * d_head)
# ------------------------------------------
#   Score matrix:  seq_len^2 dot products, each costing d_head ctxt*ctxt muls
#                  and d_head-1 additions  → O(S^2 * d) ctxt operations.
#   Weighted sum:  seq_len output vectors, each costing seq_len ctxt*ptxt muls
#                  → O(S^2) cheap operations.
#   Total multiplications: O(S^2 * d)  ciphertext × ciphertext  (expensive)
#                        + O(S^2)      ciphertext × plaintext    (cheap)
#
# Noise budget note
# -----------------
# Each ctxt × ctxt multiply consumes ~half the remaining budget.
# For d_head=4, S=4: we perform 4 ctxt×ctxt muls per score entry, depth=1.
# With n=2**14 the budget is ~438 bits; depth-1 consumes ~219 bits → safe.
# For S=8, d=8: same depth (we sum products, not chain them), so still safe.
# Relinearisation after each ctxt×ctxt mul is mandatory to keep size at 2.

# Fixed-point scale for re-encoding attention weights as integers.
# After softmax the float weights ∈ [0,1]; multiplying by WEIGHT_SCALE
# and rounding gives integers in [0, WEIGHT_SCALE].
# Higher WEIGHT_SCALE → better precision but larger ciphertext values
# (must stay < t/2 = 32768 after the final multiply with V values).
_WEIGHT_SCALE = 64   # 6-bit precision; safe: 64 * max_V_val << 32768


def he_attention_approx(
    Q_enc: "_TaggedList",
    K_enc: "_TaggedList",
    V_enc: "_TaggedList",
    HE: Pyfhel,
    causal_mask: "np.ndarray | None" = None,
) -> tuple["_TaggedList", dict]:
    """
    Approximate single-head attention in the encrypted domain (BFV).

    Implements  Attn(Q,K,V) ≈ softmax_approx(Q @ K^T / sqrt(d)) @ V
    using the plaintext-side softmax approximation described in the module
    docstring above.

    Parameters
    ----------
    Q_enc, K_enc, V_enc : _TaggedList of PyCtxt
        Encrypted query, key, value matrices, each of shape (seq_len, d_head).
        Produced by calling encrypt_tensor on an integer-scaled tensor.
        Each must contain seq_len * d_head ciphertexts (element-wise, row-major).

    HE : Pyfhel
        Initialised Pyfhel context (setup_HE_context).

    causal_mask : np.ndarray of bool, shape (seq_len, seq_len), or None
        causal_mask[i, j] = True  → position j is visible to query i (keep).
        causal_mask[i, j] = False → masked out (set score to -∞ equivalent).
        Pass None for full (bidirectional) attention.
        If omitted, a standard lower-triangular causal mask is NOT applied
        automatically; pass the result of _make_causal_mask(seq_len) explicitly.

    Returns
    -------
    output_enc : _TaggedList of PyCtxt
        Encrypted attention output, shape (seq_len, d_head).
        Decrypt with decrypt_tensor(output_enc, HE) and reshape to
        (seq_len, d_head).

    timings : dict
        Per-step wall-clock times:
          "t_qkt"        — HE Q @ K^T  (score computation)
          "t_decrypt_scores" — score decryption
          "t_softmax"    — plaintext softmax approximation
          "t_reencrypt"  — re-encrypting attention weights
          "t_weighted_v" — HE weighted sum of V
          "t_total"      — end-to-end

    Raises
    ------
    ValueError : shape mismatches, seq_len > 8, or HE not initialised.
    TypeError  : wrong argument types.

    Numerical / semantic limitations
    ---------------------------------
    1. Integer-only arithmetic: Q, K, V must be pre-scaled integers.
       Floating-point inputs lose fractional parts.  Scale by e.g. 100
       before encrypting; divide outputs by scale² to recover float values.

    2. Controlled score leakage: the server decrypts the score matrix
       (S×S integers) to compute softmax.  Q, K, V values themselves are
       never exposed, but score magnitudes are.

    3. Score overflow: Q @ K^T values grow as d_head * max_Q * max_K.
       With Q,K in [1,9] and d_head=4: max score = 4*9*9 = 324 << 32768 ✓.
       For larger d_head or value ranges, reduce the input scale.

    4. Weight precision: softmax weights are quantised to _WEIGHT_SCALE=64
       integer levels.  This introduces ≈ 1/64 ≈ 1.6% relative rounding
       error per position.  Increase _WEIGHT_SCALE for higher precision
       (but verify max_score * _WEIGHT_SCALE < t/2 to avoid overflow).

    5. seq_len ≤ 8 enforced: beyond this, the O(S²·d) ciphertext
       multiplications become prohibitively slow on CPU without batching.
    """
    # ── Validation ────────────────────────────────────────────────────────────
    if not isinstance(HE, Pyfhel):
        raise TypeError(f"HE must be a Pyfhel instance, got {type(HE)}")
    for name, enc in [("Q_enc", Q_enc), ("K_enc", K_enc), ("V_enc", V_enc)]:
        if not isinstance(enc, (list, _TaggedList)):
            raise TypeError(f"{name} must be a _TaggedList of PyCtxt")

    # Infer seq_len and d_head from the tagged shape
    def _parse_shape(enc, name):
        shape = getattr(enc, "original_shape", None)
        if shape is None or len(shape) != 2:
            raise ValueError(
                f"{name}.original_shape must be (seq_len, d_head), got {shape}.\n"
                f"Encrypt with encrypt_tensor(tensor_2d, HE, batch=False)."
            )
        return shape  # (seq_len, d_head)

    q_shape = _parse_shape(Q_enc, "Q_enc")
    k_shape = _parse_shape(K_enc, "K_enc")
    v_shape = _parse_shape(V_enc, "V_enc")

    seq_len, d_head = q_shape
    if k_shape != q_shape:
        raise ValueError(f"K shape {k_shape} must match Q shape {q_shape}")
    if v_shape[0] != seq_len:
        raise ValueError(f"V seq_len {v_shape[0]} must match Q seq_len {seq_len}")
    if seq_len > 8:
        raise ValueError(
            f"seq_len={seq_len} exceeds the supported maximum of 8.  "
            "Larger sequences require batching/CKKS and are out of scope."
        )

    if causal_mask is not None:
        causal_mask = np.asarray(causal_mask, dtype=bool)
        if causal_mask.shape != (seq_len, seq_len):
            raise ValueError(
                f"causal_mask shape {causal_mask.shape} must be ({seq_len},{seq_len})"
            )

    # Convert flat ciphertext lists to 2-D indexing: enc[i][k] = ctxt at (i,k)
    def _to_grid(enc, shape):
        S, D = shape
        flat = list(enc)
        return [[flat[i * D + k] for k in range(D)] for i in range(S)]

    Q_grid = _to_grid(Q_enc, q_shape)
    K_grid = _to_grid(K_enc, k_shape)
    V_grid = _to_grid(V_enc, v_shape)
    d_v    = v_shape[1]

    t_wall_start = time.perf_counter()

    # ── Phase 1: Homomorphic Q @ K^T  ────────────────────────────────────────
    # scores[i][j] = encrypted dot product of Q[i,:] and K[j,:]
    # Cost: seq_len^2 dot products × d_head ctxt×ctxt multiplications each.
    # Each ctxt×ctxt mul is relinearised immediately to keep ciphertext size=2.
    print(f"    [Phase 1] Computing encrypted Q @ K^T "
          f"({seq_len}×{seq_len} scores, d_head={d_head}) …", flush=True)
    t0 = time.perf_counter()

    scores_enc: list[list] = []
    for i in range(seq_len):
        row = []
        for j in range(seq_len):
            # dot(Q[i], K[j]) = sum_k Q[i,k] * K[j,k]
            acc = None
            for k in range(d_head):
                prod = Q_grid[i][k] * K_grid[j][k]   # ctxt × ctxt
                HE.relinearize(prod)                   # mandatory after ctxt×ctxt
                if acc is None:
                    acc = prod
                else:
                    acc += prod
            row.append(acc)
        scores_enc.append(row)

    t_qkt = time.perf_counter() - t0
    print(f"           done in {t_qkt:.3f}s  "
          f"({seq_len**2 * d_head} ctxt×ctxt muls + {seq_len**2*(d_head-1)} adds)")

    # ── Phase 2a: Decrypt score matrix (controlled leakage) ───────────────────
    # The server learns integer score values — not Q or K individually.
    print(f"    [Phase 2a] Decrypting score matrix ({seq_len}×{seq_len}) …",
          end=" ", flush=True)
    t0 = time.perf_counter()

    scores_plain = np.zeros((seq_len, seq_len), dtype=np.float64)
    for i in range(seq_len):
        for j in range(seq_len):
            val = HE.decryptInt(scores_enc[i][j])
            scores_plain[i, j] = float(val[0] if hasattr(val, "__len__") else val)

    t_decrypt_scores = time.perf_counter() - t0
    print(f"done in {t_decrypt_scores*1e3:.2f} ms")
    print(f"           Raw scores (before scaling):\n{scores_plain.astype(int)}")

    # ── Phase 2b: Plaintext softmax approximation ─────────────────────────────
    # Scale by 1/sqrt(d_head), apply causal mask, then integer-quantised softmax.
    print(f"    [Phase 2b] Plaintext softmax approximation …", end=" ", flush=True)
    t0 = time.perf_counter()

    scaled = scores_plain / np.sqrt(d_head)   # standard attention scaling

    # Apply causal mask: set masked positions to a large negative value
    # so exp(score) ≈ 0 after the subtraction of max.
    BIG_NEG = -1e9
    if causal_mask is not None:
        scaled[~causal_mask] = BIG_NEG

    # Numerically stable softmax: subtract row-max before exp
    shifted  = scaled - scaled.max(axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    row_sums = exp_vals.sum(axis=1, keepdims=True)
    attn_w   = exp_vals / row_sums                         # float weights ∈ [0,1]

    # Quantise to integers for BFV re-encryption
    attn_w_int = np.round(attn_w * _WEIGHT_SCALE).astype(np.int64)

    t_softmax = time.perf_counter() - t0
    print(f"done in {t_softmax*1e6:.1f} µs")
    print(f"           Attention weights (×{_WEIGHT_SCALE}):\n{attn_w_int}")

    # ── Phase 3a: Re-encrypt attention weights ────────────────────────────────
    # attn_w_int is (seq_len, seq_len) — rows are query positions, cols are keys.
    # We encrypt it so the weighted sum of V is also done in the HE domain.
    print(f"    [Phase 3a] Re-encrypting attention weights …", end=" ", flush=True)
    t0 = time.perf_counter()

    w_enc_tensor = torch.tensor(attn_w_int, dtype=torch.int64)
    w_enc = encrypt_tensor(w_enc_tensor, HE, batch=False)
    w_grid = [[list(w_enc)[i * seq_len + j] for j in range(seq_len)]
              for i in range(seq_len)]

    t_reencrypt = time.perf_counter() - t0
    print(f"done in {t_reencrypt*1e3:.2f} ms  ({seq_len**2} ciphertexts)")

    # ── Phase 3b: Homomorphic weighted sum of V ───────────────────────────────
    # output[i, k] = sum_j( w[i,j] * V[j,k] )
    # Cost: seq_len^2 × d_v  ctxt×ctxt multiplications.
    print(f"    [Phase 3b] Computing encrypted attention output "
          f"({seq_len}×{d_v}) …", flush=True)
    t0 = time.perf_counter()

    out_grid: list[list] = []
    for i in range(seq_len):
        row_out = []
        for k in range(d_v):
            acc = None
            for j in range(seq_len):
                prod = w_grid[i][j] * V_grid[j][k]   # ctxt × ctxt
                HE.relinearize(prod)
                if acc is None:
                    acc = prod
                else:
                    acc += prod
            row_out.append(acc)
        out_grid.append(row_out)

    t_weighted_v = time.perf_counter() - t0
    t_total = time.perf_counter() - t_wall_start
    print(f"           done in {t_weighted_v:.3f}s  "
          f"({seq_len**2 * d_v} ctxt×ctxt muls)")

    # Flatten output grid back to a _TaggedList
    flat_out = [out_grid[i][k] for i in range(seq_len) for k in range(d_v)]
    output_enc = _TaggedList(flat_out, original_shape=(seq_len, d_v), batch=False)

    timings = {
        "t_qkt":            t_qkt,
        "t_decrypt_scores": t_decrypt_scores,
        "t_softmax":        t_softmax,
        "t_reencrypt":      t_reencrypt,
        "t_weighted_v":     t_weighted_v,
        "t_total":          t_total,
    }
    return output_enc, timings


def _make_causal_mask(seq_len: int) -> np.ndarray:
    """Return lower-triangular bool mask: mask[i,j]=True iff j<=i."""
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


def _plain_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal_mask: "np.ndarray | None" = None,
) -> torch.Tensor:
    """Reference plaintext single-head attention (float)."""
    import math
    d = Q.shape[-1]
    scores = (Q.float() @ K.float().T) / math.sqrt(d)   # (S, S)
    if causal_mask is not None:
        mask_t = torch.tensor(causal_mask, dtype=torch.bool)
        scores = scores.masked_fill(~mask_t, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return attn @ V.float()                              # (S, d_v)


# ---------------------------------------------------------------------------
# Demo / acceptance test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  he_layers.py — Homomorphic Linear Layer Demo")
    print("=" * 60)

    # ── Small demo: 4 → 3 layer ───────────────────────────────────────────────
    results_small = run_timing_comparison(
        in_features=4,
        out_features=3,
        n_poly=2 ** 13,
        t=65537,
        n_repeats_plain=1000,
    )
    assert results_small["max_error"] == 0, (
        f"Correctness FAILED: max_error={results_small['max_error']}"
    )
    print("\n  ✓ Small layer (4→3) correctness check passed")

    # ── Medium demo: 8 → 4 layer ──────────────────────────────────────────────
    results_med = run_timing_comparison(
        in_features=8,
        out_features=4,
        n_poly=2 ** 13,
        t=65537,
        n_repeats_plain=1000,
    )
    assert results_med["max_error"] == 0, (
        f"Correctness FAILED: max_error={results_med['max_error']}"
    )
    print("\n  ✓ Medium layer (8→4) correctness check passed")

    # ── Homomorphic attention demo ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  he_attention_approx Demo")
    print("  seq_len=4, d_head=4, d_v=4, causal mask ON")
    print("=" * 60)

    torch.manual_seed(7)
    SEQ  = 4
    D    = 4       # d_head = d_v
    SCALE = 10     # integer scale for Q, K, V (divide outputs by SCALE^2)

    Q_pt = torch.randint(1, 6, (SEQ, D), dtype=torch.int64)
    K_pt = torch.randint(1, 6, (SEQ, D), dtype=torch.int64)
    V_pt = torch.randint(1, 6, (SEQ, D), dtype=torch.int64)

    # Plaintext reference (unscaled float)
    mask = _make_causal_mask(SEQ)
    plain_out = _plain_attention(Q_pt, K_pt, V_pt, causal_mask=mask)
    print(f"\n  Plaintext attention output (float):\n{plain_out.numpy().round(3)}")

    # HE context
    print("\n  Setting up HE context (n=2**13) …", end=" ", flush=True)
    t0 = time.perf_counter()
    HE = setup_HE_context(n=2 ** 13, t=65537)
    print(f"done ({time.perf_counter()-t0:.2f}s)")

    # Encrypt Q, K, V
    print("  Encrypting Q, K, V …", end=" ", flush=True)
    t0 = time.perf_counter()
    Q_enc = encrypt_tensor(Q_pt, HE, batch=False)
    K_enc = encrypt_tensor(K_pt, HE, batch=False)
    V_enc = encrypt_tensor(V_pt, HE, batch=False)
    print(f"done ({time.perf_counter()-t0:.2f}s, {3*SEQ*D} ciphertexts)")

    # Homomorphic attention
    print("\n  Running he_attention_approx …")
    out_enc, timings = he_attention_approx(
        Q_enc, K_enc, V_enc, HE, causal_mask=mask
    )

    # Decrypt output
    print("\n  Decrypting output …", end=" ", flush=True)
    t0 = time.perf_counter()
    he_out_raw = decrypt_tensor(out_enc, HE)       # (SEQ, D), integer
    t_dec_out  = time.perf_counter() - t0
    print(f"done ({t_dec_out*1e3:.2f} ms)")

    # The HE output = (Q·K^T / sqrt(d), quantised to _WEIGHT_SCALE) × V_int
    # To compare with float attention we scale plaintext V by WEIGHT_SCALE too
    scores = (Q_pt.float() @ K_pt.float().T) / (D ** 0.5)
    # Apply causal mask BEFORE softmax (same order as HE compute path)
    mask_t = torch.tensor(mask, dtype=torch.bool)
    scores_masked = scores.masked_fill(~mask_t, float("-inf"))
    plain_attn_w = torch.softmax(scores_masked, dim=-1)
    plain_attn_w_int = (plain_attn_w * _WEIGHT_SCALE).round().to(torch.int64)
    expected_int_out = plain_attn_w_int @ V_pt     # (SEQ, D)

    max_err = (he_out_raw - expected_int_out).abs().max().item()

    print(f"\n  ── Attention Results ──")
    print(f"  Decrypted HE output (integer, ×{_WEIGHT_SCALE} scaled):")
    print(f"  {he_out_raw.tolist()}")
    print(f"  Expected integer output:")
    print(f"  {expected_int_out.tolist()}")
    print(f"  Max absolute error : {max_err}  (0 = exact BFV arithmetic)")

    print(f"\n  ── Timing Breakdown ──")
    print(f"  Q @ K^T (HE)         : {timings['t_qkt']:.3f} s")
    print(f"  Score decryption     : {timings['t_decrypt_scores']*1e3:.2f} ms")
    print(f"  Softmax (plaintext)  : {timings['t_softmax']*1e6:.1f} µs")
    print(f"  Re-encrypt weights   : {timings['t_reencrypt']*1e3:.2f} ms")
    print(f"  Weighted V sum (HE)  : {timings['t_weighted_v']:.3f} s")
    print(f"  ─────────────────────────────────────────")
    print(f"  Total end-to-end     : {timings['t_total']:.3f} s")

    assert max_err == 0, f"Attention correctness FAILED: max_err={max_err}"
    print("\n  ✓ he_attention_approx correctness check passed")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  All demos complete ✓")
    print(f"  Linear slowdown 4→3  : {results_small['ratio']:.0f}×")
    print(f"  Linear slowdown 8→4  : {results_med['ratio']:.0f}×")
    print(f"  Attention total time : {timings['t_total']:.2f}s (seq=4, d=4)")
    print()
    print("  Numerical / semantic limitations:")
    print(f"  • Softmax quantised to {_WEIGHT_SCALE} levels (~{100/_WEIGHT_SCALE:.1f}% rounding)")
    print(f"  • Score matrix decrypted to server (controlled leakage)")
    print(f"  • Q/K/V must be pre-scaled integers; floats lose precision")
    print(f"  • seq_len ≤ 8 enforced; O(S²·d) ctxt×ctxt muls dominate cost")
    print(f"  • For production: use CKKS + poly-softmax + diagonal batching")
    print("=" * 60)