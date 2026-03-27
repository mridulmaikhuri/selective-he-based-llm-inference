"""
Causal Self-Attention
======================
Implements multi-head causal self-attention as used in decoder-only
transformers (GPT-style). Each token can only attend to itself and
tokens that came before it — future positions are masked out.

Reference:
    Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
    https://arxiv.org/abs/1706.03762

Usage:
    from attention import CausalSelfAttention
    import torch

    attn = CausalSelfAttention(d_model=128, num_heads=4, dropout=0.0)
    x    = torch.randn(2, 10, 128)           # (batch, seq_len, d_model)
    out, weights = attn(x)
    # out     → (2, 10, 128)
    # weights → (2, 4, 10, 10)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention layer.

    Projects the input into queries, keys, and values using independent
    linear layers, computes scaled dot-product attention with a causal mask
    so that position i cannot attend to any position j > i, optionally
    applies a caller-supplied padding mask, and projects the concatenated
    head outputs back to d_model.

    Args:
        d_model   (int):   Model / embedding dimension. Must be divisible
                           by num_heads. Default: 128.
        num_heads (int):   Number of parallel attention heads. Default: 4.
        dropout   (float): Dropout probability on attention weights.
                           Default: 0.1.

    Shape convention (used throughout this file):
        B  – batch size
        S  – sequence length
        D  – d_model
        H  – num_heads
        Dh – head dimension  (D // H)
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        # Validate arguments                                                   #
        # ------------------------------------------------------------------ #
        if d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {d_model}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads   # Dh
        self.scale     = math.sqrt(self.head_dim)  # 1/√Dh scaling factor

        # ------------------------------------------------------------------ #
        # Separate projection layers for Q, K, V (no bias — common in GPT)   #
        # Using separate layers (rather than one fused QKV projection) makes  #
        # the data-flow explicit and easier to inspect/debug.                 #
        # ------------------------------------------------------------------ #
        self.q_proj   = nn.Linear(d_model, d_model, bias=False)  # (D → D)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)  # (D → D)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)  # (D → D)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)  # (D → D)

        # Attention-weight dropout (applied before values are aggregated)
        self.attn_drop = nn.Dropout(p=dropout)

        # ------------------------------------------------------------------ #
        # Causal mask                                                          #
        # Stored as a buffer so it moves with the module (.to(device), etc.)  #
        # and is excluded from model parameters.                               #
        #                                                                      #
        # Shape: (1, 1, max_len, max_len)                                      #
        #   dim-0 (1) → broadcasts over batch                                  #
        #   dim-1 (1) → broadcasts over heads                                  #
        #                                                                      #
        # tril produces:                                                        #
        #   [[1, 0, 0, …],                                                     #
        #    [1, 1, 0, …],                                                     #
        #    [1, 1, 1, …],  …]                                                 #
        # position i may attend to positions 0 … i  (lower triangle).         #
        # ------------------------------------------------------------------ #
        max_len = 1024
        causal_mask = torch.tril(
            torch.ones(max_len, max_len, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(0)                 # (1, 1, max_len, max_len)
        self.register_buffer("causal_mask", causal_mask)

        # Weight initialisation (following GPT-2)
        self._init_weights()

    # ---------------------------------------------------------------------- #
    # Weight initialisation                                                    #
    # ---------------------------------------------------------------------- #

    def _init_weights(self) -> None:
        """Xavier-uniform init for projections; helps gradient flow early on."""
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(proj.weight)

    # ---------------------------------------------------------------------- #
    # Shape helper                                                             #
    # ---------------------------------------------------------------------- #

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, S, D) → (B, H, S, Dh) for multi-head attention.

        Steps:
          1. (B, S, D)      → (B, S, H, Dh)   via .view()
          2. (B, S, H, Dh)  → (B, H, S, Dh)   via .transpose(1, 2)

        The transpose puts the head dimension next to batch so that bmm /
        einsum can treat each (S, Dh) slice as an independent matrix.
        """
        B, S, D = x.shape
        # Step 1: carve D into H heads of size Dh
        x = x.view(B, S, self.num_heads, self.head_dim)   # (B, S, H, Dh)
        # Step 2: move head dim forward
        return x.transpose(1, 2).contiguous()             # (B, H, S, Dh)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of _split_heads: (B, H, S, Dh) → (B, S, D).

        Steps:
          1. (B, H, S, Dh) → (B, S, H, Dh)  via .transpose(1, 2)
          2. (B, S, H, Dh) → (B, S, D)       via .reshape()
        """
        B, H, S, Dh = x.shape
        return (
            x.transpose(1, 2)                    # (B, S, H, Dh)
             .contiguous()
             .reshape(B, S, H * Dh)              # (B, S, D)
        )

    # ---------------------------------------------------------------------- #
    # Forward pass                                                             #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute causal multi-head self-attention.

        Args:
            x (Tensor):
                Input of shape (B, S, D).
            attn_mask (BoolTensor | None):
                Optional padding mask of shape (B, S).
                True  → this token is REAL     (keep it).
                False → this token is PAD      (mask it out).
                If None, no padding mask is applied.

        Returns:
            out (Tensor):         shape (B, S, D)  — attended output.
            attn_weights (Tensor): shape (B, H, S, S) — softmax weights
                                   (useful for visualisation / debugging).
        """
        # ------------------------------------------------------------------ #
        # 0. Validate input                                                   #
        # ------------------------------------------------------------------ #
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3-D input (B, S, D), got shape {tuple(x.shape)}"
            )
        B, S, D = x.shape
        if D != self.d_model:
            raise ValueError(
                f"Input last dim ({D}) does not match d_model ({self.d_model})"
            )
        if S > self.causal_mask.shape[-1]:
            raise ValueError(
                f"seq_len ({S}) exceeds the pre-built causal mask length "
                f"({self.causal_mask.shape[-1]}). Increase max_len at init."
            )

        # ------------------------------------------------------------------ #
        # 1. Project input into Q, K, V                                       #
        #    Each projection: (B, S, D) → (B, S, D)                          #
        # ------------------------------------------------------------------ #
        Q = self.q_proj(x)   # (B, S, D)
        K = self.k_proj(x)   # (B, S, D)
        V = self.v_proj(x)   # (B, S, D)

        # ------------------------------------------------------------------ #
        # 2. Split into multiple heads                                         #
        #    (B, S, D) → (B, H, S, Dh)                                       #
        # ------------------------------------------------------------------ #
        Q = self._split_heads(Q)   # (B, H, S, Dh)
        K = self._split_heads(K)   # (B, H, S, Dh)
        V = self._split_heads(V)   # (B, H, S, Dh)

        # ------------------------------------------------------------------ #
        # 3. Scaled dot-product attention scores                              #
        #                                                                      #
        #    scores[b, h, i, j] = Q[b,h,i,:] · K[b,h,j,:] / √Dh            #
        #                                                                      #
        #    Using einsum "bhid,bhjd->bhij":                                  #
        #      b → batch, h → head, i → query pos, j → key pos, d → Dh      #
        #    Result shape: (B, H, S, S)                                       #
        # ------------------------------------------------------------------ #
        scores = torch.einsum("bhid,bhjd->bhij", Q, K) / self.scale
        #         (B, H, S, Dh) × (B, H, S, Dh) → (B, H, S, S)

        # ------------------------------------------------------------------ #
        # 4. Apply causal mask                                                 #
        #                                                                      #
        #    causal_mask[:, :, :S, :S] has shape (1, 1, S, S); True where    #
        #    attending is ALLOWED (lower triangle).                            #
        #                                                                      #
        #    Where the mask is False (upper triangle = future tokens),        #
        #    we fill with -inf so softmax drives those weights to 0.          #
        #                                                                      #
        #    Diagram for S=4 (✓ = allowed, ✗ = masked):                      #
        #      query\key  0   1   2   3                                       #
        #          0    [ ✓   ✗   ✗   ✗ ]                                    #
        #          1    [ ✓   ✓   ✗   ✗ ]                                    #
        #          2    [ ✓   ✓   ✓   ✗ ]                                    #
        #          3    [ ✓   ✓   ✓   ✓ ]                                    #
        # ------------------------------------------------------------------ #
        causal = self.causal_mask[:, :, :S, :S]   # (1, 1, S, S) bool
        scores = scores.masked_fill(~causal, float("-inf"))
        #   ~causal → True where future: fill those positions with -∞

        # ------------------------------------------------------------------ #
        # 5. Apply optional padding mask                                       #
        #                                                                      #
        #    attn_mask: (B, S) bool — False marks PAD tokens.                 #
        #                                                                      #
        #    We want position i to ignore key position j if j is a pad.      #
        #    Reshape to (B, 1, 1, S) so it broadcasts over (B, H, S, S):     #
        #      - dim-1 (1) broadcasts over heads                              #
        #      - dim-2 (1) broadcasts over query positions                    #
        #      - dim-3 (S) selects which key positions are pad                #
        # ------------------------------------------------------------------ #
        if attn_mask is not None:
            if attn_mask.shape != (B, S):
                raise ValueError(
                    f"attn_mask must have shape (B, S) = ({B}, {S}), "
                    f"got {tuple(attn_mask.shape)}"
                )
            # (B, S) → (B, 1, 1, S): broadcasts cleanly over all heads & queries
            pad_mask = attn_mask.unsqueeze(1).unsqueeze(2)   # (B, 1, 1, S)
            scores = scores.masked_fill(~pad_mask, float("-inf"))
            #   ~pad_mask → True where PAD: fill with -∞

        # ------------------------------------------------------------------ #
        # 6. Softmax → attention weights                                       #
        #                                                                      #
        #    Softmax over dim=-1 (key dimension).                             #
        #    Positions filled with -inf become 0 after softmax.               #
        #                                                                      #
        #    Edge-case: if every key position in a row is -inf (e.g. a fully  #
        #    padded query), softmax produces NaN.  We clamp those rows to 0.  #
        # ------------------------------------------------------------------ #
        attn_weights = F.softmax(scores, dim=-1)   # (B, H, S, S)

        # Replace any NaN rows (all-masked queries) with zeros
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Dropout on attention weights (randomly zero out some connections)
        attn_weights = self.attn_drop(attn_weights)   # (B, H, S, S)

        # ------------------------------------------------------------------ #
        # 7. Weighted aggregation of values                                    #
        #                                                                      #
        #    context[b, h, i, :] = Σ_j  attn_weights[b,h,i,j] * V[b,h,j,:]  #
        #                                                                      #
        #    einsum "bhij,bhjd->bhid":                                        #
        #      b → batch, h → head, i → query, j → key, d → Dh               #
        #    Result: (B, H, S, Dh)                                            #
        # ------------------------------------------------------------------ #
        context = torch.einsum("bhij,bhjd->bhid", attn_weights, V)
        #         (B, H, S, S) × (B, H, S, Dh) → (B, H, S, Dh)

        # ------------------------------------------------------------------ #
        # 8. Merge heads and project output                                    #
        #    (B, H, S, Dh) → (B, S, D) → (B, S, D)                          #
        # ------------------------------------------------------------------ #
        context = self._merge_heads(context)   # (B, S, D)
        out     = self.out_proj(context)       # (B, S, D)

        return out, attn_weights   # (B,S,D), (B,H,S,S)

    # ---------------------------------------------------------------------- #
    # Dunder helpers                                                           #
    # ---------------------------------------------------------------------- #

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"d_model={self.d_model}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"dropout={self.attn_drop.p})"
        )


# --------------------------------------------------------------------------- #
# Self-test block                                                               #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    print("=== CausalSelfAttention self-test ===\n")
    torch.manual_seed(42)

    B, S, D = 2, 10, 128
    H = 4

    attn = CausalSelfAttention(d_model=D, num_heads=H, dropout=0.0)
    attn.eval()
    print(f"Module : {attn}\n")

    x = torch.randn(B, S, D)

    # ------------------------------------------------------------------ #
    # Test 1 — basic output shapes                                         #
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        out, weights = attn(x)

    print(f"Input  shape : {tuple(x.shape)}")
    print(f"Output shape : {tuple(out.shape)}")
    print(f"Weights shape: {tuple(weights.shape)}")

    assert out.shape     == torch.Size([B, S, D]),    f"out shape mismatch: {out.shape}"
    assert weights.shape == torch.Size([B, H, S, S]), f"weights shape mismatch: {weights.shape}"
    print("[✓] Shape assertions passed.\n")

    # ------------------------------------------------------------------ #
    # Test 2 — no NaNs or Infs in outputs                                 #
    # ------------------------------------------------------------------ #
    assert not torch.isnan(out).any(),     "[FAIL] NaN found in output"
    assert not torch.isinf(out).any(),     "[FAIL] Inf found in output"
    assert not torch.isnan(weights).any(), "[FAIL] NaN found in attn_weights"
    print("[✓] No NaN / Inf in output or attention weights.\n")

    # ------------------------------------------------------------------ #
    # Test 3 — causal mask is enforced                                    #
    #                                                                      #
    #    Attention weights above the diagonal must all be exactly 0.      #
    #    We verify the strict upper triangle (future positions).          #
    # ------------------------------------------------------------------ #
    upper_tri_mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    # weights: (B, H, S, S) — check every batch & head
    future_weights = weights[:, :, upper_tri_mask]   # should be all zeros
    max_future = future_weights.abs().max().item()
    assert max_future == 0.0, f"[FAIL] Future tokens have non-zero weight: {max_future}"
    print(f"[✓] Causal mask enforced: max future-token weight = {max_future}\n")

    # ------------------------------------------------------------------ #
    # Test 4 — attention weights sum to 1 across key dimension            #
    #           (each query row is a valid probability distribution)       #
    # ------------------------------------------------------------------ #
    row_sums = weights.sum(dim=-1)           # (B, H, S)
    max_dev  = (row_sums - 1.0).abs().max().item()
    assert max_dev < 1e-5, f"[FAIL] Attention rows don't sum to 1; max dev = {max_dev}"
    print(f"[✓] Attention rows sum to 1.0 (max deviation = {max_dev:.2e})\n")

    # ------------------------------------------------------------------ #
    # Test 5 — optional padding mask (last 3 tokens are PAD)              #
    # ------------------------------------------------------------------ #
    pad_mask = torch.ones(B, S, dtype=torch.bool)
    pad_mask[:, -3:] = False    # last 3 positions are padding

    with torch.no_grad():
        out_pad, weights_pad = attn(x, attn_mask=pad_mask)

    assert not torch.isnan(out_pad).any(),     "[FAIL] NaN in padded output"
    assert not torch.isnan(weights_pad).any(), "[FAIL] NaN in padded attn_weights"

    # Key positions that are PAD should have zero attention weight
    pad_key_weights = weights_pad[:, :, :, -3:]   # (B, H, S, 3)
    max_pad_w = pad_key_weights.abs().max().item()
    assert max_pad_w == 0.0, f"[FAIL] PAD key positions have non-zero weight: {max_pad_w}"
    print(f"[✓] Padding mask works: max weight on PAD key positions = {max_pad_w}\n")

    # ------------------------------------------------------------------ #
    # Test 6 — error handling                                             #
    # ------------------------------------------------------------------ #
    print("--- Error handling ---")
    try:
        attn(torch.randn(2, 10))     # wrong dims
    except ValueError as e:
        print(f"[OK] Bad input dims   : {e}")

    try:
        attn(torch.randn(2, 10, 64))  # wrong d_model
    except ValueError as e:
        print(f"[OK] Wrong d_model    : {e}")

    try:
        attn(torch.randn(2, 2000, 128))  # seq > max_len
    except ValueError as e:
        print(f"[OK] seq > max_len    : {e}")

    try:
        bad_mask = torch.ones(2, 5, dtype=torch.bool)  # wrong mask shape
        attn(x, attn_mask=bad_mask)
    except ValueError as e:
        print(f"[OK] Bad mask shape   : {e}")

    print("\n[✓] All tests passed.")
    sys.exit(0)