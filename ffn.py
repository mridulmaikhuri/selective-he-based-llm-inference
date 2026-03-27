# ffn.py
"""
Transformer Feed-Forward Block
================================
Implements the position-wise feed-forward network (FFN) used in every
transformer layer. Each token's representation is projected independently
through a two-layer MLP with a GELU non-linearity in between.

The canonical form (Vaswani et al., 2017) uses ReLU; modern practice
(GPT-2 / BERT) swaps in GELU for smoother gradients.

Structure:
    x → Linear(d_model, d_ff) → GELU → Dropout → Linear(d_ff, d_model) → Dropout → out

Reference:
    Vaswani et al., "Attention Is All You Need", NeurIPS 2017. §3.3
    https://arxiv.org/abs/1706.03762

Usage:
    from ffn import FeedForward
    import torch

    ff  = FeedForward(d_model=128, d_ff=512, dropout=0.1)
    x   = torch.randn(2, 10, 128)   # (batch, seq_len, d_model)
    out = ff(x)                      # (2, 10, 128)
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise two-layer feed-forward network.

    Applied identically and independently to every position in the sequence.
    The inner dimension d_ff (commonly 4 × d_model) acts as an expansion
    bottleneck: the model can route token representations through a richer
    feature space before projecting back.

    Dropout is applied twice:
      1. After the GELU activation  — regularises the hidden representation.
      2. After the output projection — regularises the residual contribution
         before it is added back in the surrounding TransformerBlock.

    Args:
        d_model  (int):   Input and output dimension. Default: 128.
        d_ff     (int):   Inner (expansion) dimension. Default: 512.
        dropout  (float): Dropout probability (both dropout layers share
                          the same rate). Default: 0.1.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        # Argument validation                                                  #
        # ------------------------------------------------------------------ #
        if d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {d_model}")
        if d_ff < 1:
            raise ValueError(f"d_ff must be >= 1, got {d_ff}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.d_model = d_model
        self.d_ff    = d_ff

        # ------------------------------------------------------------------ #
        # Layers                                                               #
        #                                                                      #
        # nn.Sequential fuses the sub-layers into a single callable with no   #
        # intermediate Python variables, avoiding accidental extra copies and  #
        # keeping forward() a one-liner.                                       #
        #                                                                      #
        # Layer breakdown:                                                     #
        #   fc1     : (B, S, d_model) → (B, S, d_ff)   expansion             #
        #   gelu    : element-wise, shape unchanged                            #
        #   drop1   : randomly zeros hidden units during training              #
        #   fc2     : (B, S, d_ff)    → (B, S, d_model) projection            #
        #   drop2   : randomly zeros output units during training              #
        # ------------------------------------------------------------------ #
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),   # fc1 — expand
            nn.GELU(),                   # smooth non-linearity
            nn.Dropout(p=dropout),       # drop1 — hidden regularisation
            nn.Linear(d_ff, d_model),   # fc2 — project back
            nn.Dropout(p=dropout),       # drop2 — output regularisation
        )

        self._init_weights()

    # ---------------------------------------------------------------------- #
    # Weight initialisation                                                    #
    # ---------------------------------------------------------------------- #

    def _init_weights(self) -> None:
        """Initialise linear layers with Xavier uniform + zero bias.

        Xavier uniform keeps the variance of activations roughly constant
        across layers at the start of training, preventing vanishing /
        exploding signals.
        """
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ---------------------------------------------------------------------- #
    # Forward                                                                  #
    # ---------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward block to every token position.

        Args:
            x (Tensor): Input of shape (B, S, d_model).

        Returns:
            Tensor of shape (B, S, d_model), same dtype as input.

        Raises:
            ValueError: If x is not 3-D or its last dimension ≠ d_model.
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3-D input (B, S, d_model), got shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Last dimension of input ({x.shape[-1]}) "
                f"does not match d_model ({self.d_model})"
            )

        # nn.Sequential applies fc1 → GELU → drop1 → fc2 → drop2 in one call.
        # No intermediate tensors are held in Python scope, so peak memory is
        # determined by PyTorch's autograd graph, not by extra local variables.
        return self.net(x)   # (B, S, d_model)

    # ---------------------------------------------------------------------- #
    # Helpers                                                                  #
    # ---------------------------------------------------------------------- #

    def extra_repr(self) -> str:
        """Shown by print(model) alongside the default nn.Sequential repr."""
        return (
            f"d_model={self.d_model}, "
            f"d_ff={self.d_ff}, "
            f"dropout={self.net[2].p}"
        )


# --------------------------------------------------------------------------- #
# Self-test                                                                     #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    print("=== FeedForward self-test ===\n")
    torch.manual_seed(0)

    B, S, D = 2, 10, 128

    ff = FeedForward(d_model=D, d_ff=512, dropout=0.1)
    ff.eval()   # disable dropout for deterministic output
    print(ff)
    print()

    x = torch.randn(B, S, D)

    # ------------------------------------------------------------------ #
    # Test 1 — output shape and dtype                                      #
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        out = ff(x)

    print(f"Input  shape : {tuple(x.shape)}")
    print(f"Output shape : {tuple(out.shape)}")
    print(f"Input  dtype : {x.dtype}")
    print(f"Output dtype : {out.dtype}")

    assert out.shape == torch.Size([B, S, D]), f"Shape mismatch: {out.shape}"
    assert out.dtype == x.dtype,              f"dtype changed: {out.dtype}"
    print("[✓] Shape and dtype assertions passed.\n")

    # ------------------------------------------------------------------ #
    # Test 2 — no NaNs or Infs                                            #
    # ------------------------------------------------------------------ #
    assert not torch.isnan(out).any(), "[FAIL] NaN in output"
    assert not torch.isinf(out).any(), "[FAIL] Inf in output"
    print("[✓] No NaN / Inf in output.\n")

    # ------------------------------------------------------------------ #
    # Test 3 — output differs from input (transformation is non-trivial)  #
    # ------------------------------------------------------------------ #
    assert not torch.allclose(out, x), "[FAIL] Output identical to input"
    print("[✓] Output is a non-trivial transformation of input.\n")

    # ------------------------------------------------------------------ #
    # Test 4 — dropout is active during training (output changes)         #
    # ------------------------------------------------------------------ #
    ff.train()
    with torch.no_grad():
        out_a = ff(x)
        out_b = ff(x)
    assert not torch.allclose(out_a, out_b), "[FAIL] Dropout inactive in train mode"
    print("[✓] Dropout produces stochastic output in train mode.\n")

    # ------------------------------------------------------------------ #
    # Test 5 — dropout is inactive during eval (output is deterministic)  #
    # ------------------------------------------------------------------ #
    ff.eval()
    with torch.no_grad():
        out_c = ff(x)
        out_d = ff(x)
    assert torch.allclose(out_c, out_d), "[FAIL] Output non-deterministic in eval mode"
    print("[✓] Eval mode is deterministic.\n")

    # ------------------------------------------------------------------ #
    # Test 6 — half-precision passthrough                                 #
    # ------------------------------------------------------------------ #
    ff_half = FeedForward(d_model=D, d_ff=512, dropout=0.0).half().eval()
    x_half  = x.half()
    with torch.no_grad():
        out_half = ff_half(x_half)
    assert out_half.dtype  == torch.float16,       "[FAIL] dtype not float16"
    assert out_half.shape  == torch.Size([B, S, D]), "[FAIL] shape wrong in fp16"
    print("[✓] float16 passthrough works.\n")

    # ------------------------------------------------------------------ #
    # Test 7 — error handling                                             #
    # ------------------------------------------------------------------ #
    ff.eval()
    print("--- Error handling ---")
    try:
        ff(torch.randn(2, 128))        # 2-D input
    except ValueError as e:
        print(f"[OK] Bad dims    : {e}")

    try:
        ff(torch.randn(2, 10, 64))     # wrong d_model
    except ValueError as e:
        print(f"[OK] Wrong d_model : {e}")

    try:
        FeedForward(d_model=0)
    except ValueError as e:
        print(f"[OK] Bad d_model   : {e}")

    try:
        FeedForward(dropout=1.0)
    except ValueError as e:
        print(f"[OK] Bad dropout   : {e}")

    print("\n[✓] All tests passed.")
    sys.exit(0)