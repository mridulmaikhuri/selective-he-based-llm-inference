"""
transformer_block.py
====================
A modular Transformer block using Pre-LayerNorm (Pre-LN), causal self-attention,
feed-forward network, residual connections, and dropout.

Architecture (Pre-LN variant):
    x → LayerNorm → CausalSelfAttention → dropout → + x  (residual)
      → LayerNorm → FeedForward          → dropout → + x  (residual)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Sub-modules (kept in the same file for portability; import them externally
# once you split into separate files).
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (or masked) self-attention.

    Args:
        d_model  : total embedding dimension
        num_heads: number of attention heads (d_model must be divisible by num_heads)
        dropout  : attention-weight dropout probability
    """

    def __init__(self, d_model: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_head    = d_model // num_heads
        self.scale     = math.sqrt(self.d_head)

        # Fused QKV projection
        self.qkv_proj  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj   = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x        : (B, S, d_model)
            attn_mask: (B, 1, S, S) or (1, 1, S, S) additive mask (−inf blocks attention)
        Returns:
            (B, S, d_model)
        """
        B, S, _ = x.shape

        # Project and split into Q, K, V
        qkv = self.qkv_proj(x)                                    # (B, S, 3*d_model)
        qkv = qkv.reshape(B, S, 3, self.num_heads, self.d_head)   # (B, S, 3, H, d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)                          # (3, B, H, S, d_head)
        q, k, v = qkv.unbind(0)                                    # each: (B, H, S, d_head)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, S, S)

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask  # additive mask (−inf → 0 after softmax)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # Aggregate values
        out = torch.matmul(attn_weights, v)                        # (B, H, S, d_head)
        out = out.transpose(1, 2).reshape(B, S, self.d_model)      # (B, S, d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network: Linear → GELU → Linear.

    Args:
        d_model : input/output dimension
        d_ff    : hidden (inner) dimension
        dropout : dropout probability applied after the first activation
    """

    def __init__(self, d_model: int = 128, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, d_model)
        Returns:
            (B, S, d_model)
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    A single Transformer block with Pre-LayerNorm, causal self-attention,
    feed-forward sub-layer, residual connections, and dropout.

    Pre-LN order (more stable training than Post-LN):
        y = x + Dropout(Attention(LayerNorm(x), attn_mask))
        z = y + Dropout(FFN(LayerNorm(y)))

    Args:
        d_model  : embedding / model dimension          (default: 128)
        num_heads: number of attention heads            (default: 4)
        d_ff     : feed-forward hidden dimension        (default: 512)
        dropout  : dropout probability used throughout  (default: 0.1)
    """

    def __init__(
        self,
        d_model:   int   = 128,
        num_heads: int   = 4,
        d_ff:      int   = 512,
        dropout:   float = 0.1,
    ):
        super().__init__()

        # Pre-LN layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Sub-layers
        self.attention    = CausalSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Residual dropout (applied to sub-layer output before adding residual)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x:         torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x        : input tensor of shape (B, S, d_model)
            attn_mask: optional additive attention mask (B, 1, S, S) or (1, 1, S, S)
        Returns:
            output tensor of shape (B, S, d_model)
        """
        # ── Attention sub-layer (Pre-LN + residual) ──────────────────────────
        residual = x
        x = self.norm1(x)
        x = self.attention(x, attn_mask=attn_mask)
        x = self.drop1(x)
        x = x + residual

        # ── Feed-forward sub-layer (Pre-LN + residual) ───────────────────────
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.drop2(x)
        x = x + residual

        return x  # (B, S, d_model)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # Instantiate with default hyperparameters
    block = TransformerBlock(d_model=128, num_heads=4, d_ff=512, dropout=0.1)
    block.eval()  # disable dropout for deterministic output

    # Dummy input: batch=2, sequence_len=10, d_model=128
    dummy_input = torch.randn(2, 10, 128)

    with torch.no_grad():
        output = block(dummy_input)

    print("Input  shape :", dummy_input.shape)   # torch.Size([2, 10, 128])
    print("Output shape :", output.shape)         # torch.Size([2, 10, 128])

    # Verify acceptance criterion
    assert output.shape == torch.Size([2, 10, 128]), (
        f"Shape mismatch: expected [2, 10, 128], got {list(output.shape)}"
    )
    print("\n✓ Acceptance test passed — output shape is torch.Size([2, 10, 128])")

    # Optional: parameter count
    n_params = sum(p.numel() for p in block.parameters())
    print(f"  Total parameters : {n_params:,}")