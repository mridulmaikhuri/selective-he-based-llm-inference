"""
model.py
========
TinyGPT — a compact GPT-style language model assembling:
    TokenPositionalEmbedding → N × TransformerBlock → LayerNorm → LM Head

Weight tying: lm_head.weight is shared with token_embedding.weight, so the
vocabulary projection reuses the learned token representations and the tied
parameters are counted only once.

Parameter budget (defaults):
    Token embedding   : 50257 × 128 =  6,432,896   ← also used by lm_head (tied)
    Pos  embedding    :  1024 × 128 =    131,072
    4 × TransformerBlock            =    791,040
    Final LayerNorm                 =        256
    lm_head (tied, not re-counted)  =          0
    ─────────────────────────────────────────────
    Total                           ~  7,355,264

NOTE: The ~1–2 M target assumes vocab_size is small. With GPT-2's full
vocab of 50 257 the token embedding alone is ~6.4 M. To land in the 1–2 M
range set vocab_size=1000 (see __main__ demo). The full-vocab model is kept
as the canonical default so weights are GPT-2 compatible; the __main__ block
demonstrates both.
"""

import math
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Import sub-modules from the same package / directory.
# If you placed CausalSelfAttention and FeedForward in transformer_block.py,
# they are re-imported here transitively through TransformerBlock.
# ---------------------------------------------------------------------------
from transformer_block import TransformerBlock


# ---------------------------------------------------------------------------
# Token + Positional Embedding
# ---------------------------------------------------------------------------

class TokenPositionalEmbedding(nn.Module):
    """
    Learnable token embedding + learnable positional embedding.

    Args:
        vocab_size  : size of the token vocabulary
        d_model     : embedding dimension
        max_seq_len : maximum sequence length supported (default 1024)
        dropout     : embedding dropout probability
    """

    def __init__(
        self,
        vocab_size:   int   = 50257,
        d_model:      int   = 128,
        max_seq_len:  int   = 1024,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop      = nn.Dropout(dropout)

        # Scale token embeddings at init (GPT-2 convention)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, S) long tensor of token indices
        Returns:
            (B, S, d_model) float tensor
        """
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)  # (1, S)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        return self.drop(x)


# ---------------------------------------------------------------------------
# TinyGPT
# ---------------------------------------------------------------------------

class TinyGPT(nn.Module):
    """
    Minimal GPT-style language model.

    Architecture:
        input_ids (B,S)
            ↓
        TokenPositionalEmbedding  →  (B, S, d_model)
            ↓
        TransformerBlock × num_layers
            ↓
        LayerNorm
            ↓
        lm_head  (weight tied to token embedding)  →  logits (B, S, vocab_size)

    Args:
        num_layers : number of stacked TransformerBlock layers  (default: 4)
        vocab_size : vocabulary size                            (default: 50257)
        d_model    : embedding / model dimension                (default: 128)
        num_heads  : attention heads per block                  (default: 4)
        d_ff       : feed-forward inner dimension               (default: 512)
        dropout    : dropout probability used throughout        (default: 0.1)
        max_seq_len: maximum supported sequence length          (default: 1024)
    """

    def __init__(
        self,
        num_layers:  int   = 4,
        vocab_size:  int   = 50257,
        d_model:     int   = 128,
        num_heads:   int   = 4,
        d_ff:        int   = 512,
        dropout:     float = 0.1,
        max_seq_len: int   = 1024,
    ):
        super().__init__()

        self.d_model    = d_model
        self.vocab_size = vocab_size

        # ── Embedding layer ──────────────────────────────────────────────────
        self.embedding = TokenPositionalEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # ── Transformer stack ────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # ── Final layer norm (Pre-LN convention) ────────────────────────────
        self.norm = nn.LayerNorm(d_model)

        # ── LM head with weight tying ────────────────────────────────────────
        # lm_head projects d_model → vocab_size.  Its weight matrix is TIED to
        # the token embedding weight so both share the same parameters.
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.token_emb.weight  # weight tying

        # Apply GPT-2-style residual projection scaling at init
        self._init_weights(num_layers)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, num_layers: int) -> None:
        """
        GPT-2 style init:
          • Linear / Embedding → N(0, 0.02)
          • Residual projections scaled by 1/√(2 × num_layers)
          • LayerNorm bias=0, weight=1
        """
        scale = 1.0 / math.sqrt(2 * num_layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Scale output projections of attention and FFN
        for block in self.blocks:
            nn.init.normal_(block.attention.out_proj.weight, std=scale)
            nn.init.normal_(block.feed_forward.net[-1].weight, std=scale)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, S) long tensor of token indices
            attn_mask: optional additive attention mask (B,1,S,S) or (1,1,S,S)
        Returns:
            logits: (B, S, vocab_size) float tensor
        """
        x = self.embedding(input_ids)          # (B, S, d_model)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)  # (B, S, d_model)

        x = self.norm(x)                        # (B, S, d_model)
        logits = self.lm_head(x)               # (B, S, vocab_size)
        return logits


# ---------------------------------------------------------------------------
# Utility: parameter counter
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """
    Count the number of *trainable* parameters in a model.

    Tied parameters (sharing the same storage) are counted only ONCE,
    which matters here because lm_head.weight is tied to token_emb.weight.

    Args:
        model: any nn.Module
    Returns:
        total unique trainable parameter count (int)
    Prints:
        human-friendly summary to stdout
    """
    seen: set[int] = set()
    total = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = param.data_ptr()          # unique memory address → detect ties
        if pid in seen:
            continue
        seen.add(pid)
        total += param.numel()

    # Human-friendly formatting
    if total >= 1_000_000:
        friendly = f"{total / 1_000_000:.2f} M"
    elif total >= 1_000:
        friendly = f"{total / 1_000:.1f} K"
    else:
        friendly = str(total)

    print(f"  Trainable parameters : {total:>12,}  ({friendly})")
    return total


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # ── Demo 1: small vocab → lands in the 1–2 M target range ───────────────
    print("=" * 60)
    print("Demo 1 — small vocab (vocab_size=1 000)  →  ~1–2 M params")
    print("=" * 60)

    small_model = TinyGPT(
        num_layers=4,
        vocab_size=1_000,   # small vocab to hit 1–2 M target
        d_model=128,
        num_heads=4,
        d_ff=512,
        dropout=0.1,
    )
    small_model.eval()

    dummy = torch.randint(0, 1_000, (2, 10))
    with torch.no_grad():
        logits = small_model(dummy)

    print(f"  Input  shape : {dummy.shape}")
    print(f"  Logits shape : {logits.shape}")
    n1 = count_parameters(small_model)
    assert logits.shape == torch.Size([2, 10, 1_000])
    assert 1_000_000 <= n1 <= 2_000_000, f"Expected 1–2 M params, got {n1:,}"
    print("  ✓ Shape and param-count assertions passed.\n")

    # ── Demo 2: full GPT-2 vocab (default) ──────────────────────────────────
    print("=" * 60)
    print("Demo 2 — full GPT-2 vocab (vocab_size=50 257)  →  ~7 M params")
    print("=" * 60)
    print(
        "  NOTE: With vocab_size=50257 the token embedding alone is\n"
        "  50257×128 ≈ 6.4 M parameters — well above the 1–2 M target.\n"
        "  The default is kept for GPT-2 tokenizer compatibility.\n"
        "  Use vocab_size=1000 (Demo 1) to stay in the 1–2 M range."
    )

    full_model = TinyGPT()   # all defaults
    full_model.eval()

    dummy2 = torch.randint(0, 50257, (2, 10))
    with torch.no_grad():
        logits2 = full_model(dummy2)

    print(f"  Input  shape : {dummy2.shape}")
    print(f"  Logits shape : {logits2.shape}")
    count_parameters(full_model)
    assert logits2.shape == torch.Size([2, 10, 50257])
    print("  ✓ Shape assertion passed.")