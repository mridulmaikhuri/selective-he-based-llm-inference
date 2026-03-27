"""
Token + Sinusoidal Positional Embeddings
=========================================
Implements TokenPositionalEmbedding, which combines:
  - A learned token embedding table  (vocab_size → d_model)
  - Deterministic sinusoidal positional encodings (Vaswani et al., 2017)
  - Dropout applied to the summed representation

Reference:
    Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
    https://arxiv.org/abs/1706.03762  (Section 3.5)

Usage:
    from embeddings import TokenPositionalEmbedding
    import torch

    emb = TokenPositionalEmbedding(vocab_size=50257, d_model=128)
    ids = torch.randint(0, 50257, (4, 64))   # (batch, seq_len)
    out = emb(ids)                            # (4, 64, 128)
"""

import math
import torch
import torch.nn as nn


class TokenPositionalEmbedding(nn.Module):
    """Combines learned token embeddings with sinusoidal positional encodings.

    The positional encoding matrix is computed once at construction time and
    stored as a non-trainable buffer, exactly as described by Vaswani et al.:

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    The token embedding output and the positional encoding slice are summed,
    then passed through dropout.

    Args:
        vocab_size (int): Number of tokens in the vocabulary. Default: 50257 (GPT-2).
        d_model    (int): Embedding / model dimension. Default: 128.
        max_len    (int): Maximum supported sequence length. Default: 1024.
        dropout  (float): Dropout probability applied after summation. Default: 0.1.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 128,
        max_len: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        # Validate constructor arguments                                       #
        # ------------------------------------------------------------------ #
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        if d_model < 1 or d_model % 2 != 0:
            raise ValueError(f"d_model must be a positive even integer, got {d_model}")
        if max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.d_model = d_model
        self.max_len = max_len
        self.vocab_size = vocab_size

        # ------------------------------------------------------------------ #
        # Learned token embedding table                                        #
        # ------------------------------------------------------------------ #
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
        )
        # Initialise weights with Xavier uniform for stable early training
        nn.init.xavier_uniform_(self.token_embedding.weight.unsqueeze(0))

        # ------------------------------------------------------------------ #
        # Sinusoidal positional encoding (deterministic, stored as a buffer)  #
        # Shape: (1, max_len, d_model) — the leading 1 allows broadcast over  #
        # any batch size.                                                      #
        # ------------------------------------------------------------------ #
        pe = self._build_sinusoidal_encoding(max_len, d_model)  # (max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))             # (1, max_len, d_model)

        # ------------------------------------------------------------------ #
        # Dropout                                                              #
        # ------------------------------------------------------------------ #
        self.dropout = nn.Dropout(p=dropout)

    # ---------------------------------------------------------------------- #
    # Static helper: build the (max_len, d_model) sinusoidal matrix           #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _build_sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
        """Compute the sinusoidal positional encoding matrix.

        Returns:
            Tensor of shape (max_len, d_model) with dtype float32.
        """
        # positions: column vector (max_len, 1)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)

        # Division term: shape (d_model/2,)
        # exp(log(...)) form is numerically more stable than direct pow()
        half_dim = d_model // 2
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10_000.0) / d_model)
        )  # equivalent to 1 / (10000^(2i/d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)   # even indices
        pe[:, 1::2] = torch.cos(position * div_term)   # odd  indices

        return pe  # (max_len, d_model)

    # ---------------------------------------------------------------------- #
    # Forward pass                                                             #
    # ---------------------------------------------------------------------- #

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Compute token + positional embeddings.

        Args:
            input_ids (LongTensor): Token indices of shape (batch, seq_len).
                Values must satisfy 0 <= id < vocab_size.

        Returns:
            Tensor of shape (batch, seq_len, d_model), dtype float32.

        Raises:
            ValueError: If seq_len exceeds max_len.
            ValueError: If any token id is outside [0, vocab_size).
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2-D (batch, seq_len), got shape {tuple(input_ids.shape)}"
            )

        _, seq_len = input_ids.shape

        # Guard: sequence length
        if seq_len > self.max_len:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_len ({self.max_len}). "
                "Either truncate the input or increase max_len at construction time."
            )

        # Guard: vocabulary bounds
        if input_ids.min().item() < 0 or input_ids.max().item() >= self.vocab_size:
            bad_min = input_ids.min().item()
            bad_max = input_ids.max().item()
            raise ValueError(
                f"input_ids contain token id(s) outside [0, {self.vocab_size}). "
                f"Observed range: [{bad_min}, {bad_max}]."
            )

        # Token embeddings: (batch, seq_len, d_model)
        # Scale by sqrt(d_model) as recommended in Vaswani et al. §3.4
        tok_emb = self.token_embedding(input_ids) * math.sqrt(self.d_model)

        # Positional encoding slice: (1, seq_len, d_model) — broadcasts over batch
        pos_enc = self.pe[:, :seq_len, :]   # type: torch.Tensor

        # Sum and apply dropout
        return self.dropout(tok_emb + pos_enc)

    # ---------------------------------------------------------------------- #
    # Dunder helpers                                                           #
    # ---------------------------------------------------------------------- #

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"vocab_size={self.vocab_size}, "
            f"d_model={self.d_model}, "
            f"max_len={self.max_len}, "
            f"dropout={self.dropout.p})"
        )


# --------------------------------------------------------------------------- #
# Inline test                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("=== TokenPositionalEmbedding inline test ===\n")

    torch.manual_seed(0)

    # ------------------------------------------------------------------ #
    # Basic shape test (acceptance criterion)                             #
    # ------------------------------------------------------------------ #
    emb = TokenPositionalEmbedding(vocab_size=50257, d_model=128, max_len=1024, dropout=0.1)
    print(f"Module : {emb}\n")

    dummy_ids = torch.randint(0, 50257, (2, 10))   # (batch=2, seq_len=10)
    emb.eval()                                      # disable dropout for deterministic output
    with torch.no_grad():
        out = emb(dummy_ids)

    print(f"input_ids shape : {tuple(dummy_ids.shape)}")
    print(f"output shape    : {out.shape}")          # Expected: torch.Size([2, 10, 128])
    assert out.shape == torch.Size([2, 10, 128]), "Shape mismatch — test FAILED"
    print("\n[✓] Shape assertion passed.\n")

    # ------------------------------------------------------------------ #
    # Verify positional encoding is truly sinusoidal (first 4 dims)      #
    # ------------------------------------------------------------------ #
    pe_matrix = emb.pe.squeeze(0)   # (max_len, d_model)
    print("Positional encoding spot-check (pos=0, dims 0-3):")
    print(f"  {pe_matrix[0, :4].tolist()}")
    print("  Expected: [sin(0), cos(0), sin(0), cos(0)] = [0.0, 1.0, 0.0, 1.0]\n")

    # ------------------------------------------------------------------ #
    # Error handling tests                                                #
    # ------------------------------------------------------------------ #
    print("--- Error handling ---")

    # seq_len > max_len
    try:
        emb(torch.randint(0, 50257, (1, 2000)))
    except ValueError as exc:
        print(f"[OK] seq_len overflow caught  : {exc}")

    # vocab overflow
    try:
        emb(torch.tensor([[50257, 0]]))
    except ValueError as exc:
        print(f"[OK] vocab overflow caught    : {exc}")

    # wrong dimensionality
    try:
        emb(torch.randint(0, 50257, (10,)))
    except ValueError as exc:
        print(f"[OK] bad input dims caught    : {exc}")

    print("\n[✓] All tests passed.")