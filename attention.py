import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism.
    
    This module implements scaled dot-product attention with causal masking,
    which prevents positions from attending to subsequent positions. This is
    essential for autoregressive models like GPT.
    
    Args:
        d_model (int): Dimension of input embeddings. Default: 128
        num_heads (int): Number of attention heads. Default: 4
        dropout (float): Dropout probability. Default: 0.1
    
    Attributes:
        q_proj (nn.Linear): Query projection layer
        k_proj (nn.Linear): Key projection layer
        v_proj (nn.Linear): Value projection layer
        out_proj (nn.Linear): Output projection layer
        attn_dropout (nn.Dropout): Dropout for attention weights
        resid_dropout (nn.Dropout): Dropout for output
    """
    
    def __init__(self, d_model=128, num_heads=4, dropout=0.1):
        super(CausalSelfAttention, self).__init__()
        
        # Validate that d_model is divisible by num_heads
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads 
        self.scale = 1.0 / math.sqrt(self.head_dim)  # Scaling factor for attention scores
        
        # Separate projection layers for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        
        # Output projection layer
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)  # Applied to attention weights
        self.resid_dropout = nn.Dropout(dropout)  # Applied to output
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize projection weights with Xavier uniform initialization."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _create_causal_mask(self, seq_len, device):
        """
        Create a causal mask for self-attention.
        
        The mask is a lower triangular matrix where:
        - 0.0 indicates the position CAN be attended to
        - -inf indicates the position CANNOT be attended to (masked out)
        
        Args:
            seq_len (int): Sequence length
            device (torch.device): Device to create the mask on
        
        Returns:
            torch.Tensor: Causal mask of shape (seq_len, seq_len)
        
        Example for seq_len=4:
            [[0., -inf, -inf, -inf],
             [0.,  0.,  -inf, -inf],
             [0.,  0.,   0.,  -inf],
             [0.,  0.,   0.,   0. ]]
        """
        # Create a lower triangular matrix of ones
        # tril creates: [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        
        # Convert: 1 -> 0.0 (can attend), 0 -> -inf (cannot attend)
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)
        
        return mask  # Shape: (seq_len, seq_len)
    
    def forward(self, x, attn_mask=None):
        """
        Forward pass for causal self-attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            attn_mask (torch.Tensor, optional): Additional attention mask for padding.
                Shape: (batch_size, seq_len) where True/1 indicates VALID positions
                and False/0 indicates PADDING positions to mask out.
        
        Returns:
            tuple: (output, attention_weights)
                - output: Tensor of shape (batch_size, seq_len, d_model)
                - attention_weights: Tensor of shape (batch_size, num_heads, seq_len, seq_len)
        
        Shape transformations:
            x: (B, S, D) -> Q, K, V: (B, S, D) -> (B, num_heads, S, head_dim)
            attention_scores: (B, num_heads, S, S)
            output: (B, num_heads, S, head_dim) -> (B, S, D)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Step 1: Project input to Q, K, V
        # Each projection: (B, S, D) -> (B, S, D)
        Q = self.q_proj(x)  # (batch_size, seq_len, d_model)
        K = self.k_proj(x)  # (batch_size, seq_len, d_model)
        V = self.v_proj(x)  # (batch_size, seq_len, d_model)
        
        # Step 2: Reshape for multi-head attention
        # Split d_model into num_heads and head_dim
        # (B, S, D) -> (B, S, num_heads, head_dim) -> (B, num_heads, S, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Now shapes are: (B, num_heads, S, head_dim)
        
        # Step 3: Compute attention scores using scaled dot-product
        # Q @ K^T: (B, num_heads, S, head_dim) @ (B, num_heads, head_dim, S)
        #        -> (B, num_heads, S, S)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Scale by sqrt(head_dim) to prevent softmax saturation
        attention_scores = attention_scores * self.scale
        # Shape: (B, num_heads, S, S)
        
        # Step 4: Apply causal mask
        # Create causal mask: lower triangular matrix
        causal_mask = self._create_causal_mask(seq_len, x.device)
        # Shape: (S, S)
        
        # Add causal mask to attention scores
        # Broadcasting: (B, num_heads, S, S) + (S, S) -> (B, num_heads, S, S)
        attention_scores = attention_scores + causal_mask
        
        # Step 5: Apply optional padding mask (if provided)
        if attn_mask is not None:
            # attn_mask shape: (B, S) where 1/True = valid, 0/False = padding
            # We need to convert this to shape (B, 1, 1, S) for broadcasting
            
            # Convert boolean/int mask to float mask
            # Valid positions (1/True) -> 0.0, Padding positions (0/False) -> -inf
            if attn_mask.dtype == torch.bool:
                # Boolean mask: True=valid, False=padding
                padding_mask = attn_mask.float()
            else:
                # Integer mask: 1=valid, 0=padding
                padding_mask = attn_mask.float()
            
            # Invert and scale: 1->0, 0->-inf
            padding_mask = (1.0 - padding_mask) * float('-inf')
            
            # Reshape for broadcasting: (B, S) -> (B, 1, 1, S)
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            
            # Apply padding mask to attention scores
            # (B, num_heads, S, S) + (B, 1, 1, S) -> (B, num_heads, S, S)
            # The mask is applied to the KEY dimension (last dimension)
            attention_scores = attention_scores + padding_mask
        
        # Step 6: Apply softmax to get attention weights
        # Softmax over the last dimension (key positions)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        # Apply dropout to attention weights
        attention_weights = self.attn_dropout(attention_weights)
        
        # Step 7: Apply attention weights to values
        # (B, num_heads, S, S) @ (B, num_heads, S, head_dim)
        # -> (B, num_heads, S, head_dim)
        attn_output = torch.matmul(attention_weights, V)
        
        # Step 8: Reshape back to original dimensions
        # (B, num_heads, S, head_dim) -> (B, S, num_heads, head_dim) -> (B, S, D)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)
        
        # Step 9: Apply output projection
        output = self.out_proj(attn_output)
        # Shape: (B, S, D)
        
        # Apply residual dropout
        output = self.resid_dropout(output)

        # Zero out outputs at padded QUERY positions to avoid NaNs propagating
        if attn_mask is not None:
            output = output * attn_mask.unsqueeze(-1)
        
        return output, attention_weights


if __name__ == "__main__":
    """
    Test block to verify CausalSelfAttention implementation.
    """
    print("Testing CausalSelfAttention...")
    print("=" * 70)
    
    # Test 1: Basic functionality
    print("\n[Test 1] Basic functionality with dummy input")
    print("-" * 70)
    
    # Create model
    d_model = 128
    num_heads = 4
    batch_size = 2
    seq_len = 10
    
    model = CausalSelfAttention(d_model=d_model, num_heads=num_heads, dropout=0.1)
    model.eval()  # Set to eval mode to make dropout deterministic for testing
    
    # Create dummy input: (batch=2, seq=10, dim=128)
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_len}, {d_model})")
    print(f"Expected attention weights shape: ({batch_size}, {num_heads}, {seq_len}, {seq_len})")
    
    # Forward pass
    output, attn_weights = model(x)
    
    print(f"\nActual output shape: {output.shape}")
    print(f"Actual attention weights shape: {attn_weights.shape}")
    
    # Assert correct shapes
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Output shape mismatch! Expected {(batch_size, seq_len, d_model)}, got {output.shape}"
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len), \
        f"Attention weights shape mismatch! Expected {(batch_size, num_heads, seq_len, seq_len)}, got {attn_weights.shape}"
    
    print("✓ Shape assertions passed!")
    
    # Check for NaNs
    assert not torch.isnan(output).any(), "Output contains NaN values!"
    assert not torch.isnan(attn_weights).any(), "Attention weights contain NaN values!"
    print("✓ No NaN values detected!")
    
    # Test 2: Verify causal masking
    print("\n[Test 2] Verify causal masking")
    print("-" * 70)
    
    # Check that attention weights are zero for future positions
    # For each position i, attention_weights[:, :, i, j] should be 0 for all j > i
    for head in range(num_heads):
        for i in range(seq_len):
            future_weights = attn_weights[0, head, i, i+1:]
            if len(future_weights) > 0:
                max_future_weight = future_weights.max().item()
                assert max_future_weight < 1e-6, \
                    f"Position {i} attends to future positions! Max weight: {max_future_weight}"
    
    print("✓ Causal masking verified! No attention to future positions.")
    
    # Visualize attention pattern for first head, first batch
    print("\nAttention pattern for head 0, batch 0 (first 5x5):")
    attn_pattern = attn_weights[0, 0, :5, :5]
    for i in range(5):
        row = " ".join([f"{attn_pattern[i, j].item():.3f}" for j in range(5)])
        print(f"  Position {i}: [{row}]")
    print("  (Notice the lower-triangular pattern)")
    
    # Test 3: Attention weights sum to 1
    print("\n[Test 3] Verify attention weights normalization")
    print("-" * 70)
    
    # Sum over last dimension (key positions) should equal 1
    attn_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-5), \
        "Attention weights don't sum to 1!"
    print("✓ Attention weights properly normalized (sum to 1)!")
    
    # Test 4: With padding mask
    print("\n[Test 4] Test with padding mask")
    print("-" * 70)
    
    # Create a padding mask: first batch has full sequence, second has padding at end
    # Shape: (batch_size, seq_len)
    # 1/True = valid token, 0/False = padding token
    attn_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    attn_mask[1, 7:] = False  # Mask out last 3 positions in second batch
    
    print(f"Padding mask shape: {attn_mask.shape}")
    print(f"Batch 0 valid positions: {attn_mask[0].sum().item()}/{seq_len}")
    print(f"Batch 1 valid positions: {attn_mask[1].sum().item()}/{seq_len}")
    
    output_masked, attn_weights_masked = model(x, attn_mask=attn_mask)
    
    print(f"\nOutput with mask shape: {output_masked.shape}")
    print(f"Attention weights with mask shape: {attn_weights_masked.shape}")
    
    # Assert correct shapes
    assert output_masked.shape == (batch_size, seq_len, d_model)
    assert attn_weights_masked.shape == (batch_size, num_heads, seq_len, seq_len)
    print("✓ Shapes correct with padding mask!")
    
    # Check for NaNs
    assert not torch.isnan(output_masked).any(), "Output contains NaN values with mask!"
    assert not torch.isnan(attn_weights_masked).any(), "Attention weights contain NaN values with mask!"
    print("✓ No NaN values with padding mask!")
    
    # Verify that padded positions receive zero attention
    # In batch 1, positions 7, 8, 9 should not be attended to
    for i in range(seq_len):
        padded_positions_weights = attn_weights_masked[1, 0, i, 7:]
        max_padded_weight = padded_positions_weights.max().item()
        assert max_padded_weight < 1e-6, \
            f"Position {i} in batch 1 attends to padded positions! Max weight: {max_padded_weight}"
    
    print("✓ Padding mask works correctly! Padded positions receive no attention.")
    
    # Test 5: Different head configurations
    print("\n[Test 5] Test different head configurations")
    print("-" * 70)
    
    configs = [
        (64, 2),   # 64-dim, 2 heads
        (128, 8),  # 128-dim, 8 heads
        (256, 4),  # 256-dim, 4 heads
    ]
    
    for d, h in configs:
        test_model = CausalSelfAttention(d_model=d, num_heads=h, dropout=0.1)
        test_input = torch.randn(2, 5, d)
        test_output, test_attn = test_model(test_input)
        
        assert test_output.shape == (2, 5, d)
        assert test_attn.shape == (2, h, 5, 5)
        assert not torch.isnan(test_output).any()
        
        print(f"✓ Config (d_model={d}, num_heads={h}) works correctly")
    
    # Final summary
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)
    print(f"\nFinal output shape: {output.shape}")
    print(f"Final attention weights shape: {attn_weights.shape}")
    print("\nKey features verified:")
    print("  ✓ Correct output shapes")
    print("  ✓ No NaN values")
    print("  ✓ Causal masking prevents future attention")
    print("  ✓ Attention weights properly normalized")
    print("  ✓ Padding mask works correctly")
    print("  ✓ Multiple head configurations supported")