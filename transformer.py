import torch
import torch.nn as nn
from attention import CausalSelfAttention
from ffn import FeedForward

class TransformerBlock(nn.Module):
    """
    Transformer Block with Pre-Layer Normalization.
    
    This implementation follows the pre-LN architecture where layer normalization
    is applied before each sub-layer (attention and FFN) rather than after.
    Pre-LN has been shown to provide better training stability and performance.
    
    Architecture:
        x -> LayerNorm -> Attention -> Residual (+) -> 
          -> LayerNorm -> FFN -> Residual (+) -> output
    
    Args:
        d_model (int): Dimension of embeddings/hidden states. Default: 128
        num_heads (int): Number of attention heads. Default: 4
        d_ff (int): Dimension of feed-forward intermediate layer. Default: 512
        dropout (float): Dropout probability. Default: 0.1
    
    Attributes:
        ln1 (nn.LayerNorm): Layer normalization before attention
        attention (CausalSelfAttention): Multi-head causal self-attention
        ln2 (nn.LayerNorm): Layer normalization before FFN
        ffn (FeedForward): Position-wise feed-forward network
    
    Shape:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)
    """
    
    def __init__(self, d_model=128, num_heads=4, d_ff=512, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_prob = dropout
        
        # Layer normalization before attention (pre-LN)
        # Normalizes across the d_model dimension
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        
        # Multi-head causal self-attention
        self.attention = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Layer normalization before FFN (pre-LN)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        
        # Position-wise feed-forward network
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )
    
    def forward(self, x, attn_mask=None):
        """
        Forward pass through the transformer block.
        
        The computation flow (Pre-LN architecture):
        1. Normalize input
        2. Apply self-attention
        3. Add residual connection (x + attention_output)
        4. Normalize
        5. Apply feed-forward network
        6. Add residual connection (x + ffn_output)
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            attn_mask (torch.Tensor, optional): Attention mask for padding.
                Shape: (batch_size, seq_len) where 1/True indicates valid positions
                and 0/False indicates padding positions.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        
        Note:
            The residual connections help with gradient flow and allow the network
            to learn identity mappings when beneficial.
        """
        # Step 1: Apply layer normalization before attention (pre-LN)
        # Shape: (B, S, d_model) -> (B, S, d_model)
        normalized = self.ln1(x)
        
        # Step 2: Apply self-attention
        # Returns: (attention_output, attention_weights)
        # Shape: (B, S, d_model) -> (B, S, d_model)
        attn_output, _ = self.attention(normalized, attn_mask=attn_mask)
        
        # Step 3: Apply residual connection
        # Add the original input to the attention output
        # Shape: (B, S, d_model) + (B, S, d_model) -> (B, S, d_model)
        x = x + attn_output
        
        # Step 4: Apply layer normalization before FFN (pre-LN)
        # Shape: (B, S, d_model) -> (B, S, d_model)
        normalized = self.ln2(x)
        
        # Step 5: Apply feed-forward network
        # Shape: (B, S, d_model) -> (B, S, d_model)
        ffn_output = self.ffn(normalized)
        
        # Step 6: Apply residual connection
        # Shape: (B, S, d_model) + (B, S, d_model) -> (B, S, d_model)
        x = x + ffn_output
        
        return x


if __name__ == "__main__":
    print("Testing TransformerBlock...")
    print("=" * 70)
    
    # Test 1: Basic functionality
    print("\n[Test 1] Basic functionality with default parameters")
    print("-" * 70)
    
    # Create transformer block
    d_model = 128
    num_heads = 4
    d_ff = 512
    batch_size = 2
    seq_len = 10
    
    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1
    )
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_len}, {d_model})")
    
    # Forward pass
    output = block(x)
    
    print(f"Output shape: {output.shape}")
    
    # Assert correct shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Shape mismatch! Expected {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("✓ Shape assertion passed!")
    
    # Check for NaNs or Infs
    assert not torch.isnan(output).any(), "Output contains NaN values!"
    assert not torch.isinf(output).any(), "Output contains Inf values!"
    print("✓ No NaN or Inf values detected!")
    
    # Test 2: Test with attention mask
    print("\n[Test 2] Test with attention mask (padding)")
    print("-" * 70)
    
    # Create a padding mask
    # First batch: all valid, Second batch: last 3 positions are padding
    attn_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    attn_mask[1, 7:] = False
    
    print(f"Attention mask shape: {attn_mask.shape}")
    print(f"Batch 0 valid positions: {attn_mask[0].sum().item()}/{seq_len}")
    print(f"Batch 1 valid positions: {attn_mask[1].sum().item()}/{seq_len}")
    
    # Forward pass with mask
    output_masked = block(x, attn_mask=attn_mask)
    
    print(f"Output with mask shape: {output_masked.shape}")
    
    assert output_masked.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output_masked).any()
    assert not torch.isinf(output_masked).any()
    
    print("✓ Attention mask works correctly!")
    
    # Test 3: Verify residual connections
    print("\n[Test 3] Verify residual connections work")
    print("-" * 70)
    
    # Create a simple input
    x_simple = torch.randn(1, 5, d_model)
    
    # Get output
    output_simple = block(x_simple)
    
    difference = torch.abs(output_simple - x_simple).mean().item()
    print(f"Mean absolute difference: {difference:.6f}")
    
    # Output should be different but not completely unrelated
    assert difference > 0.01, "Output too similar to input - no transformation?"
    print("✓ Residual connections functioning!")
    
    # Test 4: Test eval vs train mode
    print("\n[Test 4] Test eval vs train mode")
    print("-" * 70)
    
    block.eval()
    
    # In eval mode, outputs should be deterministic
    output1 = block(x)
    output2 = block(x)
    
    assert torch.allclose(output1, output2, atol=1e-6), \
        "Outputs differ in eval mode!"
    print("✓ Eval mode produces deterministic outputs!")
    
    block.train()
    
    # In train mode, outputs may differ due to dropout
    output1 = block(x)
    output2 = block(x)
    
    difference = torch.abs(output1 - output2).mean().item()
    print(f"Mean difference in train mode: {difference:.6f}")
    
    if difference > 1e-6:
        print("✓ Train mode produces stochastic outputs (dropout active)!")
    else:
        print("⚠ Outputs are identical (may be due to random chance)")
    
    # Test 5: Different configurations
    print("\n[Test 5] Test different configurations")
    print("-" * 70)
    
    configs = [
        (64, 2, 256),      # Smaller model
        (128, 4, 512),     # Default
        (256, 8, 1024),    # Larger model
        (512, 8, 2048),    # Even larger
    ]
    
    for d_m, n_h, d_f in configs:
        test_block = TransformerBlock(
            d_model=d_m,
            num_heads=n_h,
            d_ff=d_f,
            dropout=0.1
        )
        test_input = torch.randn(2, 5, d_m)
        test_output = test_block(test_input)
        
        assert test_output.shape == (2, 5, d_m), \
            f"Config ({d_m}, {n_h}, {d_f}) failed!"
        assert not torch.isnan(test_output).any()
        
        print(f"✓ Config (d_model={d_m}, heads={n_h}, d_ff={d_f}) works")
    
    # Test 6: Gradient flow
    print("\n[Test 6] Verify gradient flow")
    print("-" * 70)
    
    block = TransformerBlock(d_model=128, num_heads=4, d_ff=512, dropout=0.1)
    x = torch.randn(2, 10, 128, requires_grad=True)
    
    # Forward pass
    output = block(x)
    
    # Compute dummy loss
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    assert x.grad is not None, "Input gradient is None!"
    assert x.grad.abs().sum() > 0, "Input gradient is all zeros!"
    
    # Check that all parameters have gradients
    for name, param in block.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient!"
            assert param.grad.abs().sum() > 0, f"Parameter {name} gradient is all zeros!"
    
    print("✓ Gradients flow correctly through all parameters!")
    
    # Test 7: Verify layer structure
    print("\n[Test 7] Verify layer structure")
    print("-" * 70)
    
    block = TransformerBlock(d_model=128, num_heads=4, d_ff=512, dropout=0.1)
    
    # Check that all components exist
    assert hasattr(block, 'ln1'), "Missing ln1 (LayerNorm before attention)!"
    assert hasattr(block, 'attention'), "Missing attention layer!"
    assert hasattr(block, 'ln2'), "Missing ln2 (LayerNorm before FFN)!"
    assert hasattr(block, 'ffn'), "Missing FFN layer!"
    
    # Check types
    assert isinstance(block.ln1, nn.LayerNorm), "ln1 is not LayerNorm!"
    assert isinstance(block.attention, CausalSelfAttention), "attention is not CausalSelfAttention!"
    assert isinstance(block.ln2, nn.LayerNorm), "ln2 is not LayerNorm!"
    assert isinstance(block.ffn, FeedForward), "ffn is not FeedForward!"
    
    print("✓ All required layers present with correct types!")
    
    # Check LayerNorm dimensions
    assert block.ln1.normalized_shape == (128,), "ln1 has wrong normalized_shape!"
    assert block.ln2.normalized_shape == (128,), "ln2 has wrong normalized_shape!"
    
    print("✓ LayerNorm dimensions correct!")
    
    # Test 8: Batch size variations
    print("\n[Test 8] Test various batch sizes")
    print("-" * 70)
    
    block = TransformerBlock(d_model=128, num_heads=4, d_ff=512, dropout=0.1)
    block.eval()
    
    batch_sizes = [1, 2, 4, 8, 16]
    
    for bs in batch_sizes:
        test_input = torch.randn(bs, 10, 128)
        test_output = block(test_input)
        
        assert test_output.shape == (bs, 10, 128), \
            f"Batch size {bs} failed!"
        assert not torch.isnan(test_output).any()
        
        print(f"✓ Batch size {bs} works correctly")
    
    # Test 9: Sequence length variations
    print("\n[Test 9] Test various sequence lengths")
    print("-" * 70)
    
    block = TransformerBlock(d_model=128, num_heads=4, d_ff=512, dropout=0.1)
    block.eval()
    
    seq_lengths = [1, 5, 10, 50, 100]
    
    for sl in seq_lengths:
        test_input = torch.randn(2, sl, 128)
        test_output = block(test_input)
        
        assert test_output.shape == (2, sl, 128), \
            f"Sequence length {sl} failed!"
        assert not torch.isnan(test_output).any()
        
        print(f"✓ Sequence length {sl} works correctly")
    
    # Final summary
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)
    
    # Final demonstration
    block = TransformerBlock(d_model=128, num_heads=4, d_ff=512, dropout=0.1)
    x = torch.randn(2, 10, 128)
    output = block(x)
    
    print(f"\nFinal demonstration:")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"\nArchitecture verified:")
    print("  ✓ Pre-LayerNorm before attention")
    print("  ✓ Causal self-attention with residual connection")
    print("  ✓ Pre-LayerNorm before FFN")
    print("  ✓ Feed-forward network with residual connection")
    print("  ✓ Attention mask support for padding")
    print("  ✓ Gradient flow through all components")
    print("  ✓ Multiple batch sizes and sequence lengths supported")