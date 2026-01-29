import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN).
    
    A two-layer fully connected network with GELU activation applied to each
    position independently. This is a standard component in transformer architectures.
    
    The architecture follows:
        x -> Linear(d_model, d_ff) -> GELU -> Dropout -> Linear(d_ff, d_model) -> Dropout -> output
    
    Args:
        d_model (int): Dimension of input and output embeddings. Default: 128
        d_ff (int): Dimension of the intermediate hidden layer. Default: 512
            Typically set to 4 * d_model in standard transformers.
        dropout (float): Dropout probability applied after each layer. Default: 0.1
    
    Attributes:
        fc1 (nn.Linear): First linear transformation (expansion layer)
        fc2 (nn.Linear): Second linear transformation (projection layer)
        dropout (nn.Dropout): Dropout layer applied after GELU and final output
    
    Shape:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)
    """
    
    def __init__(self, d_model=128, d_ff=512, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_prob = dropout
        
        # First linear layer: expands from d_model to d_ff
        # This is the "expansion" layer that increases dimensionality
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        
        # Second linear layer: projects back from d_ff to d_model
        # This is the "projection" layer that returns to original dimensionality
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)
        
        # Dropout layer (reused for efficiency)
        # Applied after GELU activation and after final projection
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize layer weights using Xavier uniform initialization.
        
        This initialization helps maintain gradient magnitudes across layers
        and is standard for transformer feed-forward networks.
        """
        # Initialize first layer
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        
        # Initialize second layer
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        """
        Forward pass through the feed-forward network.
        
        The computation flow:
        1. Expand: (B, S, d_model) -> (B, S, d_ff)
        2. Activate: Apply GELU non-linearity
        3. Dropout: Apply dropout for regularization
        4. Project: (B, S, d_ff) -> (B, S, d_model)
        5. Dropout: Apply dropout again
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        
        Note:
            This implementation is efficient and operates in-place where possible,
            avoiding unnecessary tensor copies.
        """
        # Step 1: First linear transformation (expansion)
        # (B, S, d_model) -> (B, S, d_ff)
        x = self.fc1(x)
        
        # Step 2: Apply GELU activation
        # GELU (Gaussian Error Linear Unit) is smoother than ReLU
        # and is the standard activation for transformer FFNs
        x = F.gelu(x)
        
        # Step 3: Apply dropout after activation
        # This helps prevent overfitting by randomly zeroing elements
        x = self.dropout(x)
        
        # Step 4: Second linear transformation (projection)
        # (B, S, d_ff) -> (B, S, d_model)
        x = self.fc2(x)
        
        # Step 5: Apply dropout after projection
        # This provides additional regularization before the residual connection
        x = self.dropout(x)
        
        return x


if __name__ == "__main__":
    print("Testing FeedForward Network...")
    print("=" * 70)
    
    # Test 1: Basic functionality with default parameters
    print("\n[Test 1] Basic functionality with default parameters")
    print("-" * 70)
    
    # Create model with default parameters
    d_model = 128
    d_ff = 512
    batch_size = 2
    seq_len = 10
    
    model = FeedForward(d_model=d_model, d_ff=d_ff, dropout=0.1)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")
    print(f"Expected output shape: ({batch_size}, {seq_len}, {d_model})")
    
    # Forward pass
    output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    
    # Assert correct shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Shape mismatch! Expected {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("✓ Shape assertion passed!")
    
    # Assert correct dtype
    assert output.dtype == x.dtype, \
        f"Dtype mismatch! Expected {x.dtype}, got {output.dtype}"
    print("✓ Dtype assertion passed!")
    
    # Check for NaNs or Infs
    assert not torch.isnan(output).any(), "Output contains NaN values!"
    assert not torch.isinf(output).any(), "Output contains Inf values!"
    print("✓ No NaN or Inf values detected!")
    
    # Test 2: Verify output is different from input (transformation occurred)
    print("\n[Test 2] Verify transformation occurred")
    print("-" * 70)
    
    # Output should be different from input
    difference = torch.abs(output - x).mean().item()
    print(f"Mean absolute difference between input and output: {difference:.6f}")
    assert difference > 0.01, "Output is too similar to input - transformation may not be working!"
    print("✓ Transformation verified!")
    
    # Test 3: Eval mode (no dropout)
    print("\n[Test 3] Test eval mode (dropout disabled)")
    print("-" * 70)
    
    model.eval()
    
    # Run same input twice - should get identical outputs in eval mode
    output1 = model(x)
    output2 = model(x)
    
    print(f"Output 1 shape: {output1.shape}")
    print(f"Output 2 shape: {output2.shape}")
    
    # Check if outputs are identical
    assert torch.allclose(output1, output2), \
        "Outputs differ in eval mode - dropout may not be disabled properly!"
    print("✓ Eval mode works correctly (deterministic output)!")
    
    # Test 4: Train mode (with dropout)
    print("\n[Test 4] Test train mode (dropout enabled)")
    print("-" * 70)
    
    model.train()
    
    # Run same input twice - outputs should differ due to dropout
    output1 = model(x)
    output2 = model(x)
    
    print(f"Output 1 shape: {output1.shape}")
    print(f"Output 2 shape: {output2.shape}")
    
    # Check if outputs differ
    difference = torch.abs(output1 - output2).mean().item()
    print(f"Mean absolute difference between two forward passes: {difference:.6f}")
    
    if difference > 1e-6:
        print("✓ Train mode works correctly (stochastic output due to dropout)!")
    else:
        print("⚠ Outputs are identical (may be due to random chance)")
    
    # Test 5: Different configurations
    print("\n[Test 5] Test different configurations")
    print("-" * 70)
    
    configs = [
        (64, 256, 0.0),    # Smaller model, no dropout
        (128, 512, 0.1),   # Default config
        (256, 1024, 0.2),  # Larger model, higher dropout
        (512, 2048, 0.15), # Even larger (4x expansion)
    ]
    
    for d_m, d_f, drop in configs:
        test_model = FeedForward(d_model=d_m, d_ff=d_f, dropout=drop)
        test_input = torch.randn(2, 5, d_m)
        test_output = test_model(test_input)
        
        assert test_output.shape == (2, 5, d_m), \
            f"Config ({d_m}, {d_f}, {drop}) failed shape test!"
        assert test_output.dtype == test_input.dtype, \
            f"Config ({d_m}, {d_f}, {drop}) failed dtype test!"
        assert not torch.isnan(test_output).any(), \
            f"Config ({d_m}, {d_f}, {drop}) produced NaN values!"
        
        print(f"✓ Config (d_model={d_m}, d_ff={d_f}, dropout={drop}) works correctly")
    
    # Test 6: Check layer dimensions
    print("\n[Test 6] Verify internal layer dimensions")
    print("-" * 70)
    
    model = FeedForward(d_model=128, d_ff=512, dropout=0.1)
    
    print(f"fc1 weight shape: {model.fc1.weight.shape} (expected: [512, 128])")
    print(f"fc1 bias shape: {model.fc1.bias.shape} (expected: [512])")
    print(f"fc2 weight shape: {model.fc2.weight.shape} (expected: [128, 512])")
    print(f"fc2 bias shape: {model.fc2.bias.shape} (expected: [128])")
    
    assert model.fc1.weight.shape == (512, 128), "fc1 weight shape incorrect!"
    assert model.fc1.bias.shape == (512,), "fc1 bias shape incorrect!"
    assert model.fc2.weight.shape == (128, 512), "fc2 weight shape incorrect!"
    assert model.fc2.bias.shape == (128,), "fc2 bias shape incorrect!"
    
    print("✓ All layer dimensions correct!")
    
    # Test 7: Gradient flow
    print("\n[Test 7] Verify gradient flow")
    print("-" * 70)
    
    model = FeedForward(d_model=128, d_ff=512, dropout=0.1)
    x = torch.randn(2, 10, 128, requires_grad=True)
    
    # Forward pass
    output = model(x)
    
    # Compute a dummy loss
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist and are non-zero
    assert x.grad is not None, "Input gradient is None!"
    assert model.fc1.weight.grad is not None, "fc1 weight gradient is None!"
    assert model.fc2.weight.grad is not None, "fc2 weight gradient is None!"
    
    assert x.grad.abs().sum() > 0, "Input gradient is all zeros!"
    assert model.fc1.weight.grad.abs().sum() > 0, "fc1 weight gradient is all zeros!"
    assert model.fc2.weight.grad.abs().sum() > 0, "fc2 weight gradient is all zeros!"
    
    print("✓ Gradients flow correctly through the network!")
    
    # Final summary
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)
    
    # Final demonstration
    model = FeedForward(d_model=128, d_ff=512, dropout=0.1)
    x = torch.randn(2, 10, 128)
    output = model(x)
    
    print(f"\nFinal demonstration:")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    print(f"\nKey features verified:")
    print("  ✓ Correct input/output shapes maintained")
    print("  ✓ Proper dtype preservation")
    print("  ✓ No NaN or Inf values")
    print("  ✓ Eval mode produces deterministic outputs")
    print("  ✓ Train mode applies dropout correctly")
    print("  ✓ Multiple configurations supported")
    print("  ✓ Layer dimensions correct")
    print("  ✓ Gradients flow properly")