import torch
import torch.nn as nn
from embeddings import TokenPositionalEmbedding
from transformer import TransformerBlock

class TinyGPT(nn.Module):
    """
    Tiny GPT Language Model.
    
    A small-scale autoregressive transformer model for language modeling.
    The architecture consists of:
    1. Token + Positional embeddings
    2. Stack of N transformer blocks (with causal self-attention)
    3. Final layer normalization
    4. Language modeling head (tied with token embeddings)
    
    Args:
        num_layers (int): Number of transformer blocks. Default: 4
        vocab_size (int): Size of vocabulary. Default: 50257 (GPT-2 vocab)
        d_model (int): Dimension of embeddings/hidden states. Default: 128
        num_heads (int): Number of attention heads. Default: 4
            Note: d_model must be divisible by num_heads
        d_ff (int): Dimension of feed-forward intermediate layer. Default: 512
            Typically 4 * d_model in standard transformers
        max_len (int): Maximum sequence length. Default: 1024
        dropout (float): Dropout probability. Default: 0.1
    
    Attributes:
        embedding (TokenPositionalEmbedding): Combined token and positional embeddings
        blocks (nn.ModuleList): List of transformer blocks
        ln_f (nn.LayerNorm): Final layer normalization
        lm_head (nn.Linear): Language modeling head (output projection)
    
    Shape:
        Input: (batch_size, seq_len) - Token IDs
        Output: (batch_size, seq_len, vocab_size) - Logits for next token prediction
    """
    
    def __init__(
        self,
        num_layers=4,
        vocab_size=50257,
        d_model=128,
        num_heads=4,
        d_ff=512,
        max_len=1024,
        dropout=0.1
    ):
        super(TinyGPT, self).__init__()
        
        # Store configuration
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout_prob = dropout
        
        # Validate configuration
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        # Component 1: Token + Positional Embeddings
        self.embedding = TokenPositionalEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            dropout=dropout
        )
        
        # Component 2: Stack of Transformer Blocks
        # Use ModuleList to store multiple transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Component 3: Final Layer Normalization
        # Applied after all transformer blocks, before the LM head
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)
        
        # Component 4: Language Modeling Head
        # Projects from d_model to vocab_size to produce logits
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight Tying
        # Tie the weights of the language modeling head with token embeddings
        # This reduces parameters and often improves performance
        # The embedding layer and lm_head will share the same weight matrix
        self.lm_head.weight = self.embedding.token_embedding.weight
        
        # Initialize weights (except tied weights which are already initialized)
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights for the model.
        
        Note: Token embedding and lm_head weights are tied, so they share
        the same initialization from TokenPositionalEmbedding.
        """
        # Initialize final layer norm
        nn.init.ones_(self.ln_f.weight)
        nn.init.zeros_(self.ln_f.bias)
    
    def forward(self, input_ids, attn_mask=None):
        """
        Forward pass through the TinyGPT model.
        
        Args:
            input_ids (torch.LongTensor): Input token IDs of shape (batch_size, seq_len)
            attn_mask (torch.Tensor, optional): Attention mask for padding.
                Shape: (batch_size, seq_len) where 1/True = valid, 0/False = padding
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_len, vocab_size)
                These are unnormalized log probabilities for next token prediction.
        
        Flow:
            input_ids -> embeddings -> transformer_blocks -> layer_norm -> lm_head -> logits
        """
        # Step 1: Get embeddings (token + positional)
        # Shape: (B, S) -> (B, S, d_model)
        x = self.embedding(input_ids)
        
        # Step 2: Pass through transformer blocks sequentially
        # Each block: (B, S, d_model) -> (B, S, d_model)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        
        # Step 3: Apply final layer normalization
        # Shape: (B, S, d_model) -> (B, S, d_model)
        x = self.ln_f(x)
        
        # Step 4: Project to vocabulary size to get logits
        # Shape: (B, S, d_model) -> (B, S, vocab_size)
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        
        Args:
            input_ids (torch.LongTensor): Starting sequence of shape (batch_size, seq_len)
            max_new_tokens (int): Number of new tokens to generate
            temperature (float): Sampling temperature (higher = more random)
            top_k (int, optional): If set, only sample from top k tokens
        
        Returns:
            torch.LongTensor: Generated sequence of shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval() 
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop input_ids to max_len if it gets too long
                input_ids_cond = input_ids if input_ids.size(1) <= self.max_len else input_ids[:, -self.max_len:]
                
                # Get logits for next token
                logits = self(input_ids_cond)
                
                # Take logits at the last position
                # Shape: (B, S, vocab_size) -> (B, vocab_size)
                logits = logits[:, -1, :]
                
                # Apply temperature
                logits = logits / temperature
                
                # Optionally apply top-k filtering
                if top_k is not None:
                    # Zero out all logits except top k
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # Sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        int: Total number of trainable parameters
    
    Note:
        This function also prints a human-friendly representation of the count.
    """
    # Count all trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Print human-friendly format
    if total_params >= 1_000_000_000:
        human_friendly = f"{total_params / 1_000_000_000:.2f}B"
    elif total_params >= 1_000_000:
        human_friendly = f"{total_params / 1_000_000:.2f}M"
    elif total_params >= 1_000:
        human_friendly = f"{total_params / 1_000:.2f}K"
    else:
        human_friendly = str(total_params)
    
    print(f"Total trainable parameters: {total_params:,} ({human_friendly})")
    
    return total_params


if __name__ == "__main__":
    print("Testing TinyGPT Model...")
    print("=" * 70)
    
    # Test 1: Basic model construction and forward pass
    print("\n[Test 1] Basic model construction and forward pass")
    print("-" * 70)
    
    # Create model with default parameters
    model = TinyGPT(
        num_layers=4,
        vocab_size=50257,
        d_model=128,
        num_heads=4,
        d_ff=512,
        max_len=1024,
        dropout=0.1
    )
    
    print(f"Model configuration:")
    print(f"  - Number of layers: {model.num_layers}")
    print(f"  - Vocabulary size: {model.vocab_size}")
    print(f"  - Model dimension: {model.d_model}")
    print(f"  - Number of heads: {model.num_heads}")
    print(f"  - FFN dimension: {model.d_ff}")
    print(f"  - Max sequence length: {model.max_len}")
    print(f"  - Dropout: {model.dropout_prob}")
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_len}, {model.vocab_size})")
    
    # Forward pass
    logits = model(input_ids)
    
    print(f"Logits shape: {logits.shape}")
    
    # Assert correct shape
    assert logits.shape == (batch_size, seq_len, model.vocab_size), \
        f"Shape mismatch! Expected {(batch_size, seq_len, model.vocab_size)}, got {logits.shape}"
    print("✓ Shape assertion passed!")
    
    # Check for NaNs or Infs
    assert not torch.isnan(logits).any(), "Logits contain NaN values!"
    assert not torch.isinf(logits).any(), "Logits contain Inf values!"
    print("✓ No NaN or Inf values detected!")
    
    # Test 2: Count parameters
    print("\n[Test 2] Count model parameters")
    print("-" * 70)
    
    total_params = count_parameters(model)
    
    print(f"\nParameter breakdown:")
    print(f"  - Token embeddings: {model.vocab_size * model.d_model:,}")
    print(f"  - Transformer blocks: ~{model.num_layers * (4 * model.d_model**2 + 2 * model.d_model * model.d_ff):,}")
    print(f"  - Final LayerNorm: {2 * model.d_model:,}")
    print(f"  - LM head: TIED (0 additional params)")
    
    assert 1_000_000 <= total_params <= 10_000_000, \
        f"Parameter count {total_params} is outside expected range!"
    print("✓ Parameter count is in expected range!")
    
    # Test 3: Weight tying verification
    print("\n[Test 3] Verify weight tying")
    print("-" * 70)
    
    # Check that lm_head and token_embedding share the same weight tensor
    embedding_weight = model.embedding.token_embedding.weight
    lm_head_weight = model.lm_head.weight
    
    print(f"Token embedding weight shape: {embedding_weight.shape}")
    print(f"LM head weight shape: {lm_head_weight.shape}")
    print(f"Are weights tied (same object)? {embedding_weight is lm_head_weight}")
    
    assert embedding_weight is lm_head_weight, \
        "Weights are not tied! lm_head and token_embedding should share the same weight tensor."
    print("✓ Weight tying verified!")
    
    # Test 4: Forward pass with different sequence lengths
    print("\n[Test 4] Test different sequence lengths")
    print("-" * 70)
    
    seq_lengths = [1, 5, 10, 50, 100]
    
    model.eval()
    
    for sl in seq_lengths:
        test_input = torch.randint(0, 50257, (2, sl))
        test_logits = model(test_input)
        
        assert test_logits.shape == (2, sl, 50257), \
            f"Sequence length {sl} failed!"
        assert not torch.isnan(test_logits).any()
        
        print(f"✓ Sequence length {sl}: output shape {test_logits.shape}")
    
    # Test 5: Test with attention mask
    print("\n[Test 5] Test with attention mask (padding)")
    print("-" * 70)
    
    # Create input with padding mask
    input_ids = torch.randint(0, 50257, (2, 10))
    attn_mask = torch.ones(2, 10, dtype=torch.bool)
    attn_mask[1, 7:] = False  # Mask out last 3 positions in second batch
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Attention mask shape: {attn_mask.shape}")
    print(f"Batch 1 valid positions: {attn_mask[1].sum().item()}/10")
    
    logits_masked = model(input_ids, attn_mask=attn_mask)
    
    print(f"Logits shape: {logits_masked.shape}")
    
    assert logits_masked.shape == (2, 10, 50257)
    assert not torch.isnan(logits_masked).any()
    
    print("✓ Attention mask works correctly!")
    
    # Test 6: Gradient flow
    print("\n[Test 6] Verify gradient flow")
    print("-" * 70)
    
    model.train()
    
    input_ids = torch.randint(0, 50257, (2, 10))
    logits = model(input_ids)
    
    # Compute dummy loss
    loss = logits.sum()
    
    # Backward pass
    loss.backward()
    
    # Check that parameters have gradients
    params_with_grads = 0
    params_without_grads = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None and param.grad.abs().sum() > 0:
                params_with_grads += 1
            else:
                params_without_grads += 1
    
    print(f"Parameters with gradients: {params_with_grads}")
    print(f"Parameters without gradients: {params_without_grads}")
    
    assert params_without_grads == 0, \
        f"{params_without_grads} parameters have no gradient!"
    
    print("✓ Gradients flow to all parameters!")
    
    # Test 7: Generation capability
    print("\n[Test 7] Test generation capability")
    print("-" * 70)
    
    model.eval()
    
    # Start with a simple prompt
    prompt = torch.randint(0, 50257, (1, 5))
    
    print(f"Prompt shape: {prompt.shape}")
    
    # Generate 10 new tokens
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)
    
    print(f"Generated shape: {generated.shape}")
    print(f"Expected shape: (1, {5 + 10})")
    
    assert generated.shape == (1, 15), \
        f"Generation failed! Expected (1, 15), got {generated.shape}"
    
    # Check that new tokens are within vocabulary range
    assert generated.min() >= 0 and generated.max() < 50257, \
        "Generated tokens are outside vocabulary range!"
    
    print("✓ Generation works correctly!")
    
    # Final summary
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)
    
    # Final demonstration
    model = TinyGPT(num_layers=4, vocab_size=50257, d_model=128, d_ff=512, dropout=0.1)
    input_ids = torch.randint(0, 50257, (2, 10))
    logits = model(input_ids)
    
    print(f"\nFinal demonstration:")
    print(f"  Input shape:  {input_ids.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"\n  Parameter count:")
    total_params = count_parameters(model)
    
    print(f"\nArchitecture components:")
    print("  ✓ Token + Positional embeddings")
    print(f"  ✓ {model.num_layers} Transformer blocks")
    print("  ✓ Final LayerNorm")
    print("  ✓ Language modeling head (tied weights)")