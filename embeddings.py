import torch
import torch.nn as nn
import math

class TokenPositionalEmbedding(nn.Module):
    """
    Combines token embeddings with sinusoidal positional encodings.
    
    This module creates token embeddings and adds deterministic sinusoidal
    positional encodings to inject sequence order information.
    
    Args:
        vocab_size (int): Size of vocabulary. Default: 50257 (GPT-2 vocab size)
        d_model (int): Dimension of embeddings. Default: 128
        max_len (int): Maximum sequence length supported. Default: 1024
        dropout (float): Dropout probability. Default: 0.1
    
    Attributes:
        token_embedding (nn.Embedding): Token embedding layer
        positional_encoding (torch.Tensor): Pre-computed sinusoidal positional encodings
        dropout (nn.Dropout): Dropout layer
    """
    
    def __init__(self, vocab_size=50257, d_model=128, max_len=1024, dropout=0.1):
        super(TokenPositionalEmbedding, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Initialize token embeddings with normal distribution
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Create sinusoidal positional encoding
        self.positional_encoding = self._create_sinusoidal_encoding(max_len, d_model)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)
    
    def _create_sinusoidal_encoding(self, max_len, d_model):
        """
        Create sinusoidal positional encodings as per Vaswani et al. (2017).
        
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        Args:
            max_len (int): Maximum sequence length
            d_model (int): Embedding dimension
        
        Returns:
            torch.Tensor: Positional encoding tensor of shape (1, max_len, d_model)
        """
        # Create position indices [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        
        # Create dimension indices [0, 2, 4, ..., d_model-2]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                            -(math.log(10000.0) / d_model))  # (d_model/2,)
        
        # Initialize positional encoding matrix
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        
        # Apply sin to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer('pe', pe)
        
        return pe
    
    def forward(self, input_ids):
        """
        Forward pass: combine token embeddings with positional encodings.
        
        Args:
            input_ids (torch.LongTensor): Input token IDs of shape (batch_size, seq_len)
        
        Returns:
            torch.Tensor: Combined embeddings of shape (batch_size, seq_len, d_model)
        
        Raises:
            ValueError: If input_ids contain values >= vocab_size
            ValueError: If sequence length exceeds max_len
        """
        # Error checking
        if input_ids.max() >= self.vocab_size:
            raise ValueError(
                f"Input contains token ID {input_ids.max().item()} which is >= "
                f"vocab_size {self.vocab_size}"
            )
        
        batch_size, seq_len = input_ids.shape
        
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_len}"
            )
        
        # Get token embeddings: (batch_size, seq_len, d_model)
        token_emb = self.token_embedding(input_ids)
        
        # Add positional encodings: (batch_size, seq_len, d_model)
        # positional_encoding is (1, max_len, d_model), we slice to seq_len
        pos_emb = self.pe[:, :seq_len, :]
        
        # Combine token and positional embeddings
        embeddings = token_emb + pos_emb
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings


if __name__ == "__main__":
    """
    Test block to verify TokenPositionalEmbedding implementation.
    """
    print("Testing TokenPositionalEmbedding...")
    print("-" * 50)
    
    # Create model instance with default parameters
    model = TokenPositionalEmbedding(
        vocab_size=50257,
        d_model=128,
        max_len=1024,
        dropout=0.1
    )
    
    # Create dummy input: batch_size=2, seq_len=10
    input_ids = torch.randint(0, 50257, (2, 10))
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Input IDs:\n{input_ids}")
    print()
    
    # Forward pass
    output = model(input_ids)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: torch.Size([2, 10, 128])")
    print()
    
    # Verify shape
    assert output.shape == torch.Size([2, 10, 128]), "Output shape mismatch!"
    print("✓ Shape test passed!")
    print()
    
    # Additional tests
    print("Additional validation tests:")
    print("-" * 50)
    
    # Test 1: Different sequence length
    input_ids_short = torch.randint(0, 50257, (1, 5))
    output_short = model(input_ids_short)
    print(f"✓ Short sequence (1, 5) -> {output_short.shape}")
    
    # Test 2: Maximum length
    input_ids_long = torch.randint(0, 50257, (1, 1024))
    output_long = model(input_ids_long)
    print(f"✓ Max length sequence (1, 1024) -> {output_long.shape}")
    
    # Test 3: Error checking - vocab overflow
    try:
        bad_input = torch.tensor([[50257]])  # Exactly at vocab_size
        model(bad_input)
        print("✗ Vocab overflow check failed!")
    except ValueError as e:
        print(f"✓ Vocab overflow correctly caught: {str(e)[:50]}...")
    
    # Test 4: Error checking - sequence too long
    try:
        bad_input = torch.randint(0, 50257, (1, 1025))  # Exceeds max_len
        model(bad_input)
        print("✗ Sequence length check failed!")
    except ValueError as e:
        print(f"✓ Sequence length correctly caught: {str(e)[:50]}...")
    
    print()
    print("=" * 50)
    print("All tests passed successfully!")
    print(f"Final output shape: {output.shape}")