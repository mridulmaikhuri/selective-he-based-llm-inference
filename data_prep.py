"""
Data Preparation Script

Creates dummy training and validation datasets for testing TinyGPT training.
This is a minimal example - in practice, you'd tokenize real text data.

Example usage:
    python data_prep.py --output data/ --num-tokens 10000
"""

import argparse
import os
from pathlib import Path

import torch


def create_dummy_data(num_tokens, vocab_size=50257):
    """
    Create dummy tokenized data.
    
    Args:
        num_tokens (int): Number of tokens to generate
        vocab_size (int): Maximum vocabulary size
    
    Returns:
        torch.Tensor: Random token IDs (1D tensor of integers)
    """
    # Generate random tokens (avoiding 0 which is used for padding)
    # This should be a 1D tensor of token IDs (integers)
    tokens = torch.randint(1, vocab_size, (num_tokens,), dtype=torch.long)
    
    print(f"  Created tokens with shape: {tokens.shape}")
    print(f"  Data type: {tokens.dtype}")
    print(f"  Min token ID: {tokens.min().item()}, Max token ID: {tokens.max().item()}")
    
    return tokens


def main():
    parser = argparse.ArgumentParser(description="Generate dummy training data")
    parser.add_argument('--output', type=str, default='data/',
                        help='Output directory for data files')
    parser.add_argument('--num-tokens', type=int, default=10000,
                        help='Number of tokens to generate')
    parser.add_argument('--train-split', type=float, default=0.9,
                        help='Fraction of data for training (rest for validation)')
    parser.add_argument('--vocab-size', type=int, default=50257,
                        help='Vocabulary size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Generating dummy training data...")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Total tokens: {args.num_tokens:,}")
    print(f"  Train split: {args.train_split:.1%}")
    print(f"  Vocab size: {args.vocab_size:,}")
    print(f"  Output directory: {args.output}")
    
    # Generate data
    print(f"\nGenerating {args.num_tokens:,} random tokens...")
    data = create_dummy_data(args.num_tokens, args.vocab_size)
    
    # Split into train and validation
    split_idx = int(len(data) * args.train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"  Train tokens: {len(train_data):,}")
    print(f"  Validation tokens: {len(val_data):,}")
    
    # Save to files
    train_path = output_dir / 'train_inputs.pt'
    val_path = output_dir / 'val_inputs.pt'
    
    print(f"\nSaving files...")
    torch.save(train_data, train_path)
    print(f"  ✓ Saved {train_path}")
    
    torch.save(val_data, val_path)
    print(f"  ✓ Saved {val_path}")
    
    print("\n" + "=" * 70)
    print("Data preparation complete!")
    print("=" * 70)
    print(f"\nYou can now train the model with:")
    print(f"  python train.py --data-dir {args.output} --max-steps 10")


if __name__ == "__main__":
    main()