import argparse
import os
import sys
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast

def download_dataset(max_retries=3):
    """Download Tiny Shakespeare dataset with retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"Downloading Tiny Shakespeare dataset (attempt {attempt + 1}/{max_retries})...")
            import urllib.request
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            print(f"Downloading from {url}...")
            
            with urllib.request.urlopen(url) as response:
                text = response.read().decode('utf-8')
            
            dataset = {"train": {"text": [text]}}
            print("✓ Dataset downloaded successfully")
            return dataset
        except Exception as e:
            print(f"✗ Download attempt {attempt + 1} failed: {e}", file=sys.stderr)
            if attempt == max_retries - 1:
                print("Error: Failed to download dataset after multiple attempts", file=sys.stderr)
                sys.exit(1)
    return None


def load_tokenizer(max_retries=3):
    """Load GPT-2 tokenizer with retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"Loading GPT-2 tokenizer (attempt {attempt + 1}/{max_retries})...")
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            print(f"✓ Tokenizer loaded successfully (vocab_size={tokenizer.vocab_size})")
            return tokenizer
        except Exception as e:
            print(f"✗ Tokenizer load attempt {attempt + 1} failed: {e}", file=sys.stderr)
            if attempt == max_retries - 1:
                print("Error: Failed to load tokenizer after multiple attempts", file=sys.stderr)
                sys.exit(1)
    return None


def prepare_data(args):
    """Main data preparation pipeline."""
    
    # Create output directory
    out_dir = Path(args.out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Output directory: {out_dir.absolute()}")
    except Exception as e:
        print(f"Error: Failed to create output directory {out_dir}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Download dataset
    dataset = download_dataset()
    
    # Load tokenizer
    tokenizer = load_tokenizer()
    
    # Concatenate all text
    print("Concatenating dataset text...")
    try:
        text = "".join(dataset["train"]["text"])
        print(f"✓ Total characters: {len(text):,}")
    except Exception as e:
        print(f"Error: Failed to concatenate text: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Tokenize
    print("Tokenizing text...")
    try:
        tokens = tokenizer.encode(text)
        print(f"✓ Total tokens: {len(tokens):,}")
    except Exception as e:
        print(f"Error: Failed to tokenize text: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create fixed-length sequences (non-overlapping chunks)
    seq_len = args.seq_len
    print(f"Creating sequences of length {seq_len}...")
    
    # Calculate number of complete sequences
    num_sequences = len(tokens) // seq_len
    
    # Truncate to fit complete sequences
    tokens = tokens[:num_sequences * seq_len]
    
    # Reshape into sequences
    try:
        sequences = torch.LongTensor(tokens).view(num_sequences, seq_len)
        print(f"✓ Created {num_sequences:,} sequences")
    except Exception as e:
        print(f"Error: Failed to create sequences: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Apply max_seqs limit if specified
    if args.max_seqs is not None and args.max_seqs < num_sequences:
        print(f"Limiting to {args.max_seqs} sequences for testing...")
        sequences = sequences[:args.max_seqs]
        num_sequences = args.max_seqs
    
    # Split into train/val (90/10)
    split_idx = int(num_sequences * 0.9)
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    
    print(f"✓ Train sequences: {len(train_sequences):,}")
    print(f"✓ Validation sequences: {len(val_sequences):,}")
    
    # Save sequences
    train_path = out_dir / "train_inputs.pt"
    val_path = out_dir / "val_inputs.pt"
    
    try:
        print(f"Saving train sequences to {train_path}...")
        torch.save(train_sequences, train_path)
        print(f"✓ Saved {train_path}")
        
        print(f"Saving validation sequences to {val_path}...")
        torch.save(val_sequences, val_path)
        print(f"✓ Saved {val_path}")
    except Exception as e:
        print(f"Error: Failed to save sequences: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Save tokenizer
    tokenizer_dir = out_dir / "tokenizer"
    try:
        print(f"Saving tokenizer to {tokenizer_dir}...")
        tokenizer.save_pretrained(tokenizer_dir)
        print(f"✓ Saved tokenizer to {tokenizer_dir}")
    except Exception as e:
        print(f"Error: Failed to save tokenizer: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Print statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Sequence length (seq_len):        {seq_len}")
    print(f"Tokenizer vocabulary size:        {tokenizer.vocab_size:,}")
    print(f"Total tokens processed:           {len(tokens):,}")
    print(f"Train sequences:                  {len(train_sequences):,}")
    print(f"Validation sequences:             {len(val_sequences):,}")
    print(f"Total sequences:                  {num_sequences:,}")
    print(f"Train tokens:                     {len(train_sequences) * seq_len:,}")
    print(f"Validation tokens:                {len(val_sequences) * seq_len:,}")
    print("="*60)
    
    # Print example sequence
    print("\nEXAMPLE: First training sequence (input IDs)")
    print("-"*60)
    example_ids = train_sequences[0].tolist()
    print(f"Token IDs (first 20): {example_ids[:20]}")
    print(f"Decoded text (first 100 chars):")
    example_text = tokenizer.decode(example_ids)
    print(f"{example_text[:100]}...")
    print("-"*60)
    
    print(f"\n✓ Data preparation complete! Output saved to {out_dir.absolute()}")
    print(f"  - {train_path.name}: shape {tuple(train_sequences.shape)}")
    print(f"  - {val_path.name}: shape {tuple(val_sequences.shape)}")
    print(f"  - tokenizer/: GPT-2 tokenizer files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data",
        help="Output directory for saved files (default: data)"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Sequence length for chunking (default: 256)"
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=None,
        help="Maximum number of sequences to generate (for testing, default: None = all)"
    )
    args = parser.parse_args()
    
    # Validate arguments
    if args.seq_len <= 0:
        print("Error: --seq-len must be positive", file=sys.stderr)
        sys.exit(1)
    
    if args.max_seqs is not None and args.max_seqs <= 0:
        print("Error: --max-seqs must be positive if specified", file=sys.stderr)
        sys.exit(1)
    
    # Run data preparation
    try:
        prepare_data(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()