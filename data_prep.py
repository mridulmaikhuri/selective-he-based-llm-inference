"""
Tiny Shakespeare Data Preparation Script
=========================================
Downloads Tiny Shakespeare dataset, tokenizes with GPT-2 tokenizer,
creates fixed-length sequences, splits into train/val, and saves as .pt files.

Quick run example:
    python data_prep.py --out-dir data/

Quick test with limited sequences:
    python data_prep.py --out-dir data/ --max-seqs 100

This script is idempotent: re-running overwrites existing output files.
"""

import argparse
import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Argument parsing (early, so --help works without heavy imports)
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Tiny Shakespeare dataset for language model training."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/",
        help="Directory to save train_inputs.pt, val_inputs.pt, and tokenizer/ (default: data/)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Fixed sequence length for chunking (default: 256)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of sequences used for training (default: 0.9)",
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=None,
        help="Cap total sequences (useful for quick tests, e.g. --max-seqs 100)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dependency imports (after argparse so --help is always fast)
# ---------------------------------------------------------------------------

def import_dependencies():
    """Import heavy dependencies with friendly error messages."""
    deps = {}
    missing = []

    try:
        import torch
        deps["torch"] = torch
    except ImportError:
        missing.append("torch")

    try:
        from datasets import load_dataset
        deps["load_dataset"] = load_dataset
    except ImportError:
        missing.append("datasets")

    try:
        from transformers import GPT2TokenizerFast
        deps["GPT2TokenizerFast"] = GPT2TokenizerFast
    except ImportError:
        missing.append("transformers")

    if missing:
        print(
            f"[ERROR] Missing required packages: {', '.join(missing)}\n"
            f"Install with:  pip install {' '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    return deps


# ---------------------------------------------------------------------------
# Core steps
# ---------------------------------------------------------------------------

def load_tiny_shakespeare(load_dataset):
    """Download Tiny Shakespeare via direct URL (HF dataset loader deprecated for this dataset)."""
    print("[1/6] Downloading Tiny Shakespeare dataset...")
    try:
        # The HF tiny_shakespeare dataset uses deprecated script format.
        # Fall back to direct download from GitHub
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print("  Using direct download from GitHub (HF datasets deprecated for tiny_shakespeare)...")
        
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                full_text = response.read().decode('utf-8')
            print(f"         Downloaded from GitHub successfully")
        except Exception as url_exc:
            print(f"  GitHub download failed, trying Hugging Face Hub...")
            # Try the HF loader as fallback
            ds = load_dataset("tiny_shakespeare")
            
            # Concatenate all available splits into one long string
            parts = []
            for split_name in ds:
                text = ds[split_name]["text"]
                if isinstance(text, list):
                    parts.extend(text)
                else:
                    parts.append(text)
            
            full_text = "\n".join(parts)
    
    except Exception as exc:
        print(
            f"[ERROR] Failed to download dataset.\n"
            f"  Reason : {exc}\n"
            f"  Check  : internet connection, HF Hub availability.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"         Total characters in corpus: {len(full_text):,}")
    return full_text


def load_tokenizer(GPT2TokenizerFast, out_dir: Path):
    """Load GPT-2 tokenizer, with graceful network-error handling."""
    print("[2/6] Loading GPT-2 tokenizer...")
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    except OSError as exc:
        print(
            f"[ERROR] Could not load GPT-2 tokenizer.\n"
            f"  Reason : {exc}\n"
            f"  Tip    : Ensure internet access or a local HF cache.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"         Vocab size : {tokenizer.vocab_size:,}")

    # Save tokenizer alongside the .pt files for reproducibility
    tok_dir = out_dir / "tokenizer"
    try:
        tok_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(tok_dir))
        print(f"         Tokenizer saved → {tok_dir}")
    except OSError as exc:
        print(f"[ERROR] Could not save tokenizer to {tok_dir}.\n  {exc}", file=sys.stderr)
        sys.exit(1)

    return tokenizer


def tokenize(tokenizer, text: str):
    """Encode the full corpus to a flat list of token ids."""
    print("[3/6] Tokenizing corpus...")
    try:
        token_ids = tokenizer.encode(text)
    except Exception as exc:
        print(f"[ERROR] Tokenization failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"         Total tokens : {len(token_ids):,}")
    return token_ids


def chunk_sequences(token_ids: list, seq_len: int, max_seqs=None):
    """Split flat token list into non-overlapping fixed-length sequences."""
    print(f"[4/6] Chunking into sequences of length {seq_len}...")

    num_complete = len(token_ids) // seq_len          # discard tail remainder
    sequences = [
        token_ids[i * seq_len : (i + 1) * seq_len]
        for i in range(num_complete)
    ]

    if max_seqs is not None and max_seqs < len(sequences):
        sequences = sequences[:max_seqs]
        print(f"         --max-seqs applied: using {len(sequences):,} sequences")
    else:
        print(f"         Total sequences : {len(sequences):,}")

    return sequences


def split_sequences(sequences: list, train_ratio: float):
    """90/10 train/validation split."""
    print(f"[5/6] Splitting (train {train_ratio:.0%} / val {1-train_ratio:.0%})...")

    n_train = max(1, int(len(sequences) * train_ratio))
    train_seqs = sequences[:n_train]
    val_seqs   = sequences[n_train:]

    # Edge-case: if corpus is tiny, ensure val is non-empty
    if not val_seqs:
        val_seqs = train_seqs[-1:]
        train_seqs = train_seqs[:-1]

    print(f"         Train sequences : {len(train_seqs):,}")
    print(f"         Val   sequences : {len(val_seqs):,}")
    return train_seqs, val_seqs


def save_tensors(torch, train_seqs, val_seqs, out_dir: Path):
    """Convert to LongTensors and save as .pt files."""
    print(f"[6/6] Saving tensors to {out_dir} ...")

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[ERROR] Cannot create output directory {out_dir}.\n  {exc}", file=sys.stderr)
        sys.exit(1)

    train_tensor = torch.tensor(train_seqs, dtype=torch.long)  # (N_train, seq_len)
    val_tensor   = torch.tensor(val_seqs,   dtype=torch.long)  # (N_val,   seq_len)

    train_path = out_dir / "train_inputs.pt"
    val_path   = out_dir / "val_inputs.pt"

    try:
        torch.save(train_tensor, train_path)
        torch.save(val_tensor,   val_path)
    except (OSError, RuntimeError) as exc:
        print(f"[ERROR] Failed to write .pt files.\n  {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"         train_inputs.pt → shape {tuple(train_tensor.shape)}")
    print(f"         val_inputs.pt   → shape {tuple(val_tensor.shape)}")
    return train_tensor, val_tensor


def print_stats(train_tensor, val_tensor, tokenizer, seq_len: int):
    """Human-readable summary printed to stdout."""
    total_seqs   = train_tensor.shape[0] + val_tensor.shape[0]
    total_tokens = total_seqs * seq_len

    print()
    print("=" * 54)
    print("  Dataset Statistics")
    print("=" * 54)
    print(f"  seq_len            : {seq_len}")
    print(f"  vocab_size         : {tokenizer.vocab_size:,}")
    print(f"  total tokens saved : {total_tokens:,}")
    print(f"  train sequences    : {train_tensor.shape[0]:,}")
    print(f"  val   sequences    : {val_tensor.shape[0]:,}")
    print(f"  train tensor shape : {tuple(train_tensor.shape)}")
    print(f"  val   tensor shape : {tuple(val_tensor.shape)}")
    print()
    print("  Example — first sequence input IDs (first 16 tokens):")
    first_ids = train_tensor[0, :16].tolist()
    print(f"    {first_ids}")
    decoded = tokenizer.decode(first_ids)
    print(f"  Decoded            : {repr(decoded)}")
    print("=" * 54)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    deps = import_dependencies()

    torch            = deps["torch"]
    load_dataset     = deps["load_dataset"]
    GPT2TokenizerFast = deps["GPT2TokenizerFast"]

    out_dir = Path(args.out_dir)

    # Pipeline
    full_text              = load_tiny_shakespeare(load_dataset)
    tokenizer              = load_tokenizer(GPT2TokenizerFast, out_dir)
    token_ids              = tokenize(tokenizer, full_text)
    sequences              = chunk_sequences(token_ids, args.seq_len, args.max_seqs)
    train_seqs, val_seqs   = split_sequences(sequences, args.train_ratio)
    train_tensor, val_tensor = save_tensors(torch, train_seqs, val_seqs, out_dir)

    print_stats(train_tensor, val_tensor, tokenizer, args.seq_len)
    print("\n[✓] Done — all files written successfully.")


if __name__ == "__main__":
    main()