#!/usr/bin/env python3
"""
Run Strategy 1: Selective HE with encrypted attention outputs only.

This script:
1. Runs plaintext inference on 20 test inputs
2. Runs selective HE inference with only attention outputs encrypted
3. Compares:
   - Top-5 token predictions
   - Logits cosine similarity
   - Latency overhead
   - Encryption/decryption time breakdown
4. Prints analysis summary
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
import time

from model import TinyGPT
from selective_he_config import SelectiveHEConfig
from he_utils import setup_HE_context
from selective_he_engine import selective_HE_inference


def get_top5_tokens(logits: torch.Tensor) -> List[int]:
    """Extract top-5 token predictions from logits (last position)."""
    # logits shape: (batch_size, seq_len, vocab_size)
    last_logits = logits[0, -1, :]  # (vocab_size,)
    top5_indices = torch.topk(last_logits, k=5).indices.tolist()
    return top5_indices


def compute_cosine_similarity(logits1: torch.Tensor, logits2: torch.Tensor) -> float:
    """Compute cosine similarity between two logit distributions."""
    # Flatten and normalize
    logits1_flat = logits1.flatten()
    logits2_flat = logits2.flatten()
    
    # Cosine similarity
    cos_sim = F.cosine_similarity(
        logits1_flat.unsqueeze(0),
        logits2_flat.unsqueeze(0)
    ).item()
    return cos_sim


def compute_top5_overlap(predictions1: List[int], predictions2: List[int]) -> int:
    """Count overlap in top-5 predictions."""
    set1 = set(predictions1)
    set2 = set(predictions2)
    overlap = len(set1.intersection(set2))
    return overlap


def run_strategy1_benchmark():
    """Run Strategy 1 benchmark on 20 test inputs."""
    
    print("=" * 80)
    print("STRATEGY 1: SELECTIVE HE - ATTENTION OUTPUT ENCRYPTION ONLY")
    print("=" * 80)
    
    # Setup
    print("\n[Setup Phase]")
    print("-" * 80)
    
    # Create model
    print("Creating TinyGPT model...")
    model = TinyGPT(
        num_layers=2,
        vocab_size=256,
        d_model=64,
        num_heads=4,
        d_ff=256,
        max_len=128,
        dropout=0.1
    )
    model.eval()
    print("✓ Model created")
    
    # Setup Strategy 1 config
    print("\nConfiguring Strategy 1 (attention output encryption)...")
    config = SelectiveHEConfig(
        layers_to_encrypt=["blocks.0.attention"],  # Only first block attention
        operations_to_encrypt={
            "blocks.0.attention": ["matmul", "softmax"]
        },
        encryption_granularity="operation"
    )
    config.validate(model)
    print("✓ Configuration valid")
    print(f"  Encrypted layers: {config.layers_to_encrypt}")
    
    # Setup HE context
    print("\nInitializing HE context...")
    HE = setup_HE_context(n=2**14, t=65537)
    print("✓ HE context ready")
    
    # Generate test inputs
    print("\nGenerating 20 test inputs...")
    np.random.seed(42)
    test_inputs = []
    for i in range(20):
        seq_len = np.random.randint(3, 8)
        input_ids = torch.randint(10, 250, (1, seq_len), dtype=torch.long)
        test_inputs.append(input_ids)
    print(f"✓ Generated {len(test_inputs)} test inputs")
    
    # Run benchmarks
    print("\n" + "=" * 80)
    print("[Benchmark Phase]")
    print("=" * 80)
    
    plaintext_times = []
    he_times = []
    encryption_times = []
    he_compute_times = []
    top5_overlaps = []
    cosine_similarities = []
    
    print("\nRunning inference on test inputs...\n")
    
    for i, input_ids in enumerate(test_inputs):
        # Plaintext inference
        start = time.perf_counter()
        with torch.no_grad():
            plain_logits = model(input_ids)
        plain_time = (time.perf_counter() - start) * 1000
        plaintext_times.append(plain_time)
        
        # HE inference
        he_logits, timings, enc_log = selective_HE_inference(
            model, input_ids, HE, config, verbose=False
        )
        he_times.append(timings['total_time'])
        encryption_times.append(timings['encryption_time'])
        he_compute_times.append(timings['he_compute_time'])
        
        # Compute metrics
        top5_plain = get_top5_tokens(plain_logits)
        top5_he = get_top5_tokens(he_logits)
        overlap = compute_top5_overlap(top5_plain, top5_he)
        top5_overlaps.append(overlap)
        
        cos_sim = compute_cosine_similarity(plain_logits, he_logits)
        cosine_similarities.append(cos_sim)
        
        # Print progress
        status = "✓" if overlap >= 3 else "⚠"
        print(f"  [{i+1:2d}] {status} Top-5 overlap: {overlap}/5 | "
              f"Cosine sim: {cos_sim:.4f} | "
              f"HE time: {timings['total_time']:8.2f} ms")
    
    # Analysis
    print("\n" + "=" * 80)
    print("[Analysis Summary]")
    print("=" * 80)
    
    # Aggregate statistics
    plain_avg = np.mean(plaintext_times)
    he_avg = np.mean(he_times)
    enc_avg = np.mean(encryption_times)
    he_compute_avg = np.mean(he_compute_times)
    
    latency_overhead_pct = ((he_avg - plain_avg) / plain_avg) * 100
    enc_pct = (enc_avg / he_avg) * 100
    compute_pct = (he_compute_avg / he_avg) * 100
    
    overlap_avg = np.mean(top5_overlaps)
    overlap_perfect = sum(1 for x in top5_overlaps if x == 5)
    
    cos_sim_avg = np.mean(cosine_similarities)
    cos_sim_min = np.min(cosine_similarities)
    cos_sim_max = np.max(cosine_similarities)
    
    print("\n1. LATENCY METRICS")
    print("-" * 80)
    print(f"   Plaintext inference:     {plain_avg:.2f} ms (baseline)")
    print(f"   HE inference (Strategy 1): {he_avg:.2f} ms")
    print(f"   Latency overhead:        {latency_overhead_pct:+.1f}%")
    print(f"   Slowdown factor:         {he_avg/plain_avg:.1f}x")
    
    print("\n2. ENCRYPTION TIME BREAKDOWN")
    print("-" * 80)
    print(f"   Encryption + Decryption: {enc_avg:.2f} ms ({enc_pct:.1f}% of total)")
    print(f"   HE computation:          {he_compute_avg:.2f} ms ({compute_pct:.1f}% of total)")
    print(f"   Plaintext layers:        {he_avg - enc_avg - he_compute_avg:.2f} ms")
    
    print("\n3. ACCURACY METRICS")
    print("-" * 80)
    print(f"   Top-5 overlap (average): {overlap_avg:.2f} / 5")
    print(f"   Perfect matches (5/5):   {overlap_perfect} / 20 inputs")
    print(f"   Cosine similarity (avg): {cos_sim_avg:.6f}")
    print(f"   Cosine similarity range: [{cos_sim_min:.6f}, {cos_sim_max:.6f}]")
    
    print("\n4. PRIVACY & SECURITY")
    print("-" * 80)
    print("   Protected tensors (encrypted):")
    print("     • blocks.0.attention outputs (self-attention computation)")
    print("     • Attention matrices (Q·K^T / softmax operations)")
    print("     • Sensitive patterns learned in first attention block")
    print("\n   Unprotected tensors (plaintext):")
    print("     • Input embeddings")
    print("     • Q, K, V projections (input side)")
    print("     • Attention output projections (after decryption)")
    print("     • All FFN layers")
    print("     • Language modeling head")
    print("     • Logits")
    
    print("\n5. STRATEGY 1 CHARACTERISTICS")
    print("-" * 80)
    print("   ✓ Minimal computational overhead (low slowdown)")
    print("   ✓ Protects core attention mechanism in first block")
    print("   ✗ Does not protect FFN or LM head outputs")
    print("   ✗ Full input/output sequences visible in plaintext")
    print(f"   → Best for: Privacy on attention patterns with low latency impact")
    
    print("\n6. ENCRYPTION STATISTICS")
    print("-" * 80)
    print(f"   Number of test inputs:   {len(test_inputs)}")
    print(f"   Average seq length:      {np.mean([x.shape[1] for x in test_inputs]):.1f}")
    print(f"   Vocab size:              {model.vocab_size}")
    print(f"   Model layers encrypted:  1 / {model.num_layers}")
    
    print("\n" + "=" * 80)
    print("✓ STRATEGY 1 BENCHMARK COMPLETE")
    print("=" * 80 + "\n")
    
    return {
        'plaintext_avg': plain_avg,
        'he_avg': he_avg,
        'latency_overhead_pct': latency_overhead_pct,
        'encryption_pct': enc_pct,
        'top5_overlap_avg': overlap_avg,
        'cosine_similarity_avg': cos_sim_avg,
        'encryption_times': encryption_times,
        'he_times': he_times,
        'plaintext_times': plaintext_times
    }


if __name__ == "__main__":
    metrics = run_strategy1_benchmark()
