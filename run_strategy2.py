#!/usr/bin/env python3
"""
Run Strategy 2: Selective HE with encrypted last transformer block and LM head.

This script:
1. Runs plaintext inference on 20 test inputs
2. Runs selective HE inference with last block and LM head encrypted
3. Compares:
   - Top-5 token predictions
   - Logits cosine similarity
   - Latency overhead
   - Encryption/decryption time breakdown
4. Generates side-by-side comparison with Strategy 1
5. Prints analysis summary and comparison table
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict
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


def estimate_perplexity(logits: torch.Tensor) -> float:
    """Estimate perplexity from logits at last position."""
    # logits shape: (batch_size, seq_len, vocab_size)
    last_logits = logits[0, -1, :]  # (vocab_size,)
    
    # Convert logits to probabilities
    log_probs = F.log_softmax(last_logits, dim=-1)
    
    # Perplexity is exp(-log_prob of true token)
    # For a random token, use average log prob
    avg_log_prob = log_probs.mean().item()
    perplexity = np.exp(-avg_log_prob)
    return perplexity


def run_strategy2_benchmark():
    """Run Strategy 2 benchmark on 20 test inputs."""
    
    print("=" * 80)
    print("STRATEGY 2: SELECTIVE HE - LAST BLOCK & LM HEAD ENCRYPTION")
    print("=" * 80)
    
    # Setup
    print("\n[Setup Phase]")
    print("-" * 80)
    
    # Create model
    print("Creating TinyGPT model (4 layers for better Strategy 2 comparison)...")
    model = TinyGPT(
        num_layers=4,
        vocab_size=256,
        d_model=64,
        num_heads=4,
        d_ff=256,
        max_len=128,
        dropout=0.1
    )
    model.eval()
    print("✓ Model created")
    
    # Setup Strategy 2 config (encrypt last block + lm_head)
    print("\nConfiguring Strategy 2 (last block + LM head encryption)...")
    config = SelectiveHEConfig(
        layers_to_encrypt=["blocks.3.attention", "blocks.3.ffn", "lm_head"],
        operations_to_encrypt={
            "blocks.3.attention": ["matmul", "softmax"],
            "blocks.3.ffn": ["matmul", "gelu"],
            "lm_head": ["matmul"]
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
    perplexities_plain = []
    perplexities_he = []
    
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
        
        perp_plain = estimate_perplexity(plain_logits)
        perp_he = estimate_perplexity(he_logits)
        perplexities_plain.append(perp_plain)
        perplexities_he.append(perp_he)
        
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
    
    perp_plain_avg = np.mean(perplexities_plain)
    perp_he_avg = np.mean(perplexities_he)
    perp_diff_pct = ((perp_he_avg - perp_plain_avg) / perp_plain_avg) * 100
    
    print("\n1. LATENCY METRICS")
    print("-" * 80)
    print(f"   Plaintext inference:     {plain_avg:.2f} ms (baseline)")
    print(f"   HE inference (Strategy 2): {he_avg:.2f} ms")
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
    
    print("\n4. PERPLEXITY COMPARISON")
    print("-" * 80)
    print(f"   Plaintext perplexity:    {perp_plain_avg:.4f}")
    print(f"   HE perplexity:           {perp_he_avg:.4f}")
    print(f"   Perplexity change:       {perp_diff_pct:+.2f}%")
    
    print("\n5. PRIVACY & SECURITY")
    print("-" * 80)
    print("   Protected tensors (encrypted):")
    print("     • blocks.3 (last transformer block) - all computations")
    print("     • Last block attention outputs")
    print("     • Last block FFN outputs")
    print("     • Final logits (Language modeling head)")
    print("\n   Unprotected tensors (plaintext):")
    print("     • Input embeddings")
    print("     • First 3 transformer blocks")
    print("     • Intermediate token representations")
    print("     • Layer normalization outputs (first 3 blocks)")
    
    print("\n6. STRATEGY 2 CHARACTERISTICS")
    print("-" * 80)
    print("   ✓ Protects final representations and predictions")
    print("   ✓ Moderate latency overhead (higher than Strategy 1)")
    print("   ✗ Does not protect intermediate transformations")
    print("   ✗ Input embeddings and early blocks visible")
    print(f"   → Best for: Privacy on final predictions with moderate overhead")
    
    print("\n7. ENCRYPTION STATISTICS")
    print("-" * 80)
    print(f"   Number of test inputs:   {len(test_inputs)}")
    print(f"   Average seq length:      {np.mean([x.shape[1] for x in test_inputs]):.1f}")
    print(f"   Vocab size:              {model.vocab_size}")
    print(f"   Model layers encrypted:  1.5 / {model.num_layers} (block + head)")
    
    print("\n" + "=" * 80)
    print("✓ STRATEGY 2 BENCHMARK COMPLETE")
    print("=" * 80 + "\n")
    
    return {
        'plaintext_avg': plain_avg,
        'he_avg': he_avg,
        'latency_overhead_pct': latency_overhead_pct,
        'encryption_pct': enc_pct,
        'top5_overlap_avg': overlap_avg,
        'cosine_similarity_avg': cos_sim_avg,
        'perplexity_plain': perp_plain_avg,
        'perplexity_he': perp_he_avg,
        'encryption_times': encryption_times,
        'he_times': he_times,
        'plaintext_times': plaintext_times
    }


def print_comparison_table(strategy1_metrics: Dict, strategy2_metrics: Dict):
    """Print side-by-side comparison table of Strategy 1 vs Strategy 2."""
    
    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON: STRATEGY 1 vs STRATEGY 2")
    print("=" * 100)
    
    print("\n{:<25} {:<30} {:<30} {:<10}".format(
        "Metric",
        "Strategy 1",
        "Strategy 2",
        "Ratio"
    ))
    print("-" * 100)
    
    # Latency
    s1_lat = strategy1_metrics['he_avg']
    s2_lat = strategy2_metrics['he_avg']
    print("{:<25} {:>28.2f} ms {:>28.2f} ms {:>8.2f}x".format(
        "Average Latency",
        s1_lat,
        s2_lat,
        s2_lat / s1_lat
    ))
    
    # Encryption time
    s1_enc = strategy1_metrics['encryption_pct']
    s2_enc = strategy2_metrics['encryption_pct']
    print("{:<25} {:>27.1f}% {:>27.1f}% {:>8.1f}x".format(
        "Encryption Time %",
        s1_enc,
        s2_enc,
        s2_enc / s1_enc if s1_enc > 0 else 0
    ))
    
    # Top-5 accuracy
    s1_top5 = strategy1_metrics['top5_overlap_avg']
    s2_top5 = strategy2_metrics['top5_overlap_avg']
    print("{:<25} {:>27.2f}/5 {:>27.2f}/5 {:>8.2f}%".format(
        "Top-5 Overlap",
        s1_top5,
        s2_top5,
        (s2_top5 / s1_top5 * 100) if s1_top5 > 0 else 0
    ))
    
    # Cosine similarity
    s1_cos = strategy1_metrics['cosine_similarity_avg']
    s2_cos = strategy2_metrics['cosine_similarity_avg']
    print("{:<25} {:>29.6f} {:>29.6f} {:>8.4f}x".format(
        "Cosine Similarity",
        s1_cos,
        s2_cos,
        s2_cos / s1_cos if s1_cos > 0 else 0
    ))
    
    # Latency overhead
    s1_overhead = strategy1_metrics['latency_overhead_pct']
    s2_overhead = strategy2_metrics['latency_overhead_pct']
    print("{:<25} {:>27.1f}% {:>27.1f}% {:>8.1f}x".format(
        "Latency Overhead",
        s1_overhead,
        s2_overhead,
        s2_overhead / s1_overhead if s1_overhead > 0 else 0
    ))
    
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    
    if s1_lat < s2_lat:
        print(f"✓ Strategy 1 is {s2_lat/s1_lat:.1f}x faster ({s1_lat:.2f} vs {s2_lat:.2f} ms)")
    else:
        print(f"✓ Strategy 2 is {s1_lat/s2_lat:.1f}x faster")
    
    if s2_cos > s1_cos:
        print(f"✓ Strategy 2 has better accuracy (cosine sim: {s2_cos:.6f} vs {s1_cos:.6f})")
    else:
        print(f"✓ Strategy 1 has comparable accuracy (cosine sim: {s1_cos:.6f} vs {s2_cos:.6f})")
    
    print(f"✓ Use Strategy 1 for: Low-latency privacy on attention patterns")
    print(f"✓ Use Strategy 2 for: High-accuracy privacy on final predictions")
    
    print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    # Note: This would require Strategy 1 to be run first to get its metrics
    # For now, we'll just run Strategy 2
    metrics2 = run_strategy2_benchmark()
    
    print("\nNote: To see comparison table, run both strategies together with run_all_strategies.py")
