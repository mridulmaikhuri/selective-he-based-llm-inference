#!/usr/bin/env python3
"""
Utility script to run all three strategies and compare them.

Usage:
    python run_all_strategies.py

Produces comparison metrics for:
- Strategy 1: Minimal encryption (attention block 0 only)
- Strategy 2: Balanced encryption (attention + FFN in blocks 0-1)
- Strategy 3: Maximum encryption (all blocks + LM head)
"""

import torch
import numpy as np
from typing import Dict, List
import time

from model import TinyGPT
from selective_he_config import load_strategy_config
from he_utils import setup_HE_context
from selective_he_engine import selective_HE_inference


def run_strategy_test(strategy_num: int, num_inputs: int = 5) -> Dict:
    """Run a single strategy benchmark."""
    
    print(f"\n{'='*80}")
    print(f"STRATEGY {strategy_num} BENCHMARK ({num_inputs} inputs)")
    print(f"{'='*80}")
    
    # Setup
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
    
    config = load_strategy_config(strategy_num)
    config.validate(model)
    
    HE = setup_HE_context(n=2**14, t=65537)
    
    # Generate test inputs
    np.random.seed(strategy_num)  # Different seed per strategy
    test_inputs = [
        torch.randint(10, 250, (1, np.random.randint(3, 8)), dtype=torch.long)
        for _ in range(num_inputs)
    ]
    
    # Run inference
    plaintext_times = []
    he_times = []
    encryption_times = []
    he_compute_times = []
    
    print("\nRunning inference...")
    for i, input_ids in enumerate(test_inputs):
        # Plaintext
        with torch.no_grad():
            start = time.perf_counter()
            _ = model(input_ids)
            plain_time = (time.perf_counter() - start) * 1000
            plaintext_times.append(plain_time)
        
        # HE
        _, timings, _ = selective_HE_inference(
            model, input_ids, HE, config, verbose=False
        )
        he_times.append(timings['total_time'])
        encryption_times.append(timings['encryption_time'])
        he_compute_times.append(timings['he_compute_time'])
        
        print(f"  [{i+1:2d}] HE: {timings['total_time']:8.2f} ms | "
              f"Plain: {plain_time:6.2f} ms | "
              f"Encrypt: {timings['encryption_time']:7.2f} ms")
    
    # Aggregate
    plain_avg = np.mean(plaintext_times)
    he_avg = np.mean(he_times)
    enc_avg = np.mean(encryption_times)
    compute_avg = np.mean(he_compute_times)
    
    result = {
        'strategy': strategy_num,
        'plaintext_ms': plain_avg,
        'he_ms': he_avg,
        'encryption_ms': enc_avg,
        'compute_ms': compute_avg,
        'slowdown': he_avg / plain_avg if plain_avg > 0 else 1.0,
        'overhead_pct': ((he_avg - plain_avg) / plain_avg * 100) if plain_avg > 0 else 0,
        'encryption_pct': (enc_avg / he_avg * 100) if he_avg > 0 else 0,
    }
    
    return result


def print_comparison(results: List[Dict]):
    """Print comparison table."""
    
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Metric':<30} {'Strategy 1':<20} {'Strategy 2':<20} {'Strategy 3':<20}")
    print(f"{'-'*30} {'-'*20} {'-'*20} {'-'*20}")
    
    # Latency
    latencies = [f"{r['he_ms']:.2f} ms" for r in results]
    plaintext = results[0]['plaintext_ms']
    print(f"{'Latency (HE)':<30} {latencies[0]:<20} {latencies[1]:<20} {latencies[2]:<20}")
    print(f"{'Plaintext baseline':<30} {plaintext:.2f} ms")
    
    # Slowdown
    slowdowns = [f"{r['slowdown']:.1f}x" for r in results]
    print(f"{'Slowdown factor':<30} {slowdowns[0]:<20} {slowdowns[1]:<20} {slowdowns[2]:<20}")
    
    # Overhead
    overheads = [f"{r['overhead_pct']:+.1f}%" for r in results]
    print(f"{'Latency overhead':<30} {overheads[0]:<20} {overheads[1]:<20} {overheads[2]:<20}")
    
    print(f"\n{'Encryption component breakdown':<30}")
    print(f"{'-'*30}")
    
    # Encryption time
    enc_times = [f"{r['encryption_ms']:.2f} ms" for r in results]
    print(f"{'Encryption+Decrypt time':<30} {enc_times[0]:<20} {enc_times[1]:<20} {enc_times[2]:<20}")
    
    # Encryption percentage
    enc_pcts = [f"{r['encryption_pct']:.1f}%" for r in results]
    print(f"{'Encryption % of total':<30} {enc_pcts[0]:<20} {enc_pcts[1]:<20} {enc_pcts[2]:<20}")
    
    # HE compute
    compute_times = [f"{r['compute_ms']:.2f} ms" for r in results]
    print(f"{'HE compute time':<30} {compute_times[0]:<20} {compute_times[1]:<20} {compute_times[2]:<20}")
    
    print(f"\n{'Security vs Performance Tradeoff':<30}")
    print(f"{'-'*30}")
    
    security_levels = [
        "Low (only attention)",
        "Medium (attention + FFN)",
        "High (all layers + head)"
    ]
    for i, level in enumerate(security_levels):
        print(f"Strategy {i+1}: {level:<45} → {slowdowns[i]} slower")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    print("""
Strategy 1 (Attention Only):
  ✓ Minimal latency overhead
  ✓ Fast encryption/decryption
  ✗ Limited privacy protection
  → Use when: Speed critical, attention patterns are lowest priority

Strategy 2 (Attention + FFN):
  ✓ Balanced security/performance
  ✓ Protects core computations
  ✗ Still visible embeddings and logits
  → Use when: Moderate privacy requirements with acceptable slowdown

Strategy 3 (All Layers + Head):
  ✓ Maximum privacy protection
  ✓ Protects all sensitive layers
  ✗ Significant latency overhead
  → Use when: Privacy critical, latency is secondary concern
""")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SELECTIVE HE STRATEGY COMPARISON - 5 INPUTS EACH")
    print("="*80)
    
    # Run all strategies
    results = []
    for strategy in [1, 2, 3]:
        try:
            result = run_strategy_test(strategy, num_inputs=5)
            results.append(result)
        except Exception as e:
            print(f"\n✗ Strategy {strategy} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comparison
    if len(results) == 3:
        print_comparison(results)
    else:
        print(f"\n✗ Only {len(results)}/3 strategies completed")
