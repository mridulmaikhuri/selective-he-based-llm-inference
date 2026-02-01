#!/usr/bin/env python3
"""
Run Strategy 3: Selective HE with encrypted input embeddings and LM head only.

This strategy employs a unique flow:
1. Encrypt token IDs directly
2. Perform HE embedding lookup (encrypted indices → encrypted embeddings)
3. Decrypt embeddings (plaintext hidden states)
4. Run all transformer blocks in plaintext
5. Encrypt final hidden state before LM head
6. Perform HE LM head computation (encrypted input → encrypted logits)
7. Decrypt logits

This script:
1. Runs plaintext inference on 20 test inputs
2. Runs selective HE inference with embedding and LM head encrypted
3. Compares:
   - Top-5 token predictions
   - Logits cosine similarity
   - Latency overhead
   - Encryption/decryption time breakdown
4. Saves results to CSV
5. Prints privacy vs latency analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict
import time
import csv

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
    """Estimate perplexity based on maximum probability."""
    # Get probability of top token
    last_logits = logits[0, -1, :]  # (vocab_size,)
    probs = F.softmax(last_logits, dim=0)
    max_prob = probs.max().item()
    
    # Avoid log(0)
    if max_prob <= 0:
        return 10000.0
    
    # Perplexity = 1 / max_prob (upper bound)
    perplexity = 1.0 / max_prob
    return perplexity


def print_comparison_table(plain_metrics: Dict, he_metrics: Dict) -> None:
    """Print side-by-side comparison table of plaintext vs HE metrics."""
    print("\n" + "=" * 100)
    print("PLAINTEXT vs HE COMPARISON TABLE")
    print("=" * 100)
    
    comparisons = [
        ("Average Latency (ms)", 
         f"{plain_metrics['plaintext_avg']:.2f}", 
         f"{he_metrics['he_avg']:.2f}"),
        ("Slowdown Factor", 
         "1.0x", 
         f"{he_metrics['he_avg'] / plain_metrics['plaintext_avg']:.1f}x"),
        ("Top-5 Overlap Avg", 
         f"{plain_metrics['top5_overlap_avg']:.2f} / 5", 
         f"{he_metrics['top5_overlap_avg']:.2f} / 5"),
        ("Cosine Similarity", 
         f"{plain_metrics['cosine_similarity_avg']:.4f}", 
         f"{he_metrics['cosine_similarity_avg']:.4f}"),
        ("Accuracy (Top-5 match >= 3)", 
         f"{plain_metrics.get('accuracy_pct', 100.0):.1f}%", 
         f"{he_metrics.get('accuracy_pct', 0.0):.1f}%"),
    ]
    
    print(f"\n{'Metric':<30} {'Plaintext':<25} {'HE Strategy 3':<25}")
    print("-" * 100)
    for metric, plain, he in comparisons:
        print(f"{metric:<30} {plain:>25} {he:>25}")
    print("=" * 100)


def run_strategy3_benchmark():
    """Run Strategy 3 benchmark on 20 test inputs."""
    
    print("=" * 80)
    print("STRATEGY 3: SELECTIVE HE - INPUT EMBEDDING + LM HEAD ENCRYPTION")
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
    
    # Setup Strategy 3 config
    print("\nConfiguring Strategy 3 (embedding + lm_head encryption)...")
    config = SelectiveHEConfig(
        layers_to_encrypt=["embedding", "lm_head"],
        operations_to_encrypt={
            "embedding": ["lookup"],
            "lm_head": ["matmul"]
        },
        encryption_granularity="layer"
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
        perplexities_plain.append(estimate_perplexity(plain_logits))
        
        # HE inference
        he_logits, timings, enc_log = selective_HE_inference(
            model, input_ids, HE, config, verbose=False
        )
        he_times.append(timings['total_time'])
        encryption_times.append(timings['encryption_time'])
        he_compute_times.append(timings['he_compute_time'])
        perplexities_he.append(estimate_perplexity(he_logits))
        
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
    enc_pct = (enc_avg / he_avg) * 100 if he_avg > 0 else 0
    compute_pct = (he_compute_avg / he_avg) * 100 if he_avg > 0 else 0
    
    overlap_avg = np.mean(top5_overlaps)
    overlap_perfect = sum(1 for x in top5_overlaps if x == 5)
    
    cos_sim_avg = np.mean(cosine_similarities)
    cos_sim_min = np.min(cosine_similarities)
    cos_sim_max = np.max(cosine_similarities)
    
    perp_plain_avg = np.mean(perplexities_plain)
    perp_he_avg = np.mean(perplexities_he)
    perp_change_pct = ((perp_he_avg - perp_plain_avg) / perp_plain_avg) * 100
    
    accuracy_pct = (overlap_perfect / len(test_inputs)) * 100
    
    print("\n1. LATENCY METRICS")
    print("-" * 80)
    print(f"   Plaintext inference:     {plain_avg:.2f} ms (baseline)")
    print(f"   HE inference (Strategy 3): {he_avg:.2f} ms")
    print(f"   Latency overhead:        {latency_overhead_pct:+.1f}%")
    print(f"   Slowdown factor:         {he_avg/plain_avg:.1f}x")
    
    print("\n2. ENCRYPTION TIME BREAKDOWN")
    print("-" * 80)
    print(f"   Encryption + Decryption: {enc_avg:.2f} ms ({enc_pct:.1f}% of total)")
    print(f"   HE computation:          {he_compute_avg:.2f} ms ({compute_pct:.1f}% of total)")
    print(f"   Plaintext compute:       {he_avg - enc_avg - he_compute_avg:.2f} ms ({100 - enc_pct - compute_pct:.1f}% of total)")
    
    print("\n3. ACCURACY METRICS")
    print("-" * 80)
    print(f"   Top-5 overlap (average): {overlap_avg:.2f} / 5")
    print(f"   Perfect matches (5/5):   {overlap_perfect} / 20 inputs ({accuracy_pct:.1f}%)")
    print(f"   Cosine similarity (avg): {cos_sim_avg:.6f}")
    print(f"   Cosine similarity range: [{cos_sim_min:.6f}, {cos_sim_max:.6f}]")
    
    print("\n4. PERPLEXITY ANALYSIS")
    print("-" * 80)
    print(f"   Plaintext perplexity:    {perp_plain_avg:.2f}")
    print(f"   HE perplexity:           {perp_he_avg:.2f}")
    print(f"   Perplexity change:       {perp_change_pct:+.2f}%")
    
    print("\n5. PRIVACY & SECURITY")
    print("-" * 80)
    print("   Protected tensors (encrypted):")
    print("     • Input token IDs (before embedding)")
    print("     • Embedding lookups (token→vector conversion)")
    print("     • Final hidden states (before LM head)")
    print("     • LM head computation (hidden→logits)")
    print("     • Output logits")
    print("\n   Unprotected tensors (plaintext):")
    print("     • Embedded token vectors (after decryption)")
    print("     • All transformer block computations")
    print("     • Intermediate attention matrices")
    print("     • FFN activations and outputs")
    
    print("\n6. STRATEGY 3 CHARACTERISTICS")
    print("-" * 80)
    print("   ✓ Protects input tokens (important for privacy)")
    print("   ✓ Protects output logits (prevents eavesdropping)")
    print("   ✗ Transformer blocks are plaintext (significant exposure)")
    print("   ✗ Intermediate representations visible")
    print(f"   → Latency overhead: {he_avg/plain_avg:.1f}x (moderate)")
    print("   → Best for: Input/output privacy with moderate latency tolerance")
    
    print("\n7. ENCRYPTION COVERAGE")
    print("-" * 80)
    print(f"   Encrypted layers: {len(config.layers_to_encrypt)} (embedding, lm_head)")
    print(f"   Total model parameters (approx): {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Encrypted layer types: Input embedding + Output projection")
    
    print("\n" + "=" * 80)
    print("✓ STRATEGY 3 BENCHMARK COMPLETE")
    print("=" * 80 + "\n")
    
    # Return metrics for CSV export
    metrics_dict = {
        'plaintext_avg': plain_avg,
        'he_avg': he_avg,
        'latency_overhead_pct': latency_overhead_pct,
        'encryption_pct': enc_pct,
        'top5_overlap_avg': overlap_avg,
        'cosine_similarity_avg': cos_sim_avg,
        'accuracy_pct': accuracy_pct,
        'perplexity_plain': perp_plain_avg,
        'perplexity_he': perp_he_avg,
        'perplexity_change_pct': perp_change_pct,
        'encryption_times': encryption_times,
        'he_times': he_times,
        'plaintext_times': plaintext_times,
        'top5_overlaps': top5_overlaps,
        'cosine_similarities': cosine_similarities
    }
    
    return metrics_dict


def save_metrics_to_csv(metrics: Dict, filepath: str = "strategy3_results.csv") -> None:
    """Save Strategy 3 metrics to CSV file."""
    
    # Per-input metrics
    per_input_rows = []
    for i in range(len(metrics['he_times'])):
        per_input_rows.append({
            'input_idx': i + 1,
            'plaintext_time_ms': metrics['plaintext_times'][i],
            'he_time_ms': metrics['he_times'][i],
            'slowdown_factor': metrics['he_times'][i] / metrics['plaintext_times'][i] if metrics['plaintext_times'][i] > 0 else 0,
            'top5_overlap': metrics['top5_overlaps'][i],
            'cosine_similarity': metrics['cosine_similarities'][i]
        })
    
    # Summary metrics
    summary_rows = [
        {
            'metric': 'Plaintext Average Latency (ms)',
            'value': f"{metrics['plaintext_avg']:.2f}"
        },
        {
            'metric': 'HE Average Latency (ms)',
            'value': f"{metrics['he_avg']:.2f}"
        },
        {
            'metric': 'Slowdown Factor',
            'value': f"{metrics['he_avg'] / metrics['plaintext_avg']:.1f}x"
        },
        {
            'metric': 'Latency Overhead (%)',
            'value': f"{metrics['latency_overhead_pct']:+.1f}"
        },
        {
            'metric': 'Encryption Time (% of HE)',
            'value': f"{metrics['encryption_pct']:.1f}"
        },
        {
            'metric': 'Top-5 Overlap Average',
            'value': f"{metrics['top5_overlap_avg']:.2f} / 5"
        },
        {
            'metric': 'Cosine Similarity Average',
            'value': f"{metrics['cosine_similarity_avg']:.6f}"
        },
        {
            'metric': 'Accuracy (5/5 matches)',
            'value': f"{metrics['accuracy_pct']:.1f}%"
        },
        {
            'metric': 'Plaintext Perplexity',
            'value': f"{metrics['perplexity_plain']:.2f}"
        },
        {
            'metric': 'HE Perplexity',
            'value': f"{metrics['perplexity_he']:.2f}"
        },
        {
            'metric': 'Perplexity Change (%)',
            'value': f"{metrics['perplexity_change_pct']:+.2f}"
        },
    ]
    
    # Write per-input metrics
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'input_idx', 'plaintext_time_ms', 'he_time_ms', 'slowdown_factor',
            'top5_overlap', 'cosine_similarity'
        ])
        writer.writeheader()
        writer.writerows(per_input_rows)
    
    print(f"✓ Per-input metrics saved to {filepath}")
    
    # Write summary metrics to a separate section
    summary_filepath = filepath.replace('.csv', '_summary.csv')
    with open(summary_filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'value'])
        writer.writeheader()
        writer.writerows(summary_rows)
    
    print(f"✓ Summary metrics saved to {summary_filepath}")


def print_privacy_vs_latency_analysis(metrics: Dict) -> None:
    """Print privacy vs latency tradeoff analysis for Strategy 3."""
    
    print("\n" + "=" * 80)
    print("PRIVACY VS LATENCY TRADEOFF ANALYSIS - STRATEGY 3")
    print("=" * 80)
    
    slowdown = metrics['he_avg'] / metrics['plaintext_avg']
    accuracy = metrics['accuracy_pct']
    privacy_coverage = "Input tokens + Output logits"
    plaintext_coverage = "Embeddings + All transformer blocks (60% of model)"
    
    print("\nPRIVACY PROFILE:")
    print("-" * 80)
    print(f"  Encrypted components: {privacy_coverage}")
    print(f"  Plaintext components: {plaintext_coverage}")
    print(f"  Overall protection: Input/Output encryption only")
    print(f"  Information leakage: Transformer block computations visible")
    
    print("\nLATENCY PROFILE:")
    print("-" * 80)
    print(f"  Baseline (plaintext): {metrics['plaintext_avg']:.2f} ms")
    print(f"  HE inference: {metrics['he_avg']:.2f} ms")
    print(f"  Slowdown: {slowdown:.1f}x")
    print(f"  Acceptable for: Batch inference, offline processing")
    print(f"  NOT suitable for: Real-time interactive systems")
    
    print("\nACCURACY IMPACT:")
    print("-" * 80)
    print(f"  Top-5 match accuracy: {accuracy:.1f}% (5/5 matches)")
    print(f"  Cosine similarity: {metrics['cosine_similarity_avg']:.4f}")
    print(f"  Perplexity change: {metrics['perplexity_change_pct']:+.2f}%")
    
    if accuracy >= 50:
        print(f"  Verdict: HIGH accuracy - suitable for production")
    elif accuracy >= 20:
        print(f"  Verdict: MODERATE accuracy - acceptable for some use cases")
    else:
        print(f"  Verdict: LOW accuracy - requires verification before deployment")
    
    print("\nTRADEOFF ASSESSMENT:")
    print("-" * 80)
    print(f"  Privacy gain: Moderate (protects input/output, not hidden states)")
    print(f"  Latency cost: {slowdown:.1f}x slowdown")
    print(f"  Accuracy loss: {(100 - accuracy):.1f}% of predictions affected")
    
    if slowdown < 2:
        print(f"  Latency impact: ACCEPTABLE for most applications")
    elif slowdown < 10:
        print(f"  Latency impact: MODERATE - suitable for batch processing")
    else:
        print(f"  Latency impact: HIGH - only suitable for offline inference")
    
    print("\nRECOMMENDATIONS:")
    print("-" * 80)
    print("  Use Case 1: Inference-as-a-service with privacy concerns")
    print(f"    → Acceptable if clients tolerate {slowdown:.1f}x latency increase")
    print("  Use Case 2: Batch prediction in privacy-sensitive domains")
    print("    → Ideal - latency less critical, privacy important")
    print("  Use Case 3: Real-time interactive applications")
    print(f"    → NOT recommended ({slowdown:.1f}x too slow)")
    print("  Use Case 4: Privacy-critical genomics/medical inference")
    print("    → Consider Strategy 2 or deeper encryption if accuracy permits")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # Run benchmark
    metrics = run_strategy3_benchmark()
    
    # Print detailed analysis
    print_privacy_vs_latency_analysis(metrics)
    
    # Save to CSV
    save_metrics_to_csv(metrics, "strategy3_results.csv")
    
    # Summary metrics
    print("\nFINAL SUMMARY:")
    print("-" * 80)
    print(f"✓ Plaintext latency:     {metrics['plaintext_avg']:.2f} ms")
    print(f"✓ HE latency:            {metrics['he_avg']:.2f} ms")
    print(f"✓ Slowdown:              {metrics['he_avg']/metrics['plaintext_avg']:.1f}x")
    print(f"✓ Top-5 accuracy:        {metrics['accuracy_pct']:.1f}%")
    print(f"✓ Cosine similarity:     {metrics['cosine_similarity_avg']:.6f}")
    print(f"✓ Results saved to:      strategy3_results.csv & strategy3_results_summary.csv")
    print("-" * 80 + "\n")
