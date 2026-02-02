"""
Benchmarking Module for TinyGPT with HE Inference Methods

Provides comprehensive benchmarking utilities for:
1. Inference timing (latency, throughput, TTFT)
2. Perplexity computation
3. Memory profiling (RAM and VRAM)

Supports multiple inference methods:
- 'plain': Standard PyTorch inference
- 'strategy1': Attention only encryption
- 'strategy2': Attention + FFN encryption
- 'strategy3': Full model encryption (input + output)

Key Features:
- Warm-up runs for stable timing measurements
- CUDA synchronization for accurate GPU timing
- Per-token latency breakdown
- Peak memory tracking
- Ciphertext size estimation for HE methods
- Full reproducibility support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import psutil
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from contextlib import contextmanager

from model import TinyGPT
from selective_he_config import load_strategy_config
from he_utils import setup_HE_context
from selective_he_engine import selective_HE_inference


@dataclass
class TimingStats:
    """Container for timing statistics."""
    mean: float
    std: float
    min: float
    max: float
    median: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'median': self.median
        }


@dataclass
class MemoryStats:
    """Container for memory statistics."""
    peak_ram_mb: float
    peak_vram_mb: float
    ciphertext_size_mb: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'peak_ram_mb': self.peak_ram_mb,
            'peak_vram_mb': self.peak_vram_mb,
            'ciphertext_size_mb': self.ciphertext_size_mb
        }


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@contextmanager
def torch_device_context(model: nn.Module):
    """Context manager to ensure model is on the correct device."""
    device = next(model.parameters()).device
    try:
        yield device
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def warmup_runs(
    model: TinyGPT,
    inputs: torch.Tensor,
    method: str = 'plain',
    n_warmup: int = 3
) -> None:
    """
    Run warmup iterations to stabilize timing measurements.
    
    Warmup helps stabilize GPU clocks and memory allocation patterns.
    
    Args:
        model: TinyGPT model instance
        inputs: Input token IDs tensor of shape (batch_size, seq_len)
        method: Inference method ('plain', 'strategy1', 'strategy2', 'strategy3')
        n_warmup: Number of warmup iterations
    """
    model.eval()
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    
    with torch.no_grad():
        for _ in range(n_warmup):
            if method == 'plain':
                _ = model(inputs)
            elif method in ['strategy1', 'strategy2', 'strategy3']:
                strategy_num = int(method[-1])
                config = load_strategy_config(strategy_num)
                HE = setup_HE_context()
                _, _, _ = selective_HE_inference(model, inputs, HE, config)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()


def time_inference(
    model: TinyGPT,
    inputs: torch.Tensor,
    method: str = 'plain',
    n_runs: int = 10,
    warmup: bool = True
) -> Dict[str, Any]:
    """
    Benchmark inference timing with detailed metrics.
    
    Measures:
    - TTFT (Time To First Token): Latency for first token prediction
    - Average latency per token
    - Throughput: tokens processed per second
    - Statistical summary (mean, std, min, max, median)
    
    Args:
        model: TinyGPT model instance
        inputs: Input token IDs tensor of shape (batch_size, seq_len)
        method: Inference method ('plain', 'strategy1', 'strategy2', 'strategy3')
        n_runs: Number of inference runs for averaging
        warmup: Whether to run warmup iterations first
    
    Returns:
        Dictionary containing:
        - 'ttft_ms': Time to first token in milliseconds
        - 'avg_latency_per_token_ms': Average latency per token
        - 'throughput_tokens_per_sec': Tokens processed per second
        - 'stats': TimingStats object with mean, std, min, max, median
        - 'latencies_ms': List of all latency measurements
        - 'method': Method used for inference
    """
    set_seed()
    
    # Ensure inputs are 2D (batch_size, seq_len)
    if inputs.dim() == 1:
        inputs = inputs.unsqueeze(0)
    
    if warmup:
        warmup_runs(model, inputs, method, n_warmup=3)
    
    model.eval()
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    batch_size, seq_len = inputs.shape
    
    latencies = []
    
    with torch.no_grad():
        for _ in range(n_runs):
            # Synchronize before timing if CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            # Run inference based on method
            if method == 'plain':
                logits = model(inputs)
            elif method in ['strategy1', 'strategy2', 'strategy3']:
                strategy_num = int(method[-1])
                config = load_strategy_config(strategy_num)
                HE = setup_HE_context()
                logits, _, _ = selective_HE_inference(model, inputs, HE, config)
            else:
                raise ValueError(f"Unknown inference method: {method}")
            
            # Synchronize after computation if CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
    
    # Convert to numpy array for statistics
    latencies_array = np.array(latencies)
    
    # Calculate statistics
    stats = TimingStats(
        mean=float(np.mean(latencies_array)),
        std=float(np.std(latencies_array)),
        min=float(np.min(latencies_array)),
        max=float(np.max(latencies_array)),
        median=float(np.median(latencies_array))
    )
    
    # Calculate TTFT (first token latency) - using first run as approximation
    ttft_ms = latencies[0]
    
    # Average latency per token
    avg_latency_per_token = stats.mean / seq_len
    
    # Throughput: tokens per second
    throughput = (batch_size * seq_len) / (stats.mean / 1000)
    
    result = {
        'ttft_ms': ttft_ms,
        'avg_latency_per_token_ms': avg_latency_per_token,
        'throughput_tokens_per_sec': throughput,
        'stats': stats,
        'latencies_ms': latencies,
        'method': method
    }
    
    # Print compact summary
    print(f"\n{'='*70}")
    print(f"INFERENCE TIMING SUMMARY ({method.upper()})")
    print(f"{'='*70}")
    print(f"TTFT (ms):                    {ttft_ms:.2f}")
    print(f"Avg Latency per Token (ms):   {avg_latency_per_token:.2f}")
    print(f"Throughput (tokens/sec):      {throughput:.1f}")
    print(f"Mean Latency (ms):            {stats.mean:.2f} ± {stats.std:.2f}")
    print(f"Min/Max Latency (ms):         {stats.min:.2f} / {stats.max:.2f}")
    print(f"{'='*70}\n")
    
    return result


def compute_perplexity(
    model: TinyGPT,
    val_dataset: torch.Tensor,
    method: str = 'plain',
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    stride: int = 1
) -> float:
    """
    Compute perplexity on validation dataset.
    
    Perplexity measures how well the model predicts the next token.
    Lower perplexity indicates better predictions.
    
    Formula: PPL = exp(mean(-log(p(y_i))))
    
    Args:
        model: TinyGPT model instance
        val_dataset: Validation token IDs of shape (seq_len,) or (batch_size, seq_len)
        method: Inference method ('plain', 'strategy1', 'strategy2', 'strategy3')
        batch_size: Batch size for inference
        max_samples: Maximum number of samples to evaluate (None = all)
        stride: Stride for sliding window (e.g., stride=2 for every 2nd window)
    
    Returns:
        Perplexity value (float)
    """
    set_seed()
    model.eval()
    device = next(model.parameters()).device
    
    # Handle 1D input
    if val_dataset.dim() == 1:
        val_dataset = val_dataset.unsqueeze(0)
    
    val_dataset = val_dataset.to(device)
    
    losses = []
    total_samples = 0
    
    with torch.no_grad():
        # Create sliding windows
        seq_len = val_dataset.shape[1]
        window_size = min(seq_len - 1, 128)  # Use reasonable window size
        
        for start_idx in range(0, seq_len - window_size, stride):
            if max_samples and total_samples >= max_samples:
                break
            
            end_idx = start_idx + window_size
            input_ids = val_dataset[:, start_idx:end_idx]
            target_ids = val_dataset[:, start_idx + 1:end_idx + 1]
            
            # Run inference
            if method == 'plain':
                logits = model(input_ids)
            elif method in ['strategy1', 'strategy2', 'strategy3']:
                strategy_num = int(method[-1])
                config = load_strategy_config(strategy_num)
                HE = setup_HE_context()
                logits, _, _ = selective_HE_inference(model, input_ids, HE, config)
            else:
                raise ValueError(f"Unknown inference method: {method}")
            
            # Compute loss: cross-entropy between predicted and target
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = target_ids.reshape(-1)
            
            loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean')
            losses.append(loss.item())
            total_samples += 1
    
    if not losses:
        return 10000.0
    
    # Perplexity = exp(mean loss)
    mean_loss = np.mean(losses)
    perplexity = np.exp(mean_loss)
    
    print(f"\n{'='*70}")
    print(f"PERPLEXITY SUMMARY ({method.upper()})")
    print(f"{'='*70}")
    print(f"Mean Loss:                    {mean_loss:.4f}")
    print(f"Perplexity:                   {perplexity:.2f}")
    print(f"Samples Evaluated:            {total_samples}")
    print(f"{'='*70}\n")
    
    return perplexity


def memory_profiling(
    model: TinyGPT,
    inputs: torch.Tensor,
    method: str = 'plain',
    n_runs: int = 3
) -> MemoryStats:
    """
    Profile peak memory usage during inference.
    
    Measures:
    - Peak RAM (system memory)
    - Peak VRAM (GPU memory if available)
    - Estimated ciphertext size for HE methods
    
    Args:
        model: TinyGPT model instance
        inputs: Input token IDs tensor of shape (batch_size, seq_len)
        method: Inference method ('plain', 'strategy1', 'strategy2', 'strategy3')
        n_runs: Number of runs to average over
    
    Returns:
        MemoryStats object containing peak memory measurements
    """
    set_seed()
    model.eval()
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    
    peak_ram_mb = 0.0
    peak_vram_mb = 0.0
    ciphertext_size_mb = 0.0
    
    # Get process for RAM monitoring
    process = psutil.Process(os.getpid())
    
    with torch.no_grad():
        for _ in range(n_runs):
            # Reset memory stats if CUDA
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            
            # Get initial RAM
            initial_ram_mb = process.memory_info().rss / (1024 * 1024)
            
            # Run inference
            if method == 'plain':
                _ = model(inputs)
            elif method in ['strategy1', 'strategy2', 'strategy3']:
                strategy_num = int(method[-1])
                config = load_strategy_config(strategy_num)
                HE = setup_HE_context()
                _, _, _ = selective_HE_inference(model, inputs, HE, config)
            else:
                raise ValueError(f"Unknown inference method: {method}")
            
            # Synchronize and measure
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                current_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                peak_vram_mb = max(peak_vram_mb, current_vram_mb)
            
            # Get peak RAM for this run
            current_ram_mb = process.memory_info().rss / (1024 * 1024)
            ram_delta = current_ram_mb - initial_ram_mb
            peak_ram_mb = max(peak_ram_mb, ram_delta)
    
    # Estimate ciphertext size for HE methods
    if method in ['strategy1', 'strategy2', 'strategy3']:
        ciphertext_size_mb = _estimate_ciphertext_size(model, inputs.shape, method)
    
    stats = MemoryStats(
        peak_ram_mb=peak_ram_mb,
        peak_vram_mb=peak_vram_mb,
        ciphertext_size_mb=ciphertext_size_mb
    )
    
    print(f"\n{'='*70}")
    print(f"MEMORY PROFILING SUMMARY ({method.upper()})")
    print(f"{'='*70}")
    print(f"Peak RAM (MB):                {stats.peak_ram_mb:.2f}")
    print(f"Peak VRAM (MB):               {stats.peak_vram_mb:.2f}")
    if ciphertext_size_mb > 0:
        print(f"Estimated Ciphertext (MB):    {stats.ciphertext_size_mb:.2f}")
    print(f"{'='*70}\n")
    
    return stats


def _estimate_ciphertext_size(
    model: TinyGPT,
    input_shape: Tuple[int, ...],
    method: str
) -> float:
    """
    Estimate ciphertext size for HE methods.
    
    BFV ciphertexts are typically 2-4x larger than plaintext depending on
    polynomial degree and scheme parameters.
    
    Args:
        model: TinyGPT model instance
        input_shape: Shape of input tensor
        method: Inference method
    
    Returns:
        Estimated ciphertext size in MB
    """
    # BFV parameters (from he_utils.py defaults)
    n = 2**14  # polynomial degree
    t = 65537  # plaintext modulus
    
    # Number of ciphertexts needed to encrypt a tensor
    # Each ciphertext can hold up to n values
    batch_size, seq_len = input_shape
    
    # Hidden dimension from model
    d_model = model.d_model
    
    # Estimate encrypted tensors size based on strategy
    if method == 'strategy1':
        # Encrypt attention outputs only
        encrypted_elements = batch_size * seq_len * d_model
    elif method == 'strategy2':
        # Encrypt attention + FFN outputs
        encrypted_elements = 2 * batch_size * seq_len * d_model
    elif method == 'strategy3':
        # Encrypt input and output
        encrypted_elements = (batch_size * seq_len * d_model) + (batch_size * seq_len * model.vocab_size)
    else:
        return 0.0
    
    # Number of ciphertexts
    n_ciphertexts = int(np.ceil(encrypted_elements / n))
    
    # Size per ciphertext in BFV: approximately (n * log2(t)) bits
    bits_per_ciphertext = n * int(np.log2(t))
    bytes_per_ciphertext = bits_per_ciphertext / 8
    
    # Total ciphertext size
    total_bytes = n_ciphertexts * bytes_per_ciphertext
    total_mb = total_bytes / (1024 * 1024)
    
    return total_mb


def run_full_benchmark(
    model: TinyGPT,
    val_dataset: torch.Tensor,
    test_inputs: torch.Tensor,
    methods: Optional[List[str]] = None,
    n_timing_runs: int = 10
) -> Dict[str, Dict[str, Any]]:
    """
    Run complete benchmarking suite across all methods.
    
    Args:
        model: TinyGPT model instance
        val_dataset: Validation dataset for perplexity computation
        test_inputs: Test inputs for timing and memory profiling
        methods: List of methods to benchmark (default: all strategies)
        n_timing_runs: Number of runs for timing benchmarks
    
    Returns:
        Dictionary mapping method names to benchmark results
    """
    if methods is None:
        methods = ['plain', 'strategy1', 'strategy2', 'strategy3']
    
    results = {}
    
    for method in methods:
        print(f"\n{'#'*70}")
        print(f"# BENCHMARKING: {method.upper()}")
        print(f"{'#'*70}\n")
        
        method_results = {
            'timing': time_inference(model, test_inputs, method, n_runs=n_timing_runs),
            'perplexity': compute_perplexity(model, val_dataset, method),
            'memory': memory_profiling(model, test_inputs, method).to_dict()
        }
        
        results[method] = method_results
    
    return results


def print_benchmark_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Print summary comparison across all benchmarked methods.
    
    Args:
        results: Dictionary from run_full_benchmark
    """
    print(f"\n{'='*100}")
    print(f"COMPLETE BENCHMARK SUMMARY")
    print(f"{'='*100}\n")
    
    # Print timing comparison
    print("TIMING (ms per inference):")
    print("-" * 100)
    print(f"{'Method':<15} {'TTFT':<12} {'Avg/Token':<15} {'Mean':<12} {'Std':<12} {'Throughput':<15}")
    print("-" * 100)
    
    for method, data in results.items():
        timing = data['timing']
        stats = timing['stats']
        print(f"{method:<15} {timing['ttft_ms']:<12.2f} {timing['avg_latency_per_token_ms']:<15.2f} "
              f"{stats.mean:<12.2f} {stats.std:<12.2f} {timing['throughput_tokens_per_sec']:<15.1f}")
    
    print("\n" + "PERPLEXITY:")
    print("-" * 100)
    print(f"{'Method':<15} {'Perplexity':<15}")
    print("-" * 100)
    
    for method, data in results.items():
        print(f"{method:<15} {data['perplexity']:<15.2f}")
    
    print("\n" + "MEMORY:")
    print("-" * 100)
    print(f"{'Method':<15} {'Peak RAM (MB)':<18} {'Peak VRAM (MB)':<18} {'Est. Ciphertext (MB)':<20}")
    print("-" * 100)
    
    for method, data in results.items():
        memory = data['memory']
        ciphertext = memory.get('ciphertext_size_mb', 0.0)
        print(f"{method:<15} {memory['peak_ram_mb']:<18.2f} {memory['peak_vram_mb']:<18.2f} "
              f"{ciphertext:<20.2f}")
    
    print(f"\n{'='*100}\n")


if __name__ == '__main__':
    """
    Example usage of benchmarking utilities.
    """
    # Load or create model
    model = TinyGPT(
        num_layers=4,
        vocab_size=50257,
        d_model=128,
        num_heads=4,
        d_ff=512,
        max_len=1024,
        dropout=0.1
    )
    
    # Load checkpoint if available
    try:
        checkpoint = torch.load('checkpoint.pt')
        # Handle training checkpoint format (contains model_state_dict, etc.)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✓ Loaded model from checkpoint.pt (training format)")
            except RuntimeError as load_err:
                # Checkpoint might have different config (e.g., different max_len)
                print(f"⚠ Checkpoint incompatible with model config: {str(load_err)[:80]}...")
                print("⚠ Using random initialization instead")
        else:
            model.load_state_dict(checkpoint)
            print("✓ Loaded model from checkpoint.pt")
    except FileNotFoundError:
        print("⚠ No checkpoint found, using random initialization")
    except Exception as e:
        print(f"⚠ Error loading checkpoint: {str(e)[:80]}...")
        print("⚠ Using random initialization instead")
    
    # Load validation data
    try:
        val_inputs = torch.load('data/val_inputs.pt')
        train_inputs = torch.load('data/train_inputs.pt')
        # Ensure data is 2D (batch_size, seq_len) and respects max_len
        if val_inputs.dim() == 1:
            # If 1D, reshape into (num_samples, seq_len) with seq_len=64
            val_inputs = val_inputs[:640].reshape(-1, 64)  # 10 samples of 64 tokens
        else:
            # If 2D, ensure seq_len doesn't exceed model's max_len (1024)
            if val_inputs.shape[1] > 1024:
                val_inputs = val_inputs[:, :1024]
        
        if train_inputs.dim() == 1:
            train_inputs = train_inputs[:640].reshape(-1, 64)
        else:
            if train_inputs.shape[1] > 1024:
                train_inputs = train_inputs[:, :1024]
        
        test_inputs = val_inputs[:1]  # Single sample for testing
    except FileNotFoundError:
        print("No data files found, creating dummy data")
        test_inputs = torch.randint(1, 50257, (1, 64), dtype=torch.long)
        val_inputs = torch.randint(1, 50257, (10, 64), dtype=torch.long)
    
    # Run full benchmark
    results = run_full_benchmark(
        model=model,
        val_dataset=val_inputs,
        test_inputs=test_inputs,
        methods=['plain'],  # Start with plain for testing
        n_timing_runs=5
    )
    
    # Print summary
    print_benchmark_summary(results)
