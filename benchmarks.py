#!/usr/bin/env python3
"""
benchmarks.py
=============
Timing, perplexity, and memory profiling utilities for HE-encrypted
and plaintext transformer inference.

Functions
---------
  warmup_runs(model, inputs, method, n_runs)
    Run model forward passes to stabilize performance (warm cache)

  time_inference(model, inputs, method, n_runs, include_warmup)
    Measure inference latency (TTFT, per-token, throughput, stats)

  compute_perplexity(model, val_dataset, method)
    Compute perplexity on validation split

  memory_profiling(model, method)
    Report peak RAM / VRAM usage + ciphertext size estimates

  set_reproducibility_seeds(seed)
    Set seeds for deterministic results

  print_benchmark_summary(results_dict, method)
    Print compact summary of timing results
"""

from __future__ import annotations

import gc
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA/GPU utilities
try:
    import torch.cuda
    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not installed; memory profiling will be limited")

# Try to import HE libraries
try:
    from Pyfhel import Pyfhel
    from he_utils import setup_HE_context
    HE_AVAILABLE = True
except ImportError:
    HE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Reproducibility
# ═══════════════════════════════════════════════════════════════════════════

def set_reproducibility_seeds(seed: int = 42) -> None:
    """
    Set all relevant seeds for reproducible results.

    Parameters
    ----------
    seed : int
        Random seed to use globally.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if CUDA_AVAILABLE:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms may impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Warm-up utilities
# ═══════════════════════════════════════════════════════════════════════════

def warmup_runs(
    model:   nn.Module,
    inputs:  torch.Tensor,
    method:  str = "plain",
    n_runs:  int = 3,
    HE:      Optional[Pyfhel] = None,
    inference_fn: Optional[Callable] = None,
) -> None:
    """
    Run model forward passes to warm up cache and stabilize timing.

    Parameters
    ----------
    model : nn.Module
        PyTorch model in eval mode.
    inputs : torch.Tensor
        Input tensor (typically (1, seq_len) for LM).
    method : str
        Inference method ('plain', 'strategy1', 'strategy2', 'strategy3').
    n_runs : int
        Number of warm-up iterations (default 3).
    HE : Pyfhel, optional
        HE context if using encrypted methods.
    inference_fn : callable, optional
        Custom inference function signature:
          inference_fn(model, inputs, HE, **kwargs) -> (logits, timings, ...)
    """
    model.eval()
    with torch.no_grad():
        for _ in range(n_runs):
            if method == "plain":
                _ = model(inputs)
            elif inference_fn is not None:
                _ = inference_fn(model, inputs, HE)
            else:
                _ = model(inputs)

    if CUDA_AVAILABLE:
        torch.cuda.synchronize()
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Inference timing
# ═══════════════════════════════════════════════════════════════════════════

def time_inference(
    model:        nn.Module,
    inputs:       torch.Tensor,
    method:       str = "plain",
    n_runs:       int = 10,
    include_warmup: bool = True,
    HE:           Optional[Pyfhel] = None,
    inference_fn: Optional[Callable] = None,
    verbose:      bool = True,
) -> Dict[str, Any]:
    """
    Measure inference latency, throughput, and statistics.

    Measures:
      • TTFT (Time to First Token): First output token time
      • Avg latency per token: Total time / sequence length
      • Throughput: Tokens per second
      • Statistics: mean, std, min, max latencies

    Parameters
    ----------
    model : nn.Module
        PyTorch model in eval mode.
    inputs : torch.Tensor
        Input tensor, shape (batch, seq_len) for LM.
    method : str
        Inference method ('plain', 'strategy1', 'strategy2', 'strategy3').
    n_runs : int
        Number of timing runs (default 10).
    include_warmup : bool
        If True, run warm-up iterations first (default True).
    HE : Pyfhel, optional
        HE context if using encrypted methods.
    inference_fn : callable, optional
        Custom inference function; if None, calls model(inputs).
    verbose : bool
        If True, print summary to stdout (default True).

    Returns
    -------
    results : dict
        Keys:
          'ttft_ms'              : Time to first token (ms)
          'latency_per_token_ms' : Avg latency per token (ms)
          'throughput_tokens_sec': Tokens per second
          'mean_latency_ms'      : Mean total latency (ms)
          'std_latency_ms'       : Std dev of latencies (ms)
          'min_latency_ms'       : Min latency (ms)
          'max_latency_ms'       : Max latency (ms)
          'n_runs'               : Number of timing runs
          'seq_len'              : Sequence length of inputs
          'method'               : Inference method used
    """
    model.eval()

    if include_warmup:
        warmup_runs(model, inputs, method, n_runs=3, HE=HE, inference_fn=inference_fn)

    # Sequence length
    if isinstance(inputs, torch.Tensor):
        seq_len = inputs.shape[1] if len(inputs.shape) > 1 else inputs.shape[0]
    else:
        seq_len = 1

    latencies = []

    with torch.no_grad():
        for _ in range(n_runs):
            if CUDA_AVAILABLE:
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            if method == "plain":
                logits = model(inputs)
            elif inference_fn is not None:
                logits, _, _ = inference_fn(model, inputs, HE)
            else:
                logits = model(inputs)

            if CUDA_AVAILABLE:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            latencies.append(elapsed)

    latencies = np.array(latencies) * 1e3  # Convert to ms

    # Compute metrics
    ttft_ms = latencies[0]  # Approximation: full latency as TTFT
    mean_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    min_latency = float(np.min(latencies))
    max_latency = float(np.max(latencies))

    latency_per_token = mean_latency / max(seq_len, 1)
    throughput = 1000.0 / latency_per_token if latency_per_token > 0 else 0.0

    results = {
        "ttft_ms":               ttft_ms,
        "latency_per_token_ms":  latency_per_token,
        "throughput_tokens_sec": throughput,
        "mean_latency_ms":       mean_latency,
        "std_latency_ms":        std_latency,
        "min_latency_ms":        min_latency,
        "max_latency_ms":        max_latency,
        "n_runs":                n_runs,
        "seq_len":               seq_len,
        "method":                method,
    }

    if verbose:
        print_benchmark_summary(results, method)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Perplexity computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_perplexity(
    model:        nn.Module,
    val_dataset:  List[torch.Tensor] | torch.Tensor,
    method:       str = "plain",
    HE:           Optional[Pyfhel] = None,
    inference_fn: Optional[Callable] = None,
    batch_size:   int = 1,
    verbose:      bool = True,
) -> float:
    """
    Compute perplexity on validation dataset.

    Perplexity = exp(mean cross-entropy loss) over all tokens.

    Parameters
    ----------
    model : nn.Module
        Language model in eval mode.
    val_dataset : list of tensors or single tensor
        Validation input IDs. If list, each item is (1, seq_len).
        If single tensor, shape (n_samples, seq_len).
    method : str
        Inference method ('plain', 'strategy1', 'strategy2', 'strategy3').
    HE : Pyfhel, optional
        HE context if using encrypted methods.
    inference_fn : callable, optional
        Custom inference function.
    batch_size : int
        Batch size for evaluation (default 1; HE requires batch_size=1).
    verbose : bool
        If True, print perplexity (default True).

    Returns
    -------
    perplexity : float
        Perplexity score (lower is better).
    """
    model.eval()

    # Convert dataset to list if needed
    if isinstance(val_dataset, torch.Tensor):
        if len(val_dataset.shape) == 2:
            # (n_samples, seq_len) -> list of (1, seq_len)
            val_list = [val_dataset[i:i+1] for i in range(val_dataset.shape[0])]
        else:
            val_list = [val_dataset]
    else:
        val_list = val_dataset

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(val_list):
            if isinstance(batch, torch.Tensor):
                input_ids = batch
            else:
                continue

            # Ensure shape is (batch, seq_len)
            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)

            # Get logits
            if method == "plain":
                logits = model(input_ids)  # (batch, seq_len, vocab_size)
            elif inference_fn is not None:
                logits, _, _ = inference_fn(model, input_ids, HE)
            else:
                logits = model(input_ids)

            # Compute cross-entropy loss (shift: predict next token)
            # input_ids: (batch, seq_len) -> logits: (batch, seq_len, vocab_size)
            # We predict tokens[1:] from tokens[:-1]
            shift_logits = logits[:, :-1, :].contiguous()  # (batch, seq_len-1, vocab_size)
            shift_labels = input_ids[:, 1:].contiguous()   # (batch, seq_len-1)

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                reduction="sum"
            )

            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    # Perplexity = exp(avg_loss)
    if total_tokens == 0:
        perplexity = float('inf')
    else:
        avg_loss = total_loss / total_tokens
        perplexity = float(np.exp(avg_loss))

    if verbose:
        print(f"  Perplexity ({method}): {perplexity:.4f}  (avg_loss={avg_loss:.4f})")

    return perplexity


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Memory profiling
# ═══════════════════════════════════════════════════════════════════════════

def memory_profiling(
    model:       nn.Module,
    method:      str = "plain",
    HE:          Optional[Pyfhel] = None,
    sample_input: Optional[torch.Tensor] = None,
    verbose:     bool = True,
) -> Dict[str, float]:
    """
    Profile peak memory usage (RAM and VRAM if available).

    Estimates:
      • Peak RAM: Process resident set size (if psutil available)
      • Peak VRAM: GPU memory (if CUDA available)
      • Ciphertext size: Approximate HE ciphertext size (for HE methods)

    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    method : str
        Inference method ('plain', 'strategy1', 'strategy2', 'strategy3').
    HE : Pyfhel, optional
        HE context (used to estimate ciphertext size).
    sample_input : torch.Tensor, optional
        Sample input for forward pass (if None, skipped).
    verbose : bool
        If True, print memory summary (default True).

    Returns
    -------
    memory : dict
        Keys:
          'peak_ram_mb'       : Peak RAM usage (MB)
          'peak_vram_mb'      : Peak VRAM usage (MB)
          'model_size_mb'     : Model parameters (MB)
          'ciphertext_size_mb': Estimated ciphertext size (MB)
    """
    results = {
        "peak_ram_mb":       0.0,
        "peak_vram_mb":      0.0,
        "model_size_mb":     0.0,
        "ciphertext_size_mb": 0.0,
    }

    # Model size
    model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    results["model_size_mb"] = model_size_bytes / (1024 ** 2)

    # RAM profiling (if psutil available)
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        gc.collect()
        ram_before = process.memory_info().rss / (1024 ** 2)

        if sample_input is not None:
            with torch.no_grad():
                _ = model(sample_input)

        ram_after = process.memory_info().rss / (1024 ** 2)
        results["peak_ram_mb"] = max(ram_before, ram_after)

    # VRAM profiling (if CUDA available)
    if CUDA_AVAILABLE:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        if sample_input is not None:
            with torch.no_grad():
                _ = model(sample_input.cuda() if not sample_input.is_cuda else sample_input)
            torch.cuda.synchronize()

        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        results["peak_vram_mb"] = peak_vram

    # Ciphertext size estimate (for HE methods)
    if method in ("strategy1", "strategy2", "strategy3") and HE is not None:
        # Rough estimate: ciphertext is ~10x plaintext size in BFV (conservative)
        # Encrypted tensors per method:
        #   Strategy 1: (seq_len, d_model) × 1 layer × n_layers
        #   Strategy 2: (seq_len, d_model) × n_params in last block + lm_head
        #   Strategy 3: (vocab_size, d_model) + (d_model, vocab_size) for embedding & lm_head
        # Very rough: 10x plaintext
        ciphertext_multiplier = 10.0  # Conservative estimate
        results["ciphertext_size_mb"] = results["model_size_mb"] * ciphertext_multiplier

    if verbose:
        print(f"\n  Memory Profiling ({method})")
        print(f"  {'─'*50}")
        print(f"    Model size           : {results['model_size_mb']:>10.2f} MB")
        if results["peak_ram_mb"] > 0:
            print(f"    Peak RAM usage       : {results['peak_ram_mb']:>10.2f} MB")
        if results["peak_vram_mb"] > 0:
            print(f"    Peak VRAM usage      : {results['peak_vram_mb']:>10.2f} MB")
        if results["ciphertext_size_mb"] > 0:
            print(f"    Est. ciphertext size : {results['ciphertext_size_mb']:>10.2f} MB")
        print(f"  {'─'*50}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Reporting utilities
# ═══════════════════════════════════════════════════════════════════════════

def print_benchmark_summary(results: Dict[str, Any], method: str) -> None:
    """
    Print a compact summary of benchmark results.

    Parameters
    ----------
    results : dict
        Output from time_inference() or similar.
    method : str
        Name of the inference method.
    """
    print(f"\n  Benchmark Summary ({method})")
    print(f"  {'─'*60}")
    print(f"    TTFT                : {results.get('ttft_ms', 0):>10.2f} ms")
    print(f"    Latency/token       : {results.get('latency_per_token_ms', 0):>10.2f} ms")
    print(f"    Throughput          : {results.get('throughput_tokens_sec', 0):>10.2f} tok/s")
    print(f"    Mean latency        : {results.get('mean_latency_ms', 0):>10.2f} ms  "
          f"± {results.get('std_latency_ms', 0):.2f}")
    print(f"    Min / Max           : {results.get('min_latency_ms', 0):>10.2f} / "
          f"{results.get('max_latency_ms', 0):>10.2f} ms")
    print(f"    Runs                : {results.get('n_runs', 0):>10} × "
          f"{results.get('seq_len', 0)} tokens")
    print(f"  {'─'*60}")


def compare_benchmarks(
    results_dict: Dict[str, Dict[str, Any]],
    metric: str = "mean_latency_ms",
) -> None:
    """
    Print a side-by-side comparison of benchmark results across methods.

    Parameters
    ----------
    results_dict : dict
        Mapping from method name to results dict (from time_inference).
    metric : str
        Metric to compare (default 'mean_latency_ms').
    """
    if not results_dict:
        return

    print(f"\n  Benchmark Comparison ({metric})")
    print(f"  {'─'*70}")
    print(f"  {'Method':<20}  {metric:<25}  {'Relative':<15}")
    print(f"  {'─'*70}")

    # Find baseline (first method or 'plain')
    baseline_method = next(iter(results_dict.keys()))
    if "plain" in results_dict:
        baseline_method = "plain"
    baseline_value = results_dict[baseline_method].get(metric, 1.0)

    for method, results in results_dict.items():
        value = results.get(metric, 0.0)
        relative = value / baseline_value if baseline_value > 0 else 1.0
        rel_str = f"{relative:.2f}× baseline"
        print(f"  {method:<20}  {value:>24.2f}  {rel_str:>15}")

    print(f"  {'─'*70}")


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Integration helpers
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_all_methods(
    model:        nn.Module,
    inputs:       torch.Tensor,
    val_dataset:  Optional[List[torch.Tensor]] = None,
    methods:      List[str] = None,
    HE:           Optional[Pyfhel] = None,
    inference_fns: Dict[str, Callable] = None,
    n_runs:       int = 10,
) -> Dict[str, Dict[str, Any]]:
    """
    Run benchmarks (timing + perplexity) across all methods.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    inputs : torch.Tensor
        Sample input for timing.
    val_dataset : list of tensors, optional
        Validation data for perplexity.
    methods : list of str, optional
        Methods to benchmark (default: ['plain', 'strategy1', 'strategy2', 'strategy3']).
    HE : Pyfhel, optional
        HE context for encrypted methods.
    inference_fns : dict, optional
        Mapping from method name to inference function.
    n_runs : int
        Timing runs per method (default 10).

    Returns
    -------
    all_results : dict
        Mapping from method → results dict.
    """
    if methods is None:
        methods = ["plain", "strategy1", "strategy2", "strategy3"]
    if inference_fns is None:
        inference_fns = {}

    all_results = {}

    for method in methods:
        print(f"\n  Benchmarking {method} …")
        timing_results = time_inference(
            model, inputs, method=method, n_runs=n_runs,
            HE=HE, inference_fn=inference_fns.get(method),
            verbose=True
        )
        all_results[method] = timing_results

        if val_dataset is not None:
            compute_perplexity(
                model, val_dataset, method=method,
                HE=HE, inference_fn=inference_fns.get(method),
                verbose=True
            )

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# 8.  Example usage (if run directly)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "benchmarks.py — Inference Timing & Profiling" + " " * 9 + "║")
    print("╚" + "═" * 68 + "╝")

    # ── 1. Set reproducibility ────────────────────────────────────────────────
    print("\n[1] Setting reproducibility seeds …")
    set_reproducibility_seeds(42)
    print("    ✓ Seeds set for reproducible results")

    # ── 2. Create a minimal test model ────────────────────────────────────────
    print("\n[2] Creating test model …")
    
    class TinyLM(nn.Module):
        """Minimal language model for demonstration."""
        def __init__(self, vocab_size=128, d_model=32, seq_len=8):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.pos_embed = nn.Embedding(seq_len, d_model)
            self.fc1 = nn.Linear(d_model, 64)
            self.fc2 = nn.Linear(64, vocab_size)
            
        def forward(self, input_ids):
            B, S = input_ids.shape
            pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
            x = self.embed(input_ids) + self.pos_embed(pos)
            x = F.relu(self.fc1(x))
            logits = self.fc2(x)
            return logits
    
    model = TinyLM(vocab_size=128, d_model=32, seq_len=8)
    model.eval()
    print(f"    ✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # ── 3. Create test inputs ─────────────────────────────────────────────────
    print("\n[3] Creating test inputs …")
    test_input = torch.randint(0, 128, (1, 8), dtype=torch.long)
    val_data = [torch.randint(0, 128, (1, 8), dtype=torch.long) for _ in range(5)]
    print(f"    ✓ Test input shape: {test_input.shape}")
    print(f"    ✓ Validation dataset: {len(val_data)} samples")

    # ── 4. Warm-up runs ──────────────────────────────────────────────────────
    print("\n[4] Running warm-up iterations …")
    warmup_runs(model, test_input, method="plain", n_runs=2)
    print("    ✓ Warm-up complete (cache stabilized)")

    # ── 5. Time inference ────────────────────────────────────────────────────
    print("\n[5] Benchmarking inference latency …")
    timing_results = time_inference(
        model, test_input,
        method="plain",
        n_runs=10,
        include_warmup=False,
        verbose=True
    )

    # ── 6. Perplexity computation ────────────────────────────────────────────
    print("\n[6] Computing perplexity on validation data …")
    perplexity = compute_perplexity(
        model, val_data,
        method="plain",
        verbose=True
    )

    # ── 7. Memory profiling ──────────────────────────────────────────────────
    print("\n[7] Profiling memory usage …")
    memory_results = memory_profiling(
        model,
        method="plain",
        sample_input=test_input,
        verbose=True
    )

    # ── 8. Compare benchmarks (simulate multiple methods) ───────────────────
    print("\n[8] Comparing benchmark results across methods …")
    # Create synthetic results for other "methods"
    timing_results_dict = {
        "plain": timing_results,
        "method_2x": {**timing_results, "mean_latency_ms": timing_results["mean_latency_ms"] * 2},
        "method_3x": {**timing_results, "mean_latency_ms": timing_results["mean_latency_ms"] * 3},
    }
    compare_benchmarks(timing_results_dict, metric="mean_latency_ms")

    # ── 9. Function summary ──────────────────────────────────────────────────
    print("\n[9] Available Functions Summary")
    print("╔" + "═" * 68 + "╗")
    functions_info = [
        ("set_reproducibility_seeds(seed)", "Set seeds for reproducible results"),
        ("warmup_runs(model, inputs, method, ...)", "Warm up cache before timing"),
        ("time_inference(model, inputs, method, ...)", "Measure latency & throughput"),
        ("compute_perplexity(model, val_dataset, ...)", "Compute perplexity on validation"),
        ("memory_profiling(model, method, ...)", "Profile RAM/VRAM usage"),
        ("print_benchmark_summary(results, method)", "Print formatted timing summary"),
        ("compare_benchmarks(results_dict, metric)", "Compare across methods"),
        ("benchmark_all_methods(model, inputs, ...)", "Benchmark all methods together"),
    ]
    
    for func_sig, description in functions_info:
        print(f"║  {func_sig:<42}  {description:<24}║")
    
    print("╚" + "═" * 68 + "╝")

    # ── 10. Return values ────────────────────────────────────────────────────
    print("\n[10] Return Value Examples")
    print("╔" + "═" * 68 + "╗")
    print("║  time_inference() returns dict with keys:                         ║")
    print(f"║    {str(list(timing_results.keys())):<65}║")
    print("║                                                                    ║")
    print("║  memory_profiling() returns dict with keys:                       ║")
    print(f"║    {str(list(memory_results.keys())):<65}║")
    print("║                                                                    ║")
    print(f"║  compute_perplexity() returns float: {perplexity:.4f}                     ║")
    print("╚" + "═" * 68 + "╝")

    print("\n✓ All benchmarking functions demonstrated successfully!\n")
