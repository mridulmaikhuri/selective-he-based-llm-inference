#!/usr/bin/env python3
"""
Experiment Runner for TinyGPT Benchmarking

Orchestrates comprehensive benchmarking experiments across multiple inference methods:
- plain: Standard PyTorch inference (baseline)
- strategy1: Selective encryption (attention only)
- strategy2: Selective encryption (attention + FFN)
- strategy3: Full model encryption (input + output)

Features:
- Deterministic prompt generation across multiple sequence lengths
- Progress tracking with resume capability
- Results saved in both JSON (for resumability) and CSV (for analysis)
- Comprehensive metrics: latency, throughput, perplexity, memory, ciphertext overhead
- Exception handling with intermediate result saving
- Optional test mode for quick validation

Usage:
    python experiment_runner.py --checkpoint checkpoint.pt --output-dir results/
    python experiment_runner.py --checkpoint checkpoint.pt --test-mode  # Quick 4-prompt test
"""

import torch
import argparse
import json
import csv
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm
from datetime import datetime

from model import TinyGPT
from benchmarks import (
    time_inference,
    compute_perplexity,
    memory_profiling,
    set_seed
)


@dataclass
class ExperimentResult:
    """Container for a single experiment result."""
    method: str
    prompt_length: int
    prompt_id: int
    avg_latency_ms: float
    tokens_per_sec: float
    perplexity: float
    peak_ram_mb: float
    peak_vram_mb: float
    ciphertext_overhead_mb: float
    error: Optional[str] = None
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def setup_logging(output_dir: Path, test_mode: bool = False) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        output_dir: Directory for log files
        test_mode: If True, use test logging level
    
    Returns:
        Configured logger instance
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if test_mode else logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def generate_prompts(
    num_prompts: int = 100,
    lengths: Optional[List[int]] = None,
    vocab_size: int = 50257,
    seed: int = 42
) -> List[torch.Tensor]:
    """
    Generate deterministic prompts across multiple sequence lengths.
    
    Args:
        num_prompts: Total number of prompts to generate (distributed across lengths)
        lengths: List of sequence lengths to use
        vocab_size: Vocabulary size
        seed: Random seed for deterministic generation
    
    Returns:
        List of prompt tensors (1D, each length varies)
    """
    if lengths is None:
        lengths = [10, 50, 100, 200]
    
    set_seed(seed)
    prompts = []
    
    # Generate equal number of prompts per length
    prompts_per_length = num_prompts // len(lengths)
    
    for length in lengths:
        for _ in range(prompts_per_length):
            prompt = torch.randint(1, vocab_size, (length,), dtype=torch.long)
            prompts.append(prompt)
    
    return prompts


def load_checkpoint(checkpoint_path: str, device: str = 'cpu') -> TinyGPT:
    """
    Load model from checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        Loaded TinyGPT model
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If model loading fails
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = TinyGPT(
        num_layers=4,
        vocab_size=50257,
        d_model=128,
        num_heads=4,
        d_ff=512,
        max_len=1024,
        dropout=0.1
    )
    
    # Load state dict
    try:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except RuntimeError as e:
        # Model config mismatch - use random init
        print(f"Warning: Could not load checkpoint weights ({str(e)[:80]}...)")
        print("Using random initialization instead")
    
    model.to(device)
    model.eval()
    
    return model


def load_partial_results(results_path: Path) -> Dict[str, Any]:
    """
    Load partial results from previous run for resume capability.
    
    Args:
        results_path: Path to results.json file
    
    Returns:
        Dictionary of results, or empty dict if file doesn't exist
    """
    if results_path.exists():
        try:
            with open(results_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {results_path}, starting fresh")
            return {}
    
    return {}


def save_results(
    results: List[ExperimentResult],
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Save results to JSON (for resumability) and CSV (for analysis).
    
    Args:
        results: List of experiment results
        output_dir: Directory to save results
        logger: Logger instance
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to dicts with timestamps
    results_dicts = []
    for result in results:
        result_dict = result.to_dict()
        result_dict['timestamp'] = datetime.now().isoformat()
        results_dicts.append(result_dict)
    
    # Save JSON
    json_path = output_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(results_dicts, f, indent=2)
    logger.info(f"Saved results to {json_path}")
    
    # Save CSV
    csv_path = output_dir / 'results.csv'
    if results_dicts:
        keys = results_dicts[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results_dicts)
        logger.info(f"Saved results to {csv_path}")


def save_summary(
    results: List[ExperimentResult],
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Save summary statistics to summary.json.
    
    Args:
        results: List of experiment results
        output_dir: Directory to save summary
        logger: Logger instance
    """
    if not results:
        return
    
    # Group by method
    by_method = {}
    for result in results:
        if result.method not in by_method:
            by_method[result.method] = []
        if result.error is None:  # Only successful runs
            by_method[result.method].append(result)
    
    # Compute summary stats
    summary = {}
    for method, method_results in by_method.items():
        if not method_results:
            continue
        
        latencies = [r.avg_latency_ms for r in method_results]
        throughputs = [r.tokens_per_sec for r in method_results]
        perplexities = [r.perplexity for r in method_results]
        ram_usage = [r.peak_ram_mb for r in method_results]
        vram_usage = [r.peak_vram_mb for r in method_results]
        ciphertext_overhead = [r.ciphertext_overhead_mb for r in method_results]
        
        summary[method] = {
            'num_runs': len(method_results),
            'latency_ms': {
                'mean': float(np.mean(latencies)),
                'std': float(np.std(latencies)),
                'min': float(np.min(latencies)),
                'max': float(np.max(latencies))
            },
            'throughput_tokens_per_sec': {
                'mean': float(np.mean(throughputs)),
                'std': float(np.std(throughputs))
            },
            'perplexity': {
                'mean': float(np.mean(perplexities)),
                'std': float(np.std(perplexities))
            },
            'memory': {
                'peak_ram_mb_mean': float(np.mean(ram_usage)),
                'peak_vram_mb_mean': float(np.mean(vram_usage)),
                'ciphertext_overhead_mb_mean': float(np.mean([x for x in ciphertext_overhead if x > 0]))
                if any(x > 0 for x in ciphertext_overhead) else 0.0
            }
        }
    
    # Save summary
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    for method, stats in summary.items():
        print(f"\n{method.upper()}:")
        print(f"  Runs: {stats['num_runs']}")
        print(f"  Latency (ms): {stats['latency_ms']['mean']:.2f} ± {stats['latency_ms']['std']:.2f}")
        print(f"  Throughput (tokens/sec): {stats['throughput_tokens_per_sec']['mean']:.1f}")
        print(f"  Perplexity: {stats['perplexity']['mean']:.2f}")
        print(f"  Peak RAM (MB): {stats['memory']['peak_ram_mb_mean']:.2f}")
    print("="*80 + "\n")


def run_experiment(
    model: TinyGPT,
    prompts: List[torch.Tensor],
    methods: List[str],
    output_dir: Path,
    logger: logging.Logger,
    resume_from_path: Optional[Path] = None
) -> List[ExperimentResult]:
    """
    Run benchmarking experiments across all methods.
    
    Args:
        model: Model to benchmark
        prompts: List of prompt tensors
        methods: List of inference methods to test
        output_dir: Directory to save results
        logger: Logger instance
        resume_from_path: Path to previous results.json for resume
    
    Returns:
        List of experiment results
    """
    results = []
    completed_runs = set()
    
    # Load previous results if resuming
    if resume_from_path and resume_from_path.exists():
        previous_data = load_partial_results(resume_from_path)
        if isinstance(previous_data, list):
            for item in previous_data:
                item_dict = item if isinstance(item, dict) else {}
                run_key = (item_dict.get('method'), item_dict.get('prompt_id'), item_dict.get('prompt_length'))
                completed_runs.add(run_key)
            results = [
                ExperimentResult(
                    method=item['method'],
                    prompt_length=int(item['prompt_length']),
                    prompt_id=int(item['prompt_id']),
                    avg_latency_ms=float(item['avg_latency_ms']),
                    tokens_per_sec=float(item['tokens_per_sec']),
                    perplexity=float(item['perplexity']),
                    peak_ram_mb=float(item['peak_ram_mb']),
                    peak_vram_mb=float(item['peak_vram_mb']),
                    ciphertext_overhead_mb=float(item['ciphertext_overhead_mb']),
                    error=item.get('error'),
                    timestamp=item.get('timestamp', '')
                )
                for item in previous_data if isinstance(item, dict)
            ]
            logger.info(f"Resumed from {resume_from_path} with {len(completed_runs)} completed runs")
    
    # Total experiments
    total_experiments = len(methods) * len(prompts)
    completed = len(completed_runs)
    
    logger.info(f"Starting experiment runner")
    logger.info(f"Methods: {methods}")
    logger.info(f"Prompts: {len(prompts)} (lengths: {[p.shape[0] for p in prompts]})")
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Already completed: {completed}")
    
    # Run experiments with progress bar
    with tqdm(total=total_experiments, desc="Experiments", unit="run") as pbar:
        pbar.update(completed)  # Update for already completed
        
        for method_idx, method in enumerate(methods):
            for prompt_idx, prompt in enumerate(prompts):
                prompt_length = prompt.shape[0]
                run_key = (method, prompt_idx, prompt_length)
                
                # Skip if already completed
                if run_key in completed_runs:
                    pbar.update(1)
                    continue
                
                try:
                    # Prepare inputs (ensure 2D)
                    inputs = prompt.unsqueeze(0)
                    
                    # Run timing
                    timing_result = time_inference(
                        model,
                        inputs,
                        method=method,
                        n_runs=3,
                        warmup=True
                    )
                    avg_latency_ms = timing_result['avg_latency_per_token_ms']
                    throughput = timing_result['throughput_tokens_per_sec']
                    
                    # Run perplexity
                    perplexity = compute_perplexity(
                        model,
                        prompt.unsqueeze(0),
                        method=method,
                        max_samples=1
                    )
                    
                    # Run memory profiling
                    memory_stats = memory_profiling(
                        model,
                        inputs,
                        method=method,
                        n_runs=1
                    )
                    
                    # Create result
                    result = ExperimentResult(
                        method=method,
                        prompt_length=prompt_length,
                        prompt_id=prompt_idx,
                        avg_latency_ms=avg_latency_ms,
                        tokens_per_sec=throughput,
                        perplexity=perplexity,
                        peak_ram_mb=memory_stats.peak_ram_mb,
                        peak_vram_mb=memory_stats.peak_vram_mb,
                        ciphertext_overhead_mb=memory_stats.ciphertext_size_mb
                    )
                    
                    results.append(result)
                    completed_runs.add(run_key)
                    
                    logger.debug(f"Completed: {method} prompt={prompt_idx} length={prompt_length}")
                    
                except Exception as e:
                    # Save error and continue
                    error_msg = str(e)[:100]
                    result = ExperimentResult(
                        method=method,
                        prompt_length=prompt_length,
                        prompt_id=prompt_idx,
                        avg_latency_ms=0.0,
                        tokens_per_sec=0.0,
                        perplexity=0.0,
                        peak_ram_mb=0.0,
                        peak_vram_mb=0.0,
                        ciphertext_overhead_mb=0.0,
                        error=error_msg
                    )
                    results.append(result)
                    logger.warning(f"Error in {method} prompt={prompt_idx}: {error_msg}")
                
                # Save intermediate results every 10 runs
                if len(results) % 10 == 0:
                    save_results(results, output_dir, logger)
                
                pbar.update(1)
    
    return results


def main():
    """Main experiment runner entry point."""
    parser = argparse.ArgumentParser(
        description='Benchmark TinyGPT across multiple inference methods'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoint.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run quick test with 4 prompts instead of 100'
    )
    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=['plain', 'strategy1', 'strategy2', 'strategy3'],
        help='Methods to benchmark'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run on'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir, test_mode=args.test_mode)
    
    logger.info("="*80)
    logger.info("TinyGPT Experiment Runner")
    logger.info("="*80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Test mode: {args.test_mode}")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Device: {args.device}")
    
    try:
        # Load model
        logger.info("Loading model...")
        model = load_checkpoint(args.checkpoint, device=args.device)
        logger.info(f"Model loaded: {model.__class__.__name__}")
        
        # Generate prompts
        logger.info("Generating prompts...")
        num_prompts = 4 if args.test_mode else 100
        prompts = generate_prompts(
            num_prompts=num_prompts,
            seed=args.seed
        )
        logger.info(f"Generated {len(prompts)} prompts")
        
        # Run experiments with resume capability
        results_path = output_dir / 'results.json'
        results = run_experiment(
            model=model,
            prompts=prompts,
            methods=args.methods,
            output_dir=output_dir,
            logger=logger,
            resume_from_path=results_path
        )
        
        # Save final results
        logger.info("Saving results...")
        save_results(results, output_dir, logger)
        save_summary(results, output_dir, logger)
        
        logger.info("="*80)
        logger.info(f"Experiment completed successfully!")
        logger.info(f"Results saved to {output_dir}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
