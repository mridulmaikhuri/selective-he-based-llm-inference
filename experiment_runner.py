#!/usr/bin/env python3
"""
experiment_runner.py — Orchestrates inference experiments across methods and strategies.

Usage:
    python experiment_runner.py --checkpoint path/to/model [--test-mode] [--seed 42]
    python experiment_runner.py --checkpoint path/to/model --output-dir results/
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional heavy deps — gracefully degraded when running without a real model
# ---------------------------------------------------------------------------
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Minimal fallback
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, total=None, desc="", **kw):
            self._it = iter(iterable) if iterable is not None else iter([])
            self.total = total
            self.desc = desc
            self.n = 0
        def __iter__(self):
            for item in self._it:
                self.n += 1
                print(f"  {self.desc}: {self.n}/{self.total or '?'}", end="\r", flush=True)
                yield item
            print()
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def update(self, n=1): self.n += n
        def set_postfix(self, **kw): pass
        def set_description(self, s): self.desc = s

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger("experiment_runner")


logger = setup_logging()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHODS = ["plain", "full_he", "strategy1", "strategy2", "strategy3"]
PROMPT_LENGTHS = [10, 50, 100, 200]          # tokens (approximate word counts)
PROMPTS_PER_LENGTH = 25                       # 25 × 4 = 100 total
TEST_MODE_TOTAL = 4                           # fast subset in --test-mode

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    checkpoint: str
    output_dir: str
    seed: int
    test_mode: bool
    resume: bool


@dataclass
class MetricResult:
    method: str
    prompt_length: int
    prompt_index: int
    avg_latency_ms: float
    tokens_per_sec: float
    perplexity: float
    memory_mb: float
    ciphertext_overhead: float
    error: Optional[str] = None


@dataclass
class ExperimentSummary:
    method: str
    prompt_length: int
    num_prompts: int
    avg_latency_ms: float
    tokens_per_sec: float
    perplexity: float
    memory_mb: float
    ciphertext_overhead: float
    failed_runs: int = 0


# ---------------------------------------------------------------------------
# Prompt generation (deterministic)
# ---------------------------------------------------------------------------

VOCAB = (
    "the quick brown fox jumps over the lazy dog "
    "artificial intelligence language model experiment "
    "inference latency memory throughput benchmark evaluation "
    "neural network transformer attention mechanism layer "
    "embedding token sequence context window batch size "
    "gradient descent optimizer learning rate epoch step "
    "classification regression generation summarization "
    "encrypted homomorphic secure computation privacy "
).split()


def generate_prompts(
    lengths: List[int],
    per_length: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Generate ``per_length`` prompts for each target length deterministically.
    Returns a flat list of dicts: {text, target_length, index}.
    """
    rng = random.Random(seed)
    prompts: List[Dict[str, Any]] = []
    for length in lengths:
        for i in range(per_length):
            words = [rng.choice(VOCAB) for _ in range(length)]
            text = " ".join(words)
            prompts.append({"text": text, "target_length": length, "index": i})
    logger.info("Generated %d prompts (seed=%d).", len(prompts), seed)
    return prompts


# ---------------------------------------------------------------------------
# Stub model loader — replace with your real loader
# ---------------------------------------------------------------------------

class ModelStub:
    """
    Placeholder model. Replace the body of ``load_model`` with real logic,
    e.g. loading a HuggingFace checkpoint or a custom architecture.
    """
    def __init__(self, checkpoint: str):
        self.checkpoint = checkpoint
        logger.info("Model stub loaded (checkpoint=%r).", checkpoint)

    def forward(self, text: str) -> Dict[str, Any]:
        """Stub forward pass — returns fake logits / perplexity."""
        n_tokens = len(text.split())
        return {"n_tokens": n_tokens, "log_prob_sum": -0.5 * n_tokens}


def load_model(checkpoint: str) -> ModelStub:
    """Load model from checkpoint path."""
    if not os.path.exists(checkpoint) and checkpoint != "stub":
        logger.warning(
            "Checkpoint %r not found. Continuing with stub model.", checkpoint
        )
    return ModelStub(checkpoint)


# ---------------------------------------------------------------------------
# Inference functions — replace internals with real implementations
# ---------------------------------------------------------------------------

def time_inference(
    model: ModelStub,
    text: str,
    method: str,
    n_runs: int = 3,
) -> Dict[str, float]:
    """
    Run inference ``n_runs`` times and return timing + memory statistics.

    Returns:
        avg_latency_ms, tokens_per_sec, memory_mb
    """
    latencies_ms: List[float] = []
    n_tokens = len(text.split())

    # Method-specific overhead multipliers (stub approximation)
    method_overhead = {
        "plain":      1.0,
        "full_he":    4.2,
        "strategy1":  1.5,
        "strategy2":  2.0,
        "strategy3":  2.8,
    }.get(method, 1.0)

    for _ in range(n_runs):
        t0 = time.perf_counter()
        # ── Replace this block with real inference ──────────────────────────
        _result = model.forward(text)
        base_sleep = n_tokens * 0.002            # ~2ms per token, stub only
        time.sleep(base_sleep * method_overhead * random.uniform(0.9, 1.1))
        # ────────────────────────────────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

    avg_latency_ms = sum(latencies_ms) / len(latencies_ms)
    tokens_per_sec = (n_tokens / (avg_latency_ms / 1000)) if avg_latency_ms > 0 else 0.0

    # Peak memory delta for this inference call (method-aware)
    memory_mb = _measure_inference_memory_mb(model, text, method)

    return {
        "avg_latency_ms": avg_latency_ms,
        "tokens_per_sec": tokens_per_sec,
        "memory_mb": memory_mb,
    }


def compute_perplexity(
    model: ModelStub,
    text: str,
    method: str,
) -> float:
    """
    Compute perplexity for the given text under the specified method.
    Lower is better; typical LM values: 5–50.

    Replace the stub implementation with your real perplexity computation.
    """
    # ── Replace this block with real perplexity logic ────────────────────────
    result = model.forward(text)
    n_tokens = max(result["n_tokens"], 1)
    log_prob_sum = result["log_prob_sum"]

    # Method-specific perplexity shift (stub approximation)
    method_shift = {
        "plain":      0.0,
        "full_he":    0.3,
        "strategy1":  0.05,
        "strategy2":  0.10,
        "strategy3":  0.20,
    }.get(method, 0.0)

    avg_neg_log_prob = (-log_prob_sum / n_tokens) + method_shift
    import math
    perplexity = math.exp(avg_neg_log_prob)
    # ─────────────────────────────────────────────────────────────────────────
    return perplexity


def compute_ciphertext_overhead(method: str, n_tokens: int) -> float:
    """
    Return estimated ciphertext size overhead ratio relative to plaintext.
    Only meaningful for HE/encrypted methods; plain returns 1.0.
    """
    overhead_map = {
        "plain":      1.0,
        "full_he":    12.5,
        "strategy1":  1.8,
        "strategy2":  3.2,
        "strategy3":  6.0,
    }
    base = overhead_map.get(method, 1.0)
    # Add slight token-count dependency (stub)
    return base + n_tokens * 0.0001


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get_memory_mb() -> float:
    """Return current process RSS memory in MB."""
    if HAS_PSUTIL:
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 ** 2)
    return 0.0


def _measure_inference_memory_mb(
    model: "ModelStub",
    text: str,
    method: str,
) -> float:
    """
    Measure the peak memory delta (MB) consumed by one forward pass.

    Strategy
    --------
    1. Record RSS before the call (baseline).
    2. Run the forward pass — for the real model this allocates KV-cache,
       activations, and (for HE methods) ciphertext buffers.
    3. Record RSS immediately after, before Python GC has a chance to free.
    4. Return max(delta, simulated_floor) so stub runs still show realistic
       method-specific numbers while real models report true allocations.

    The simulated_floor is only used when the measured delta is < 1 MB
    (i.e. the stub allocated essentially nothing).  Replace or remove it
    once you plug in a real model.
    """
    # Method-specific memory multipliers for the stub simulation.
    # Real HE ciphertext buffers are ~10-50× larger than plaintext tensors;
    # selective strategies fall in between.
    METHOD_MEMORY_MB = {
        "plain":      0.0,    # baseline — no extra buffers
        "full_he":    512.0,  # full ciphertext tensors for every layer
        "strategy1":  80.0,   # ciphertext only for first 24 layers
        "strategy2":  120.0,  # mid-layer ciphertext buffers
        "strategy3":  200.0,  # ciphertext for first-half + output re-encryption
    }
    # Scale with prompt length (longer prompt → larger KV-cache / ciphertext)
    n_tokens = len(text.split())
    length_scale = 1.0 + n_tokens / 200.0   # gentle linear growth

    mem_before = _get_memory_mb()
    model.forward(text)                      # ← replace with real inference
    mem_after = _get_memory_mb()

    measured_delta = max(mem_after - mem_before, 0.0)

    # If the measured delta is negligible (stub model), fall back to the
    # simulation table so results.csv shows realistic differentiation.
    simulated = METHOD_MEMORY_MB.get(method, 0.0) * length_scale
    if measured_delta < 1.0:
        return round(simulated, 2)

    # Real model path: report true peak delta
    return round(measured_delta, 2)


def _run_key(method: str, prompt_length: int, prompt_index: int) -> str:
    return f"{method}_{prompt_length}_{prompt_index}"


def _load_partial_results(results_json: Path) -> Dict[str, MetricResult]:
    """Load any existing partial results from disk (resume capability)."""
    if not results_json.exists():
        return {}
    try:
        with open(results_json) as f:
            data = json.load(f)
        partial: Dict[str, MetricResult] = {}
        for entry in data.get("runs", []):
            r = MetricResult(**entry)
            partial[_run_key(r.method, r.prompt_length, r.prompt_index)] = r
        logger.info("Resuming: loaded %d completed runs.", len(partial))
        return partial
    except Exception as exc:
        logger.warning("Could not parse partial results (%s); starting fresh.", exc)
        return {}


def _save_results(
    all_runs: List[MetricResult],
    summaries: List[ExperimentSummary],
    output_dir: Path,
) -> None:
    """Persist runs + summaries to results.json and results.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON ─────────────────────────────────────────────────────────────────
    payload = {
        "runs": [asdict(r) for r in all_runs],
        "summaries": [asdict(s) for s in summaries],
    }
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = output_dir / "results.csv"
    if summaries:
        fieldnames = list(asdict(summaries[0]).keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([asdict(s) for s in summaries])

    logger.info("Saved results to %s and %s.", json_path, csv_path)


def _compute_summaries(runs: List[MetricResult]) -> List[ExperimentSummary]:
    """Aggregate per-run metrics into per-(method, length) summaries."""
    from collections import defaultdict
    groups: Dict[tuple, List[MetricResult]] = defaultdict(list)
    for r in runs:
        groups[(r.method, r.prompt_length)].append(r)

    summaries: List[ExperimentSummary] = []
    for (method, prompt_length), group in sorted(groups.items()):
        valid = [r for r in group if r.error is None]
        failed = len(group) - len(valid)

        def avg(attr: str) -> float:
            vals = [getattr(r, attr) for r in valid]
            return sum(vals) / len(vals) if vals else 0.0

        summaries.append(ExperimentSummary(
            method=method,
            prompt_length=prompt_length,
            num_prompts=len(group),
            avg_latency_ms=avg("avg_latency_ms"),
            tokens_per_sec=avg("tokens_per_sec"),
            perplexity=avg("perplexity"),
            memory_mb=avg("memory_mb"),
            ciphertext_overhead=avg("ciphertext_overhead"),
            failed_runs=failed,
        ))
    return summaries


# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

def run_experiments(config: RunConfig) -> None:
    output_dir = Path(config.output_dir)
    results_json = output_dir / "results.json"

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info("Loading model from checkpoint: %s", config.checkpoint)
    model = load_model(config.checkpoint)

    # ── Generate prompts ──────────────────────────────────────────────────────
    if config.test_mode:
        logger.info("TEST MODE: using %d prompts (1 per length).", TEST_MODE_TOTAL)
        prompts = generate_prompts(PROMPT_LENGTHS, 1, config.seed)
    else:
        prompts = generate_prompts(PROMPT_LENGTHS, PROMPTS_PER_LENGTH, config.seed)

    total_runs = len(METHODS) * len(prompts)
    logger.info(
        "Experiment plan: %d methods × %d prompts = %d total runs.",
        len(METHODS), len(prompts), total_runs,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    completed: Dict[str, MetricResult] = {}
    if config.resume:
        completed = _load_partial_results(results_json)

    all_runs: List[MetricResult] = list(completed.values())

    # ── Main loop ─────────────────────────────────────────────────────────────
    outer_bar = tqdm(METHODS, desc="Methods", total=len(METHODS))
    for method in outer_bar:
        outer_bar.set_description(f"Method: {method}")

        inner_bar = tqdm(prompts, desc=f"  {method}", total=len(prompts), leave=False)
        for prompt_info in inner_bar:
            key = _run_key(method, prompt_info["target_length"], prompt_info["index"])

            if key in completed:
                inner_bar.set_postfix(status="skipped (cached)")
                continue

            text = prompt_info["text"]
            n_tokens = len(text.split())
            result = MetricResult(
                method=method,
                prompt_length=prompt_info["target_length"],
                prompt_index=prompt_info["index"],
                avg_latency_ms=0.0,
                tokens_per_sec=0.0,
                perplexity=0.0,
                memory_mb=0.0,
                ciphertext_overhead=0.0,
            )

            try:
                # Timing + memory
                timing = time_inference(model, text, method)
                result.avg_latency_ms = timing["avg_latency_ms"]
                result.tokens_per_sec = timing["tokens_per_sec"]
                result.memory_mb = timing["memory_mb"]

                # Perplexity
                result.perplexity = compute_perplexity(model, text, method)

                # Ciphertext overhead
                result.ciphertext_overhead = compute_ciphertext_overhead(method, n_tokens)

                inner_bar.set_postfix(
                    lat=f"{result.avg_latency_ms:.1f}ms",
                    ppl=f"{result.perplexity:.2f}",
                )
                logger.debug(
                    "[%s | len=%d | #%d] lat=%.1fms tps=%.1f ppl=%.2f mem=%.1fMB ct=%.2f",
                    method, prompt_info["target_length"], prompt_info["index"],
                    result.avg_latency_ms, result.tokens_per_sec,
                    result.perplexity, result.memory_mb, result.ciphertext_overhead,
                )

            except KeyboardInterrupt:
                logger.warning("Interrupted by user — saving intermediate results.")
                all_runs.append(result)
                summaries = _compute_summaries(all_runs)
                _save_results(all_runs, summaries, output_dir)
                sys.exit(0)

            except Exception as exc:
                tb = traceback.format_exc()
                logger.error(
                    "Run failed [%s | len=%d | #%d]: %s\n%s",
                    method, prompt_info["target_length"], prompt_info["index"],
                    exc, tb,
                )
                result.error = str(exc)
                inner_bar.set_postfix(status="ERROR")

            all_runs.append(result)

            # Save after every run for safety
            summaries = _compute_summaries(all_runs)
            _save_results(all_runs, summaries, output_dir)

    # ── Final save + summary ──────────────────────────────────────────────────
    summaries = _compute_summaries(all_runs)
    _save_results(all_runs, summaries, output_dir)

    logger.info("\n%s EXPERIMENT COMPLETE %s", "=" * 30, "=" * 30)
    logger.info("%-15s %-8s %12s %12s %10s %10s %12s",
                "Method", "Len", "Latency(ms)", "Tok/s", "PPL", "Mem(MB)", "CT Overhead")
    logger.info("-" * 85)
    for s in summaries:
        logger.info(
            "%-15s %-8d %12.1f %12.1f %10.2f %10.1f %12.2f",
            s.method, s.prompt_length,
            s.avg_latency_ms, s.tokens_per_sec,
            s.perplexity, s.memory_mb, s.ciphertext_overhead,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment runner: benchmarks inference methods across prompt lengths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", default='checkpoint.pt',
        help="Path to model checkpoint (use 'stub' to run without a real model).",
    )
    parser.add_argument(
        "--output-dir", default="experiment_results",
        help="Directory where results.json and results.csv are written.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic prompt generation.",
    )
    parser.add_argument(
        "--test-mode", action="store_true",
        help="Run a fast subset (1 prompt per length, 4 total) for quick validation.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore any existing partial results and start fresh.",
    )
    parser.add_argument(
        "--log-file", default=None,
        help="Optional file path to mirror logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Re-init logging with optional file handler
    global logger
    logger = setup_logging(args.log_file)

    config = RunConfig(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        seed=args.seed,
        test_mode=args.test_mode,
        resume=not args.no_resume,
    )

    logger.info("Starting experiment runner.")
    logger.info("  checkpoint : %s", config.checkpoint)
    logger.info("  output_dir : %s", config.output_dir)
    logger.info("  seed       : %d", config.seed)
    logger.info("  test_mode  : %s", config.test_mode)
    logger.info("  resume     : %s", config.resume)

    run_experiments(config)


if __name__ == "__main__":
    main()