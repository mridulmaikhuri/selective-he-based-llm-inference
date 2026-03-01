import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from model import TinyGPT
from benchmarks import (
    set_random_seeds,
    time_inference,
    compute_perplexity,
    memory_profiling,
)


class PromptDataset(Dataset):
    """
    Simple dataset wrapping prompts for perplexity computation.

    Each item is a pair (x, y) where y is the next-token shifted
    version of x, matching standard language modeling targets.
    """

    def __init__(self, prompts: List[torch.Tensor]) -> None:
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Any:
        tokens = self.prompts[idx]
        # tokens shape: (seq_len,)
        x = tokens[:-1]
        y = tokens[1:]
        return x, y


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> TinyGPT:
    """
    Load TinyGPT model from training checkpoint.

    Expects checkpoint produced by train.py, containing 'model_state_dict'
    and 'args' (with model hyperparameters).
    """
    logging.info("Loading checkpoint from %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)

    args_dict = ckpt.get("args")
    if args_dict is None:
        raise ValueError(
            "Checkpoint does not contain 'args'. Cannot reconstruct model config."
        )

    model = TinyGPT(
        num_layers=args_dict.get("num_layers", 4),
        vocab_size=args_dict.get("vocab_size", 50257),
        d_model=args_dict.get("d_model", 128),
        num_heads=args_dict.get("num_heads", 4),
        d_ff=args_dict.get("d_ff", 512),
        max_len=args_dict.get("seq_len", 128),
        dropout=args_dict.get("dropout", 0.1),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logging.info(
        "Loaded model: layers=%d, d_model=%d, vocab_size=%d, max_len=%d",
        model.num_layers,
        model.d_model,
        model.vocab_size,
        model.max_len,
    )
    return model


def generate_prompts(
    num_per_length: int,
    lengths: List[int],
    vocab_size: int,
    seed: int,
    device: torch.device,
    max_len: int,
) -> List[torch.Tensor]:
    """
    Generate prompts deterministically across specified lengths.
    """
    set_random_seeds(seed)

    prompts: List[torch.Tensor] = []
    for length in lengths:
        # Ensure prompts do not exceed model capacity
        eff_len = min(length, max_len)
        for _ in range(num_per_length):
            # Avoid token id 0 so that ignore_index=0 does not drop tokens
            prompt = torch.randint(
                1,
                vocab_size,
                (eff_len,),
                dtype=torch.long,
                device=device,
            )
            prompts.append(prompt)

    logging.info(
        "Generated %d prompts across lengths %s (capped at max_len=%d)",
        len(prompts),
        lengths,
        max_len,
    )
    return prompts


def default_results_structure(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "config": config,
        "methods": {},
    }


def save_results_json(path: Path, results: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logging.info("Saved JSON results to %s", path)


def save_results_csv(path: Path, results: Dict[str, Any]) -> None:
    methods = results.get("methods", {})
    fieldnames = [
        "method",
        "status",
        "avg_latency_ms",
        "tokens_per_sec",
        "perplexity",
        "memory_mb",
        "ciphertext_overhead_mb",
        "error",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for method_name, metrics in methods.items():
            row = {
                "method": method_name,
                "status": metrics.get("status", "unknown"),
                "avg_latency_ms": metrics.get("avg_latency_ms"),
                "tokens_per_sec": metrics.get("tokens_per_sec"),
                "perplexity": metrics.get("perplexity"),
                "memory_mb": metrics.get("memory_mb"),
                "ciphertext_overhead_mb": metrics.get("ciphertext_overhead_mb"),
                "error": metrics.get("error"),
            }
            writer.writerow(row)

    logging.info("Saved CSV results to %s", path)


def run_experiments(
    checkpoint: Path,
    output_prefix: Path,
    seed: int,
    test_mode: bool,
) -> None:
    logging.info("Starting experiment runner (test_mode=%s)", test_mode)
    print(f"[experiment_runner] Starting experiments (test_mode={test_mode})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    model = load_model_from_checkpoint(checkpoint, device)
    print("[experiment_runner] Model loaded from checkpoint.")

    # Configure prompt generation
    if test_mode:
        lengths = [10, 50]
        num_per_length = 2  # 4 prompts total
        # In test mode, keep a minimal comparison without per-strategy breakdown.
        methods = ["plain", "full_he"]
        n_runs = 2
    else:
        lengths = [10, 50, 100, 200]
        num_per_length = 25  # 100 prompts total
        # Omit strategy3 from comparisons; too heavy / unstable in practice.
        methods = ["plain", "full_he", "strategy1", "strategy2"]
        n_runs = 5

    print(
        f"[experiment_runner] Generating prompts: lengths={lengths}, "
        f"num_per_length={num_per_length}, seed={seed}"
    )
    prompts = generate_prompts(
        num_per_length=num_per_length,
        lengths=lengths,
        vocab_size=model.vocab_size,
        seed=seed,
        device=device,
        max_len=model.max_len,
    )
    print(f"[experiment_runner] Generated {len(prompts)} prompts.")

    # Prepare outputs and resume state
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")

    if json_path.exists():
        logging.info("Found existing results JSON at %s, loading for resume", json_path)
        with json_path.open("r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        config = {
            "checkpoint": str(checkpoint),
            "lengths": lengths,
            "num_per_length": num_per_length,
            "seed": seed,
            "test_mode": test_mode,
        }
        results = default_results_structure(config)

    methods_state: Dict[str, Any] = results.setdefault("methods", {})

    for method in tqdm(methods, desc="Methods", unit="method"):
        method_metrics = methods_state.get(method, {})
        if method_metrics.get("status") == "done":
            logging.info("Skipping method '%s' (already completed).", method)
            print(f"[experiment_runner] Skipping '{method}' (already done).")
            continue

        logging.info("Running method '%s'", method)
        print(f"\n[experiment_runner] === Running method: {method} ===")

        method_result: Dict[str, Any] = {
            "status": "pending",
        }
        methods_state[method] = method_result

        try:
            # Select a lighter subset for HE-heavy methods to avoid OOM/kill.
            # full_he only supports seq_len=1, so we create 1-token prompts.
            if method == "full_he":
                max_prompts = 2 if test_mode else 4
                # Build dedicated 1-token prompts for full HE
                method_prompts = [
                    torch.randint(
                        1,
                        model.vocab_size,
                        (1,),
                        dtype=torch.long,
                        device=device,
                    )
                    for _ in range(max_prompts)
                ]
                n_runs_method = min(n_runs, 2)
            elif method in {"strategy1", "strategy2"}:
                max_prompts = 4 if test_mode else 8
                method_prompts = prompts[:max_prompts]
                n_runs_method = min(n_runs, 2)
            else:
                method_prompts = prompts
                n_runs_method = n_runs

            time_inputs: List[torch.Tensor] = [p.unsqueeze(0) for p in method_prompts]
            val_dataset = PromptDataset(method_prompts)

            # Timing
            print(f"[experiment_runner]  -> timing ({method})...")
            timing = time_inference(
                model,
                time_inputs,
                method=method,
                n_runs=n_runs_method,
            )

            avg_latency_ms = float(timing["stats"]["mean_ms"])
            tokens_per_sec = float(timing["throughput"])

            # Memory
            print(f"[experiment_runner]  -> memory profiling ({method})...")
            mem = memory_profiling(
                model,
                method=method,
                sample_input=time_inputs[0],
            )
            memory_mb = float(mem["peak_ram_mb"] + mem["peak_vram_mb"])
            ciphertext_overhead_mb = float(mem.get("he_ciphertext_memory_mb", 0.0))

            # Perplexity
            if method == "full_he":
                # Perplexity for full HE (seq_len=1 only) is not meaningful.
                print(
                    "[experiment_runner]  -> perplexity (full_he skipped; "
                    "only seq_len=1 supported)"
                )
                perplexity = float("nan")
            else:
                print(f"[experiment_runner]  -> perplexity ({method})...")
                perplexity = float(
                    compute_perplexity(
                        model,
                        val_dataset,
                        method=method,
                        batch_size=1,
                        ignore_index=0,
                    )
                )

            method_result.update(
                {
                    "status": "done",
                    "avg_latency_ms": avg_latency_ms,
                    "tokens_per_sec": tokens_per_sec,
                    "perplexity": perplexity,
                    "memory_mb": memory_mb,
                    "ciphertext_overhead_mb": ciphertext_overhead_mb,
                }
            )

            logging.info(
                "Method '%s' done: latency=%.2f ms, throughput=%.2f tok/s, "
                "ppl=%.2f, mem=%.2f MB, ct_overhead=%.2f MB",
                method,
                avg_latency_ms,
                tokens_per_sec,
                perplexity,
                memory_mb,
                ciphertext_overhead_mb,
            )
            print(
                f"[experiment_runner] DONE {method}: "
                f"lat={avg_latency_ms:.2f}ms, tok/s={tokens_per_sec:.2f}, "
                f"ppl={perplexity:.2f}, mem={memory_mb:.2f}MB, "
                f"ct≈{ciphertext_overhead_mb:.2f}MB"
            )

        except Exception as exc:  # noqa: BLE001
            logging.exception("Error while running method '%s'", method)
            method_result.update(
                {
                    "status": "error",
                    "error": str(exc),
                }
            )
            print(f"[experiment_runner] ERROR in method '{method}': {exc}")

        # Always save intermediate state
        save_results_json(json_path, results)
        save_results_csv(csv_path, results)

    logging.info("All methods processed. Final results written.")
    print("[experiment_runner] All methods processed. Results saved.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment runner for TinyGPT benchmarking across methods.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained TinyGPT checkpoint (from train.py).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="results",
        help="Prefix for results files (e.g., 'results' -> results.json, results.csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt generation.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run a fast subset of experiments (few prompts, fewer methods).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )

    args = parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_prefix = Path(args.output_prefix).expanduser().resolve()

    run_experiments(
        checkpoint=checkpoint_path,
        output_prefix=output_prefix,
        seed=args.seed,
        test_mode=bool(args.test_mode),
    )


if __name__ == "__main__":
    main()

