import time
import math
import os
import random
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import torch
import numpy as np


TensorOrList = Union[torch.Tensor, Iterable[torch.Tensor]]


def set_random_seeds(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducible benchmarks.

    This controls Python, NumPy and PyTorch RNGs. For fully reproducible
    CUDA results you may also need to set environment flags outside Python.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def _model_uses_cuda(model: torch.nn.Module) -> bool:
    return any(p.is_cuda for p in model.parameters())


def _maybe_synchronize_cuda(model: torch.nn.Module) -> None:
    if torch.cuda.is_available() and _model_uses_cuda(model):
        torch.cuda.synchronize()


def _ensure_tensor_iter(inputs: TensorOrList) -> List[torch.Tensor]:
    if isinstance(inputs, torch.Tensor):
        return [inputs]
    return list(inputs)


def _infer_plain(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(x)


def _build_method_inference(
    model: torch.nn.Module,
    method: str,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Dict[str, Any]]:
    """
    Return a callable that runs one forward pass according to `method`.

    Additional metadata useful for memory profiling can be returned in
    the second element of the tuple.
    """
    method = method.lower()

    if method == "plain":
        return _infer_plain, {}

    # Selective HE strategies: strategy1 / strategy2 / strategy3
    if method.startswith("strategy"):
        from selective_he_config import load_strategy_config
        from he_utils import setup_HE_context
        from selective_he_engine import selective_HE_inference

        try:
            strategy_id = int(method.replace("strategy", ""))
        except ValueError as exc:
            raise ValueError(f"Unsupported method '{method}'.") from exc

        config = load_strategy_config(strategy_id)
        config.validate(model)
        HE = setup_HE_context(n=2 ** 14, t=65537)

        def _infer_he(x: torch.Tensor) -> torch.Tensor:
            logits, _, _ = selective_HE_inference(
                model, x, HE, config, verbose=False
            )
            return logits

        meta = {"he_config": config, "he_context": HE, "he_type": "selective"}
        return _infer_he, meta

    # Full HE pipeline for ToyTransformer (he_inference.full_he_inference)
    if method in {"full_he", "he_full"}:
        from he_utils import setup_HE_context
        from he_inference import full_he_inference

        HE = setup_HE_context(n=2 ** 14, t=65537)

        def _infer_full_he(x: torch.Tensor) -> torch.Tensor:
            # full_he_inference returns (top_5_tokens, timings, memory_info)
            _, _, _ = full_he_inference(model, x, HE)
            # For timing interfaces that expect logits, we just return
            # a dummy tensor with the correct batch dimension.
            return torch.empty(0)

        meta = {"he_context": HE, "he_type": "full"}
        return _infer_full_he, meta

    raise ValueError(f"Unknown inference method '{method}'.")


def warmup_runs(
    model: torch.nn.Module,
    inputs: TensorOrList,
    method: str = "plain",
    n_warmup: int = 3,
) -> None:
    """
    Run a few warm-up passes to stabilize caches and JITs.
    """
    model.eval()
    input_list = _ensure_tensor_iter(inputs)
    infer_fn, _ = _build_method_inference(model, method)

    for _ in range(max(0, n_warmup)):
        for x in input_list:
            _ = infer_fn(x)
    _maybe_synchronize_cuda(model)


def time_inference(
    model: torch.nn.Module,
    inputs: TensorOrList,
    method: str = "plain",
    n_runs: int = 10,
) -> Dict[str, Any]:
    """
    Benchmark inference latency.

    Returns a dict with:
        - TTFT: float, time to first token (ms, approximated)
        - avg_latency_per_token: float (ms/token)
        - throughput: float (tokens/sec)
        - stats: dict with mean/std/min/max latency in ms
    """
    model.eval()
    if n_runs <= 0:
        raise ValueError("n_runs must be > 0.")

    input_list = _ensure_tensor_iter(inputs)
    infer_fn, _ = _build_method_inference(model, method)

    # Warm-up (un-timed but synchronized)
    warmup_runs(model, input_list, method=method, n_warmup=2)

    latencies_ms: List[float] = []
    total_tokens = 0

    for run_idx in range(n_runs):
        for x in input_list:
            # Estimate token count from final dimension before vocab
            if x.dim() == 2:
                total_tokens += int(x.numel())
            else:
                total_tokens += int(x.shape[0] * x.shape[1])

            _maybe_synchronize_cuda(model)
            t0 = time.perf_counter()
            _ = infer_fn(x)
            _maybe_synchronize_cuda(model)
            t1 = time.perf_counter()

            lat_ms = (t1 - t0) * 1000.0
            latencies_ms.append(lat_ms)

    lat_arr = np.array(latencies_ms, dtype=np.float64)
    mean_ms = float(lat_arr.mean())
    std_ms = float(lat_arr.std(ddof=0))
    min_ms = float(lat_arr.min())
    max_ms = float(lat_arr.max())

    # Approximate TTFT as latency of first run divided by sequence length.
    first_latency_ms = latencies_ms[0]
    first_tokens = max(1, total_tokens // len(latencies_ms))
    ttft_ms = first_latency_ms / first_tokens

    avg_latency_per_token_ms = mean_ms / max(1, first_tokens)
    total_time_s = float(lat_arr.sum() / 1000.0)
    throughput = (total_tokens / total_time_s) if total_time_s > 0 else 0.0

    result: Dict[str, Any] = {
        "TTFT": float(ttft_ms),
        "avg_latency_per_token": float(avg_latency_per_token_ms),
        "throughput": float(throughput),
        "stats": {
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
        },
    }

    print(
        f"[time_inference] method={method} "
        f"runs={len(latencies_ms)} "
        f"mean={mean_ms:.3f}ms "
        f"ttft≈{ttft_ms:.3f}ms "
        f"throughput={throughput:.2f} tok/s"
    )

    return result


@torch.no_grad()
def compute_perplexity(
    model: torch.nn.Module,
    val_dataset: Any,
    method: str = "plain",
    batch_size: int = 8,
    ignore_index: int = 0,
) -> float:
    """
    Compute perplexity on a validation split using method-specific inference.

    `val_dataset` can be a `torch.utils.data.Dataset` of (x, y) pairs or
    an existing `DataLoader`. For HE methods, selective HE inference is
    used where available.
    """
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    model.eval()
    device = next(model.parameters()).device

    if isinstance(val_dataset, DataLoader):
        val_loader = val_dataset
    else:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    infer_fn, _ = _build_method_inference(model, method)

    total_loss = 0.0
    total_tokens = 0

    for batch in val_loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        else:
            # Assume language-model style where y is next-token shifted version of x
            x = batch
            if isinstance(x, torch.Tensor):
                y = x[:, 1:].clone()
                x = x[:, :-1].clone()
            else:
                raise ValueError("Unsupported batch format for perplexity computation.")

        if x.dim() == 1:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        x = x.to(device)
        y = y.to(device)

        logits = infer_fn(x)
        if logits.numel() == 0:
            raise RuntimeError(
                "Inference method did not return logits; cannot compute perplexity."
            )

        logits = logits.view(-1, logits.size(-1))
        targets = y.view(-1)

        loss = F.cross_entropy(
            logits, targets, reduction="sum", ignore_index=ignore_index
        )
        nonpad = (targets != ignore_index).sum().item()
        total_loss += loss.item()
        total_tokens += nonpad

    avg_loss = total_loss / total_tokens if total_tokens > 0 else math.inf
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    print(
        f"[compute_perplexity] method={method} "
        f"loss={avg_loss:.4f} ppl={ppl:.2f} tokens={total_tokens}"
    )
    return float(ppl)


def memory_profiling(
    model: torch.nn.Module,
    method: str = "plain",
    sample_input: torch.Tensor | None = None,
) -> Dict[str, Any]:
    """
    Profile peak RAM and VRAM usage for a single inference run.

    For HE methods, also estimates ciphertext memory footprint where possible.
    """
    import psutil

    model.eval()
    infer_fn, meta = _build_method_inference(model, method)

    device = next(model.parameters()).device
    if sample_input is None:
        seq_len = getattr(model, "max_len", 16)
        vocab_size = getattr(model, "vocab_size", 256)
        sample_input = torch.randint(
            0, vocab_size, (1, seq_len), dtype=torch.long, device=device
        )
    else:
        sample_input = sample_input.to(device)

    # Reset CUDA peak stats if applicable
    if torch.cuda.is_available() and _model_uses_cuda(model):
        torch.cuda.reset_peak_memory_stats(device)

    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss

    _maybe_synchronize_cuda(model)
    t0 = time.perf_counter()
    _ = infer_fn(sample_input)
    _maybe_synchronize_cuda(model)
    t1 = time.perf_counter()

    ram_after = process.memory_info().rss
    peak_ram_bytes = max(ram_before, ram_after) - ram_before
    peak_ram_mb = peak_ram_bytes / (1024 ** 2)

    if torch.cuda.is_available() and _model_uses_cuda(model):
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
    else:
        peak_vram_bytes = 0
    peak_vram_mb = peak_vram_bytes / (1024 ** 2)

    he_ciphertexts_est = 0
    he_ciphertext_mb = 0.0

    # If full HE pipeline exposes memory info, re-use its estimate.
    if meta.get("he_type") == "full":
        from he_utils import setup_HE_context
        from he_inference import full_he_inference

        HE = setup_HE_context(n=2 ** 14, t=65537)
        _, _, mem_info = full_he_inference(model, sample_input, HE)
        he_ciphertexts_est = int(mem_info.get("peak_ciphertexts", 0))
        he_ciphertext_mb = float(mem_info.get("est_ciphertext_size_mb", 0.0))
    elif meta.get("he_type") == "selective":
        # Very rough estimate based on sequence length and hidden size.
        seq_len = sample_input.shape[1]
        d_model = getattr(model, "d_model", 64)
        # Assume ~3 ciphertexts per hidden unit for encrypted layers on average.
        he_ciphertexts_est = int(3 * seq_len * d_model)
        he_ciphertext_mb = he_ciphertexts_est / 1024.0  # ~1KB per ct → MB

    elapsed_ms = (t1 - t0) * 1000.0

    result = {
        "elapsed_ms": float(elapsed_ms),
        "peak_ram_mb": float(max(peak_ram_mb, 0.0)),
        "peak_vram_mb": float(max(peak_vram_mb, 0.0)),
        "he_num_ciphertexts_est": int(he_ciphertexts_est),
        "he_ciphertext_memory_mb": float(he_ciphertext_mb),
    }

    print(
        f"[memory_profiling] method={method} "
        f"time={elapsed_ms:.2f}ms "
        f"RAM≈{result['peak_ram_mb']:.2f}MB "
        f"VRAM≈{result['peak_vram_mb']:.2f}MB "
        f"HE_ct≈{he_ciphertexts_est} (~{he_ciphertext_mb:.2f}MB)"
    )

    return result


__all__ = [
    "set_random_seeds",
    "warmup_runs",
    "time_inference",
    "compute_perplexity",
    "memory_profiling",
]

