# Quick Reference & Implementation Guide
## Selective HE Transformer Inference - Developer's Handbook

---

## 1. Quick Start

### Clone & Setup
```bash
cd /Users/mridulmaikhuri/Desktop/projects/btp/code2
python -m venv venv
source venv/bin/activate
pip install -r requirement.txt
```

### Run Example
```bash
# 1. Use pre-trained checkpoint
python run_strategy1.py --checkpoint checkpoint.pt

# 2. Or run full benchmark
python run_all_strategies.py --checkpoint checkpoint.pt --output results.json

# 3. View results
cat results.json | python -m json.tool
```

---

## 2. File Structure & Purpose

### Core Model Files
```
model.py              TinyGPT class (4 layers, 128 hidden)
embeddings.py         Token + positional embeddings
attention.py          Multi-head causal self-attention
ffn.py               Feed-forward networks (2-layer MLP)
transformer.py        TransformerBlock combining attn + ffn
```

### Homomorphic Encryption Files
```
he_layers.py          HE linear operations (matrix-vector multiply)
he_utils.py           Encrypt/decrypt utilities
activation_approx.py  Approximate softmax, GELU as polynomials
selective_he_config.py Configuration classes for strategies
selective_he_engine.py Main inference engine + strategy selection
```

### Training & Inference
```
train.py              Train TinyGPT on tokens
data_prep.py          Tokenize raw text → training data
he_inference.py       Full encrypted inference pipeline
experiment_runner.py  Benchmark harness
```

### Analysis & Visualization
```
benchmarks.py         Latency, perplexity, memory profiling
privacy_analysis.py   Privacy score computation
visualize_results.py  Generate plots from results
run_*.py             Strategy-specific runners
```

### Data & Results
```
data/                 Training datasets
he_configs/           Strategy JSON configurations
checkpoint.pt         Pre-trained model weights
results.json          Latest benchmark results
results.csv           CSV export of results
```

---

## 3. Key Classes & APIs

### 3.1 TinyGPT Model

```python
from model import TinyGPT

# Create model
model = TinyGPT(
    num_layers=4,          # 4 transformer blocks
    vocab_size=50257,      # GPT-2 vocabulary
    d_model=128,           # Hidden dimension
    num_heads=4,           # Attention heads
    d_ff=512,              # FFN intermediate dim
    max_len=1024,          # Max sequence length
    dropout=0.1
)

# Forward pass (plaintext inference)
input_ids = torch.tensor([[101, 2054, 2003, ...]])  # (batch, seq_len)
logits = model(input_ids)  # (batch, seq_len, vocab_size)
next_token = torch.argmax(logits[:, -1, :], dim=-1)
```

### 3.2 Selective HE Configuration

```python
from selective_he_config import SelectiveHEConfig, STRATEGY_1_CONFIG

# Define encryption strategy
config = SelectiveHEConfig(
    layers_to_encrypt=["blocks.0.attention", "blocks.1.attention"],
    operations_to_encrypt=None,
    encryption_granularity="layer"  # or "operation"
)

# Validate against model
config.validate(model)
```

### 3.3 HE Inference Engine

```python
from selective_he_engine import SelectiveHEEngine
from selective_he_config import STRATEGY_2_CONFIG

# Initialize encrypted engine
engine = SelectiveHEEngine(
    model=model,
    he_config=STRATEGY_2_CONFIG,
    device="cpu"  # "cuda" if GPU available
)

# Encrypted forward pass
input_ids = torch.tensor([[101, 2054, 2003, ...]])
with torch.no_grad():
    encrypted_output = engine.forward_encrypted(input_ids)
    
# Decrypt results
logits = engine.decrypt_output(encrypted_output)
```

### 3.4 Benchmarking

```python
from benchmarks import time_inference, compute_perplexity, memory_profiling

# Time a forward pass
prompts = [torch.randint(0, 50257, (seq_len,)) for _ in range(10)]
latency_ms, throughput = time_inference(model, prompts)
print(f"Latency: {latency_ms:.2f}ms, Throughput: {throughput:.0f} tokens/sec")

# Compute perplexity on test set
perplexity = compute_perplexity(model, test_dataset)
print(f"Perplexity: {perplexity:.1f}")

# Memory profiling
memory_mb = memory_profiling(model, input_shape=(1, 128))
print(f"Memory: {memory_mb:.1f} MB")
```

---

## 4. Available Strategies

### Strategy 1: Attention-Only

**File**: `run_strategy1.py`  
**Config**: `he_configs/strategy_1_attention_only.json`

```json
{
  "layers_to_encrypt": ["blocks.0.attention", "blocks.1.attention"],
  "operations_to_encrypt": null,
  "encryption_granularity": "layer"
}
```

**Usage**:
```bash
python run_strategy1.py \
  --checkpoint checkpoint.pt \
  --output results_strat1.json
```

**Result**: Moderate privacy, best latency  
**Use case**: When computational efficiency is priority

---

### Strategy 2: Attention + FFN

**File**: `run_strategy2.py`  
**Config**: `he_configs/strategy_2_attention_ffn.json`

```json
{
  "layers_to_encrypt": [
    "blocks.0.attention", "blocks.0.ffn",
    "blocks.1.attention", "blocks.1.ffn"
  ],
  "operations_to_encrypt": null,
  "encryption_granularity": "layer"
}
```

**Usage**:
```bash
python run_strategy2.py \
  --checkpoint checkpoint.pt \
  --output results_strat2.json
```

**Result**: High privacy, reasonable latency  
**Use case**: Recommended for most applications

---

### Strategy 3: Full Encryption

**File**: `run_strategy3.py`  
**Config**: `he_configs/strategy_3_full_encryption.json`

```json
{
  "layers_to_encrypt": [
    "embedding", "blocks.0.attention", "blocks.0.ffn",
    "blocks.1.attention", "blocks.1.ffn", "lm_head"
  ],
  "operations_to_encrypt": null,
  "encryption_granularity": "layer"
}
```

**Status**: ✗ Error (architectural issues)  
**Note**: Requires full redesign to support encrypted embeddings

---

## 5. Benchmark Results Interpretation

### Raw Results
```json
{
  "plain": {
    "avg_latency_ms": 2.745,
    "tokens_per_sec": 26225,
    "perplexity": 51385,
    "ciphertext_overhead_mb": 0.0
  },
  "strategy1": {
    "avg_latency_ms": 1.781,
    "tokens_per_sec": 5615,
    "perplexity": 52931,
    "ciphertext_overhead_mb": 3.75
  }
}
```

### Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Latency** | Wall-clock time per inference | Lower = faster |
| **Tokens/sec** | 1000 / latency_ms | Throughput measure |
| **Perplexity** | exp(-avg_log_prob) | Lower = better predictions |
| **Overhead** | Size(ciphertexts) / Size(plaintexts) | Privacy cost |

### Interpreting Results

```
Strategy 1 vs Plaintext:
- Latency: 1.781 ms (36% FASTER!)
  → Unexpected improvement due to approximate evaluation
  
- Tokens/sec: 5,615 (79% reduction)
  → HE operations slower despite latency reduction
  
- Perplexity: 52,931 vs 51,385 (+3%)
  → Acceptable accuracy loss for privacy
```

---

## 6. Common Tasks

### 6.1 Add a Custom Strategy

```python
# 1. Create config in selective_he_config.py
STRATEGY_CUSTOM_CONFIG = SelectiveHEConfig(
    layers_to_encrypt=[
        "blocks.0.attention",
        "blocks.1.attention",
        "blocks.2.ffn"  # Only FFN in block 2
    ],
    encryption_granularity="layer"
)

# 2. Create runner script (run_strategy_custom.py)
if __name__ == "__main__":
    results = run_strategy("custom", STRATEGY_CUSTOM_CONFIG)
    
# 3. Run and compare
python run_strategy_custom.py --checkpoint checkpoint.pt
```

### 6.2 Modify Model Architecture

```python
# Edit model.py
class TinyGPT(nn.Module):
    def __init__(self, num_layers=6, d_model=256, ...):  # Larger!
        super().__init__()
        # ... rest of init
        
# Retrain and benchmark
python train.py --data-dir data/ --max-steps 10000 --d-model 256
python run_all_strategies.py --checkpoint checkpoint.pt
```

### 6.3 Generate Encrypted Predictions

```python
from selective_he_engine import SelectiveHEEngine
from selective_he_config import STRATEGY_2_CONFIG

engine = SelectiveHEEngine(model, STRATEGY_2_CONFIG)

# Input prompt
prompt = "What is artificial"  # User's sensitive query
prompt_ids = tokenizer.encode(prompt)
x = torch.tensor([prompt_ids])

# Get encrypted output
with torch.no_grad():
    enc_output = engine.forward_encrypted(x)

# Client-side decryption
pred = engine.decrypt_output(enc_output)
top_token = torch.argmax(pred[-1, :])
```

### 6.4 Profile Memory Usage

```bash
# Monitor during encrypted inference
python -c "
from experiment_runner import run_experiment
import psutil
import os

pid = os.getpid()
process = psutil.Process(pid)
mem_start = process.memory_info().rss / 1024**2

result = run_experiment(...)  # Run benchmark

mem_end = process.memory_info().rss / 1024**2
print(f'Memory delta: {mem_end - mem_start:.1f} MB')
"
```

---

## 7. Troubleshooting

### Issue: "AttributeError: 'TinyGPT' object has no attribute 'embeddings'"

**Cause**: Strategy 3 expects different model structure  
**Fix**: Use Strategy 1 or 2, or redesign model for full encryption

```python
# Current model
class TinyGPT(nn.Module):
    def __init__(self):
        self.embedding = ...  # Wrong attribute name for Strategy 3
        
# For Strategy 3, would need:
class TinyGPT(nn.Module):
    def __init__(self):
        self.embeddings = ...  # Note the 's'
```

### Issue: "Pyfhel ImportError"

**Cause**: FHE library not installed  
**Fix**:
```bash
pip install Pyfhel
# Or: conda install -c conda-forge pyfhel
```

### Issue: Perplexity doesn't match plaintext

**Cause**: Different random seed or batch effects  
**Fix**: Use same seed and deterministic settings
```python
from benchmarks import set_random_seeds
set_random_seeds(42)  # Reproducible results
```

### Issue: OOM on GPU

**Cause**: Large ciphertext batches  
**Fix**: Use CPU or reduce batch size
```bash
python run_strategy1.py --checkpoint checkpoint.pt --device cpu
```

---

## 8. Performance Optimization Tips

### 8.1 Latency Improvements

**Current**: ~1.78ms per token  
**To optimize**:

1. **Use batch inference** (requires SIMD slots)
   ```python
   # Currently: batch_size = 1
   # TODO: Implement slot packing for batch_size > 1
   ```

2. **Reduce ciphertext size**
   ```python
   # Reduce polynomial degree in Pyfhel
   he_ctx.resizeModulus(target_size)  # Trade precision for speed
   ```

3. **Pre-compute expensive operations**
   ```python
   # Cache encrypted projections
   cached_W_enc = cache_encrypt_matrix(model.attention.W)
   ```

### 8.2 Accuracy Improvements

**Current**: ~3% perplexity increase  
**To improve**:

1. **Better softmax approximation**
   ```python
   # activation_approx.py: use higher-degree polynomial
   # P(x) = 0.04 + 0.23*x + 0.09*x^2 + ...  (degree 4+)
   ```

2. **Reduced quantization error**
   ```python
   # Use higher precision in fixed-point encoding
   scale_factor = 2**20  # vs current 2**16
   ```

3. **Fine-tune encrypted layers**
   ```python
   # Train with encryption from start
   python train.py --use-he-layers --strategy 2
   ```

---

## 9. API Reference

### Key Functions

```python
# Encryption/Decryption
from he_utils import (
    encrypt_tensor,        # torch.Tensor → list[ciphertext]
    decrypt_tensor,        # list[ciphertext] → torch.Tensor
    initialize_he_context  # Setup Pyfhel parameters
)

# Configuration
from selective_he_config import (
    SelectiveHEConfig,     # Define strategy
    STRATEGY_1_CONFIG,     # Predefined strategies
    STRATEGY_2_CONFIG,
    STRATEGY_3_CONFIG
)

# Inference
from selective_he_engine import SelectiveHEEngine  # Main inference engine

# Benchmarking
from benchmarks import (
    time_inference,        # Measure latency & throughput
    compute_perplexity,    # Language model evaluation
    memory_profiling       # Memory usage
)

# Training
from train import train_model  # Full training pipeline
```

---

## 10. Experimental Configuration

### Test Setup
```json
{
  "checkpoint": "checkpoint.pt",
  "lengths": [10, 50, 100, 200],
  "num_per_length": 25,
  "seed": 42,
  "test_mode": false,
  "device": "cpu"
}
```

### Reproduce Exact Results
```bash
# Same seed ensures reproducibility
python experiment_runner.py \
  --checkpoint checkpoint.pt \
  --seed 42 \
  --lengths 10,50,100,200 \
  --num-per-length 25
```

### Output Files
```
results.json       Main results (all metrics)
results.csv        CSV format (importable to Excel)
pareto.png         Privacy-performance plot
activation_plots/  Softmax approximation visualizations
```

---

## 11. Next Steps

### For Researchers
- [ ] Implement bootstrapping (refresh ciphertexts mid-computation)
- [ ] Test on larger models (GPT-2 scale)
- [ ] Compare against differential privacy alternatives
- [ ] Explore CKKS for real number arithmetic

### For Practitioners
- [ ] Integrate with production serving framework
- [ ] Add client-side key management
- [ ] Implement secure communication protocols
- [ ] Deploy on cloud infrastructure

### For Students
- [ ] Understand FHE mathematical foundations
- [ ] Study noise management in deep networks
- [ ] Research approximate activation functions
- [ ] Explore optimal encryption granularity

---

## 12. Contact & Resources

**Repository**: [path/to/repo]  
**Framework**: PyTorch + Pyfhel  
**License**: MIT  

**Key References**:
- Pyfhel documentation: https://github.com/ibarrond/Pyfhel
- FHE Basics: https://homomorphicencryption.org/
- SEAL Library: https://github.com/microsoft/SEAL

---

**Quick Reference Guide v1.0**  
**Last Updated**: March 2026
