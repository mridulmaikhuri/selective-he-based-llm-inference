# API Reference

Complete listing of all public functions and classes with signatures, descriptions, and usage examples.

---

## Table of Contents

1. [Model Components](#model-components)
2. [Transformer Blocks](#transformer-blocks)
3. [Homomorphic Encryption](#homomorphic-encryption)
4. [Configuration & Strategy](#configuration--strategy)
5. [Activation Approximations](#activation-approximations)
6. [Data Preparation](#data-preparation)
7. [Benchmarking](#benchmarking)
8. [Training](#training)

---

## Model Components

### `TokenPositionalEmbedding`

**Signature:**
```python
class TokenPositionalEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 128,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ) -> None
```

**Description:**
Converts token IDs to embeddings by summing learned token embeddings with learned positional embeddings. Applies dropout.

**Parameters:**
- `vocab_size`: Vocabulary size (number of unique tokens)
- `d_model`: Embedding dimension
- `max_seq_len`: Maximum sequence length supported
- `dropout`: Dropout probability applied to embeddings

**Returns:** Module that outputs shape `(batch, seq_len, d_model)`

**Example:**
```python
import torch
from model import TokenPositionalEmbedding

# Create embedding layer
embed = TokenPositionalEmbedding(vocab_size=50257, d_model=128)

# Token IDs: shape (batch=2, seq_len=5)
input_ids = torch.tensor([[101, 2054, 2003, 2017, 102],
                          [101, 1045, 2572, 102, 0]])

# Forward pass
embeddings = embed(input_ids)  # Shape: (2, 5, 128)

print(embeddings.shape)  # torch.Size([2, 5, 128])
```

**Methods:**
- `forward(input_ids: torch.Tensor) → torch.Tensor`: Main forward pass

---

### `TinyGPT`

**Signature:**
```python
class TinyGPT(nn.Module):
    def __init__(
        self,
        num_layers: int = 4,
        vocab_size: int = 50257,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1024
    ) -> None
```

**Description:**
Minimal GPT-style transformer language model. Computes token embeddings → N transformer blocks → layer norm → lm_head projection. Weight tying: lm_head shares weights with token embedding.

**Parameters:**
- `num_layers`: Number of transformer blocks
- `vocab_size`: Vocabulary size
- `d_model`: Hidden dimension
- `num_heads`: Number of attention heads
- `d_ff`: Feed-forward hidden dimension (typically 4×d_model)
- `dropout`: Dropout probability
- `max_seq_len`: Maximum sequence length

**Returns:** Shape `(batch, seq_len, vocab_size)` logits

**Example:**
```python
import torch
from model import TinyGPT

# Create model
model = TinyGPT(
    num_layers=4,
    vocab_size=50257,
    d_model=128,
    num_heads=4,
    d_ff=512,
    dropout=0.1
)

# Input tokens
input_ids = torch.tensor([[101, 2054, 2003]])  # Shape: (1, 3)

# Forward pass
logits = model(input_ids)  # Shape: (1, 3, 50257)

# Get next token predictions
next_token_logits = logits[:, -1, :]  # Last token logits
next_tokens = torch.argmax(next_token_logits, dim=-1)  # Most likely tokens
print(next_tokens)  # tensor([12345])
```

**Methods:**
- `forward(input_ids: torch.Tensor, attn_mask: torch.Tensor | None = None) → torch.Tensor`
- `count_parameters() → int`: Count trainable parameters (accounts for weight tying)

---

## Transformer Blocks

### `CausalSelfAttention`

**Signature:**
```python
class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1
    ) -> None
```

**Description:**
Multi-head causal (masked) self-attention. Prevents future tokens from attending to past tokens via additive causal mask. Uses fused QKV projection for efficiency.

**Parameters:**
- `d_model`: Total embedding dimension (must be divisible by num_heads)
- `num_heads`: Number of parallel attention heads
- `dropout`: Attention weight dropout probability

**Returns:** Shape `(batch, seq_len, d_model)`

**Example:**
```python
import torch
from transformer_block import CausalSelfAttention

# Create attention layer
attn = CausalSelfAttention(d_model=128, num_heads=4, dropout=0.1)

# Input: batch_size=2, seq_len=5, d_model=128
x = torch.randn(2, 5, 128)

# Forward pass
output = attn(x)  # Shape: (2, 5, 128)

# Optional: provide padding mask
padding_mask = torch.tensor([[1, 1, 1, 0, 0],
                             [1, 1, 1, 1, 0]])  # 1=attend, 0=ignore
# Converts to additive mask internally
output_masked = attn(x, attn_mask=padding_mask)
```

**Methods:**
- `forward(x: torch.Tensor, attn_mask: torch.Tensor | None = None) → torch.Tensor`

---

### `FeedForward`

**Signature:**
```python
class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        d_ff: int = 512,
        dropout: float = 0.1
    ) -> None
```

**Description:**
Position-wise feed-forward network: Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model). Applied identically to each token position.

**Parameters:**
- `d_model`: Input/output dimension
- `d_ff`: Hidden dimension (expansion factor)
- `dropout`: Dropout probability

**Returns:** Shape `(batch, seq_len, d_model)`

**Example:**
```python
import torch
from transformer_block import FeedForward

# Create FFN layer
ffn = FeedForward(d_model=128, d_ff=512, dropout=0.1)

# Input: any shape ending in d_model
x = torch.randn(2, 5, 128)

# Forward pass
output = ffn(x)  # Shape: (2, 5, 128)

# Can be applied in-place to sequences of any length
x_long = torch.randn(2, 100, 128)
output_long = ffn(x_long)  # Shape: (2, 100, 128)
```

**Methods:**
- `forward(x: torch.Tensor) → torch.Tensor`

---

### `TransformerBlock`

**Signature:**
```python
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1
    ) -> None
```

**Description:**
Pre-LayerNorm transformer block: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual. This is the standard building block used in TinyGPT.

**Parameters:**
- `d_model`: Hidden dimension
- `num_heads`: Number of attention heads
- `d_ff`: Feed-forward hidden dimension
- `dropout`: Dropout probability

**Returns:** Shape `(batch, seq_len, d_model)`

**Example:**
```python
import torch
from transformer_block import TransformerBlock

# Create block
block = TransformerBlock(d_model=128, num_heads=4, d_ff=512, dropout=0.1)

# Input
x = torch.randn(2, 5, 128)

# Forward pass
output = block(x)  # Shape: (2, 5, 128)

# Stack multiple blocks for a transformer
blocks = torch.nn.ModuleList([
    TransformerBlock(d_model=128, num_heads=4) for _ in range(4)
])

x = torch.randn(2, 10, 128)
for block in blocks:
    x = block(x)  # Sequentially apply blocks
```

**Methods:**
- `forward(x: torch.Tensor, attn_mask: torch.Tensor | None = None) → torch.Tensor`

---

## Homomorphic Encryption

### `setup_HE_context`

**Signature:**
```python
def setup_HE_context(
    n: int = 2**13,
    t: int = 65537
) -> Pyfhel
```

**Description:**
Initialize Pyfhel BFV context for homomorphic encryption. Generates public/secret keys and sets up the encryption scheme.

**Parameters:**
- `n`: Polynomial degree (BFV parameter). Larger = more security but slower. Common: 2^12, 2^13, 2^14
- `t`: Plaintext modulus (integers mod t). Larger = larger number space. Common: 65537 (16-bit prime)

**Returns:** Initialized Pyfhel object ready for encryption/decryption

**Raises:**
- `ImportError`: If Pyfhel not installed

**Example:**
```python
from he_utils import setup_HE_context
import torch

# Initialize HE context
context = setup_HE_context(n=2**13, t=65537)

# Use context for encryption
x = torch.tensor([100, 200, 300])
x_enc = encrypt_tensor(x, context)  # Ciphertexts ready for HE operations

# Decrypt
x_dec = decrypt_tensor(x_enc, context)  # Recover plaintext
print(x_dec)  # tensor([100, 200, 300], dtype=torch.int64)
```

---

### `encrypt_tensor`

**Signature:**
```python
def encrypt_tensor(
    tensor: torch.Tensor,
    HE: Pyfhel,
    batch: bool = False
) -> list[PyCtxt]
```

**Description:**
Encrypt a PyTorch tensor element-wise using BFV homomorphic encryption. Each scalar becomes a separate ciphertext (~48 KB each for n=2^13).

**Parameters:**
- `tensor`: Input tensor (float32 or int64). Will be rounded to integers.
- `HE`: Pyfhel context from `setup_HE_context()`
- `batch`: If True, use SIMD batching (experimental)

**Returns:** List of PyCtxt (ciphertexts), one per scalar element

**Example:**
```python
from he_utils import setup_HE_context, encrypt_tensor
import torch

context = setup_HE_context()

# Single vector
x = torch.tensor([1.5, -2.3, 0.8])  # Floats are rounded to ints
x_enc = encrypt_tensor(x, context)  # List of 3 ciphertexts

print(len(x_enc))  # 3

# 2D tensor
W = torch.randn(128, 64)  # Will be flattened
W_enc = encrypt_tensor(W, context)  # List of 8192 ciphertexts
```

---

### `decrypt_tensor`

**Signature:**
```python
def decrypt_tensor(
    ciphertexts: list[PyCtxt],
    HE: Pyfhel
) -> torch.Tensor
```

**Description:**
Decrypt a list of ciphertexts back to integer tensor. Outputs int64 (may contain rounding errors from prior integer conversions).

**Parameters:**
- `ciphertexts`: List of PyCtxt objects from `encrypt_tensor()` or HE computation
- `HE`: Pyfhel context

**Returns:** int64 tensor matching the original encrypted shape

**Example:**
```python
from he_utils import setup_HE_context, encrypt_tensor, decrypt_tensor
import torch

context = setup_HE_context()

# Encrypt
x = torch.tensor([100, 200, 300])
x_enc = encrypt_tensor(x, context)

# Do some HE operations (e.g., he_linear)
# ...

# Decrypt
x_recovered = decrypt_tensor(x_enc, context)
print(x_recovered)  # tensor([100, 200, 300], dtype=torch.int64)
```

---

### `he_linear`

**Signature:**
```python
def he_linear(
    encrypted_input: "_TaggedList",
    plain_weights: Union[np.ndarray, torch.Tensor],
    plain_bias: Union[np.ndarray, torch.Tensor],
    HE: Pyfhel
) -> "_TaggedList"
```

**Description:**
Compute matrix-vector multiplication entirely in encrypted domain: `y_enc = x_enc @ W_plain + b_plain`. Uses plaintext weights for efficiency (client-server model). One ciphertext × plaintext multiplication per output neuron.

**Parameters:**
- `encrypted_input`: Encrypted input vector(s). Shape: (1, in_features)
- `plain_weights`: Plaintext weight matrix. Shape: (in_features, out_features)
- `plain_bias`: Plaintext bias vector. Shape: (out_features,). Can be None.
- `HE`: Pyfhel context

**Returns:** Encrypted output vector(s). Shape: (1, out_features)

**Example:**
```python
from he_utils import setup_HE_context, encrypt_tensor, decrypt_tensor
from he_layers import he_linear
import torch
import numpy as np

context = setup_HE_context()

# Plaintext weights
W = torch.randn(128, 64)  # in_features=128, out_features=64
b = torch.randn(64)

# Input (plaintext)
x = torch.randn(1, 128)

# Encrypt input
x_enc = encrypt_tensor(x.int(), context)

# HE computation
y_enc = he_linear(x_enc, W.numpy(), b.numpy(), context)

# Decrypt output
y_recovered = decrypt_tensor(y_enc, context).float()

# Compare with plaintext computation
y_plain = x @ W + b

print(f"Recovered: {y_recovered[:5]}")
print(f"Expected:  {y_plain[0, :5]}")
```

---

### `selective_HE_inference`

**Signature:**
```python
def selective_HE_inference(
    model: nn.Module,
    input_ids: torch.Tensor,
    HE_context: Pyfhel | None,
    config: SelectiveHEConfig
) -> tuple[torch.Tensor, dict[str, float], list[tuple[str, bool]]]
```

**Description:**
Run mixed plaintext/homomorphic encryption inference through a model. Routes each layer to either plaintext forward or HE computation based on configuration.

**Parameters:**
- `model`: PyTorch model to infer through
- `input_ids`: Input token IDs. Shape: (batch_size, seq_len). Typically batch_size=1 for HE.
- `HE_context`: Pyfhel context from `setup_HE_context()`. If None, all compute is plaintext.
- `config`: SelectiveHEConfig specifying which layers to encrypt

**Returns:** Tuple of:
- `logits`: Model output. Shape: (batch_size, seq_len, vocab_size)
- `timing_dict`: Per-phase timing in seconds: keys are "encryption_time", "he_compute_time", "decryption_time", "plaintext_time"
- `encryption_log`: List of (layer_name, was_encrypted) tuples

**Raises:**
- `ConfigError`: If config requests encryption of non-Linear layer
- `PyfhelUnavailableError`: If Pyfhel not installed but encryption requested

**Example:**
```python
import torch
from model import TinyGPT
from selective_he_config import SelectiveHEConfig
from selective_he_engine import selective_HE_inference
from he_utils import setup_HE_context

# Load model
model = TinyGPT(num_layers=4, vocab_size=50257, d_model=128)
model.eval()

# Setup HE
he_context = setup_HE_context()

# Configure encryption: encrypt attention outputs only (Strategy 1)
config = SelectiveHEConfig(
    layers_to_encrypt=[
        "transformer_blocks.0.attention.out_proj",
        "transformer_blocks.1.attention.out_proj",
        "transformer_blocks.2.attention.out_proj",
        "transformer_blocks.3.attention.out_proj",
    ],
    weight_scale=100
)

# Input
input_ids = torch.tensor([[101, 2054, 2003]])  # 3 tokens

# Run selective HE inference
logits, timing, enc_log = selective_HE_inference(
    model=model,
    input_ids=input_ids,
    HE_context=he_context,
    config=config
)

print(f"Logits shape: {logits.shape}")  # (1, 3, 50257)
print(f"Total HE time: {timing['he_compute_time']:.2f}s")

# Show which layers were encrypted
for layer_name, was_encrypted in enc_log:
    status = "✓ ENCRYPTED" if was_encrypted else "plaintext"
    print(f"  {layer_name}: {status}")
```

---

## Configuration & Strategy

### `SelectiveHEConfig`

**Signature:**
```python
@dataclass
class SelectiveHEConfig:
    layers_to_encrypt: list[str] = field(default_factory=list)
    weight_scale: int = 1
    bias_scale: int = 1
    input_scale: int = 1
    batch_size_check: bool = True
```

**Description:**
Configuration for selective homomorphic encryption. Specifies which layers to encrypt, scaling factors for integer conversion, and validation options.

**Attributes:**
- `layers_to_encrypt`: List of layer names (as returned by `model.named_modules()`) to encrypt. Only `nn.Linear` layers supported.
- `weight_scale`: Integer scale factor applied to weights before HE. Helps with BFV integer arithmetic. Example: `weight_scale=100` means multiply weights by 100 before encryption.
- `bias_scale`: Scale factor for bias (rarely different from weight_scale)
- `input_scale`: Scale factor for input activations
- `batch_size_check`: If True, raises error if batch_size > 1 (element-wise encryption only supports batch_size=1)

**Example:**
```python
from selective_he_config import SelectiveHEConfig

# Strategy 1: Encrypt attention output projections only
strategy1 = SelectiveHEConfig(
    layers_to_encrypt=[
        "transformer_blocks.0.attention.out_proj",
        "transformer_blocks.1.attention.out_proj",
        "transformer_blocks.2.attention.out_proj",
        "transformer_blocks.3.attention.out_proj",
    ],
    weight_scale=100
)

# Strategy 2: Encrypt entire last block + LM head
strategy2 = SelectiveHEConfig(
    layers_to_encrypt=[
        "transformer_blocks.3.attention.qkv_proj",
        "transformer_blocks.3.attention.out_proj",
        "transformer_blocks.3.ffn.fc1",
        "transformer_blocks.3.ffn.fc2",
        "lm_head",
    ],
    weight_scale=100
)

# Strategy 3: Encrypt embedding + LM head
strategy3 = SelectiveHEConfig(
    layers_to_encrypt=[
        "token_embedding",  # Actually handled specially; uses embedding lookup
        "lm_head",
    ],
    weight_scale=100
)
```

**Methods:**
- `validate(model: nn.Module) → None`: Check config against actual model
- `to_dict() → dict`: Serialize to dictionary
- `to_json(path: str | Path) → None`: Save to JSON file
- `from_dict(data: dict) → SelectiveHEConfig`: Load from dictionary
- `from_json(path: str | Path) → SelectiveHEConfig`: Load from JSON file
- `summary() → str`: Return human-readable encryption plan

---

## Activation Approximations

### `gelu_approx`

**Signature:**
```python
def gelu_approx(
    x: torch.Tensor,
    degree: int = 5
) -> torch.Tensor
```

**Description:**
Polynomial approximation to GELU activation function using pre-fitted Chebyshev coefficients. Degree 5 approximation is ~0.5% MSE on typical ranges.

**Parameters:**
- `x`: Input tensor (values should be in range [-5, 5])
- `degree`: Polynomial degree. Options: 3, 5, 7 (higher = more accurate but more computation)

**Returns:** Approximated GELU output, same shape as input

**Example:**
```python
import torch
from activation_approx import gelu_approx

# Input
x = torch.randn(2, 128)  # Typical hidden layer activations

# Approximate GELU (instead of torch.nn.functional.gelu)
y_approx = gelu_approx(x, degree=5)

# For comparison
y_true = torch.nn.functional.gelu(x)

# Compute error
mse = ((y_approx - y_true) ** 2).mean()
print(f"MSE: {mse:.6f}")  # Should be ~0.001 or better
```

---

### `softmax_approx`

**Signature:**
```python
def softmax_approx(
    x: torch.Tensor,
    degree: int = 5,
    dim: int = -1
) -> torch.Tensor
```

**Description:**
Numerically stable softmax approximation via polynomial approximation to exp(). Useful for HE where only polynomial operations are efficient.

**Parameters:**
- `x`: Input logits
- `degree`: Polynomial degree for exp approximation (3, 5, or 7)
- `dim`: Dimension to compute softmax over

**Returns:** Approximate softmax probabilities, same shape as input, sums to 1 over `dim`

**Example:**
```python
import torch
from activation_approx import softmax_approx

# Attention logits
logits = torch.randn(2, 4, 8)  # (batch, heads, seq_len)

# Approximate softmax
probs_approx = softmax_approx(logits, degree=5, dim=-1)

# True softmax
probs_true = torch.softmax(logits, dim=-1)

# Check approximation quality
error = (probs_approx - probs_true).abs().max()
print(f"Max error: {error:.6f}")
```

---

### `compute_mse_gelu` / `compute_mse_softmax`

**Signature:**
```python
def compute_mse_gelu(degrees: List[int] = [3, 5, 7]) -> Dict[int, float]

def compute_mse_softmax(
    degrees: List[int] = [3, 5, 7],
    num_samples: int = 1000,
    dim: int = 4
) -> Dict[int, float]
```

**Description:**
Compute mean squared error between true activation and polynomial approximation over typical input ranges.

**Returns:** Dictionary mapping degree → MSE

**Example:**
```python
from activation_approx import compute_mse_gelu, compute_mse_softmax

# Check approximation quality
gelu_errors = compute_mse_gelu(degrees=[3, 5, 7])
softmax_errors = compute_mse_softmax(degrees=[3, 5, 7])

print("GELU MSE:")
for degree, mse in gelu_errors.items():
    print(f"  Degree {degree}: {mse:.6f}")

print("Softmax MSE:")
for degree, mse in softmax_errors.items():
    print(f"  Degree {degree}: {mse:.6f}")
```

---

## Data Preparation

### `import_dependencies`

**Signature:**
```python
def import_dependencies() -> dict
```

**Description:**
Import heavy dependencies with friendly error messages. Abstracts away version-specific imports.

**Returns:** Dictionary of imported modules

**Example:**
```python
from data_prep import import_dependencies

deps = import_dependencies()
# deps['torch'], deps['datasets'], deps['transformers'], etc.
```

---

### `chunk_sequences`

**Signature:**
```python
def chunk_sequences(
    token_ids: list[int],
    seq_len: int,
    max_seqs: int | None = None
) -> list[list[int]]
```

**Description:**
Split flat list of token IDs into fixed-length non-overlapping sequences.

**Parameters:**
- `token_ids`: Flat list of token IDs
- `seq_len`: Target sequence length (e.g., 128)
- `max_seqs`: Maximum number of sequences to return (None = no limit)

**Returns:** List of sequences, each of length `seq_len`

**Example:**
```python
from data_prep import chunk_sequences

# Flat token list (e.g., from corpus)
tokens = list(range(1000))  # [0, 1, 2, ..., 999]

# Chunk into sequences of length 128
sequences = chunk_sequences(tokens, seq_len=128)

print(f"Number of sequences: {len(sequences)}")  # ~7
print(f"Sequence 0 length: {len(sequences[0])}")  # 128
print(f"Sequence 0: {sequences[0][:10]}")  # [0, 1, 2, ..., 9]
```

---

### `load_tokenizer`

**Signature:**
```python
def load_tokenizer(
    GPT2TokenizerFast,
    out_dir: Path
) -> GPT2TokenizerFast
```

**Description:**
Load GPT-2 tokenizer and cache it in `out_dir` for reproducibility.

**Example:**
```python
from pathlib import Path
from data_prep import load_tokenizer
from transformers import GPT2TokenizerFast

tokenizer = load_tokenizer(GPT2TokenizerFast, Path("./data"))

# Use tokenizer
text = "Hello, world!"
tokens = tokenizer.encode(text)
print(tokens)  # [Hello, ,, world, !] as token IDs
```

---

## Benchmarking

### `time_inference`

**Signature:**
```python
def time_inference(
    model: nn.Module,
    inputs: torch.Tensor,
    method: str = "plain",
    n_runs: int = 10,
    include_warmup: bool = True,
    HE: Pyfhel | None = None,
    inference_fn: Callable | None = None,
    verbose: bool = True
) -> Dict[str, Any]
```

**Description:**
Measure inference latency and throughput over multiple runs.

**Parameters:**
- `model`: Model to benchmark
- `inputs`: Input tensor (typically token IDs)
- `method`: Inference method ("plain", "he", or other)
- `n_runs`: Number of inference runs to average
- `include_warmup`: If True, run warm-up iterations first
- `HE`: Pyfhel context if using HE
- `inference_fn`: Custom inference function (if None, uses `model.forward()`)
- `verbose`: Print timing summary

**Returns:** Dictionary with keys:
- `"mean_latency_ms"`: Average latency per run
- `"std_latency_ms"`: Standard deviation
- `"min_latency_ms"`, `"max_latency_ms"`: Min/max
- `"throughput_samples_per_sec"`: Batch throughput

**Example:**
```python
import torch
from model import TinyGPT
from benchmarks import time_inference

model = TinyGPT(num_layers=4, vocab_size=50257, d_model=128)
model.eval()

inputs = torch.tensor([[101, 2054, 2003]])  # 3 tokens

# Benchmark plaintext inference
results_plain = time_inference(
    model=model,
    inputs=inputs,
    method="plain",
    n_runs=10,
    verbose=True
)

print(f"Latency: {results_plain['mean_latency_ms']:.2f} ± {results_plain['std_latency_ms']:.2f} ms")
print(f"Throughput: {results_plain['throughput_samples_per_sec']:.2f} samples/sec")
```

---

### `compute_perplexity`

**Signature:**
```python
def compute_perplexity(
    model: nn.Module,
    val_dataset: List[torch.Tensor] | torch.Tensor,
    method: str = "plain",
    HE: Pyfhel | None = None,
    inference_fn: Callable | None = None,
    batch_size: int = 1,
    verbose: bool = True
) -> float
```

**Description:**
Compute perplexity (exp of mean cross-entropy loss) on validation dataset.

**Parameters:**
- `model`: Language model
- `val_dataset`: List of token tensors or single tensor
- `method`: Inference method
- `HE`: Pyfhel context if using HE
- `inference_fn`: Custom inference function
- `batch_size`: Batch size (1 recommended for HE)
- `verbose`: Print progress

**Returns:** Perplexity score (lower is better)

**Example:**
```python
import torch
from model import TinyGPT
from benchmarks import compute_perplexity

model = TinyGPT(num_layers=4)
model.eval()

# Validation data: list of token sequences
val_data = [
    torch.randint(0, 50257, (1, 128)),
    torch.randint(0, 50257, (1, 128)),
    torch.randint(0, 50257, (1, 128)),
]

# Compute perplexity
ppl = compute_perplexity(
    model=model,
    val_dataset=val_data,
    method="plain",
    verbose=True
)

print(f"Validation perplexity: {ppl:.2f}")
```

---

### `memory_profiling`

**Signature:**
```python
def memory_profiling(
    model: nn.Module,
    method: str = "plain",
    HE: Pyfhel | None = None,
    sample_input: torch.Tensor | None = None,
    verbose: bool = True
) -> Dict[str, float]
```

**Description:**
Profile peak RAM/VRAM usage and estimate ciphertext memory for HE.

**Returns:** Dictionary with memory metrics (keys vary by method)

**Example:**
```python
from model import TinyGPT
from benchmarks import memory_profiling

model = TinyGPT(num_layers=4, d_model=128)

# Memory profile
mem_stats = memory_profiling(
    model=model,
    method="plain",
    sample_input=torch.tensor([[101, 2054, 2003]]),
    verbose=True
)

for key, val in mem_stats.items():
    print(f"{key}: {val:.2f} MB")
```

---

## Training

### `TokenDataset`

**Signature:**
```python
class TokenDataset(Dataset):
    def __init__(self, data: torch.Tensor, seq_len: int = 128) -> None
```

**Description:**
PyTorch Dataset wrapper for token sequences. Converts flat token tensor into (input, target) pairs for next-token prediction.

**Parameters:**
- `data`: Token tensor of shape (total_tokens,)
- `seq_len`: Sequence length for each sample

**Example:**
```python
import torch
from torch.utils.data import DataLoader
from train import TokenDataset

# Token data
tokens = torch.randint(0, 50257, (100000,))  # 100k tokens

# Create dataset
dataset = TokenDataset(data=tokens, seq_len=128)

print(f"Dataset size: {len(dataset)}")  # ~781 samples

# Get a sample
input_ids, target_ids = dataset[0]
print(f"Input shape: {input_ids.shape}")  # (128,)
print(f"Target shape: {target_ids.shape}")  # (128,)

# Create DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
for input_ids, target_ids in loader:
    print(input_ids.shape)  # (batch_size, 128)
    break
```

**Methods:**
- `__len__() → int`: Number of samples
- `__getitem__(idx: int) → tuple[torch.Tensor, torch.Tensor]`: Return (input, target) pair

---

### `evaluate`

**Signature:**
```python
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
    max_batches: int = 50
) -> tuple[float, float]
```

**Description:**
Compute validation loss and perplexity over a DataLoader without gradient computation.

**Parameters:**
- `model`: Language model in eval mode
- `loader`: Validation DataLoader
- `device`: Device (cpu, cuda, etc.)
- `use_fp16`: Use mixed precision (float16)
- `max_batches`: Maximum batches to evaluate

**Returns:** Tuple of (mean_loss, perplexity)

**Example:**
```python
import torch
from torch.utils.data import DataLoader
from train import evaluate, TokenDataset

model = TinyGPT()
model.eval()

val_tokens = torch.randint(0, 50257, (10000,))
val_dataset = TokenDataset(val_tokens, seq_len=128)
val_loader = DataLoader(val_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss, ppl = evaluate(
    model=model,
    loader=val_loader,
    device=device,
    use_fp16=False,
    max_batches=50
)

print(f"Validation loss: {loss:.4f}")
print(f"Validation perplexity: {ppl:.2f}")
```

---

### `save_checkpoint`

**Signature:**
```python
def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler_state: dict,
    step: int,
    best_val_loss: float
) -> None
```

**Description:**
Save a training checkpoint (model, optimizer, scheduler state, step, best loss).

**Example:**
```python
import torch
from pathlib import Path
from train import save_checkpoint

model = TinyGPT()
optimizer = torch.optim.AdamW(model.parameters())
scheduler_state = {"lr_scheduler": "cosine", "step": 1000}

save_checkpoint(
    path=Path("checkpoint_1000.pt"),
    model=model,
    optimizer=optimizer,
    scheduler_state=scheduler_state,
    step=1000,
    best_val_loss=2.45
)

# Load later
checkpoint = torch.load("checkpoint_1000.pt")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
step = checkpoint["step"]
```

---

### `get_lr`

**Signature:**
```python
def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    lr: float
) -> float
```

**Description:**
Compute learning rate at a given step using warmup → cosine decay schedule.

**Example:**
```python
from train import get_lr
import matplotlib.pyplot as plt

# Learning rate schedule
lrs = []
for step in range(10000):
    lr = get_lr(step, warmup_steps=500, max_steps=10000, lr=1e-3)
    lrs.append(lr)

plt.plot(lrs)
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Warmup → Cosine Decay Schedule")
plt.show()
```

---

## Summary Table

| Module | Function/Class | Purpose |
|--------|----------------|---------|
| **Model** | TinyGPT | Main language model |
| **Model** | TokenPositionalEmbedding | Token + positional embeddings |
| **Transformer** | CausalSelfAttention | Multi-head attention with causal mask |
| **Transformer** | FeedForward | 2-layer MLP with GELU |
| **Transformer** | TransformerBlock | Pre-LN block: attention + FFN + residuals |
| **HE** | setup_HE_context | Initialize Pyfhel BFV scheme |
| **HE** | encrypt_tensor | Encrypt tensor element-wise |
| **HE** | decrypt_tensor | Decrypt ciphertexts to tensor |
| **HE** | he_linear | HE matrix-vector multiplication |
| **HE** | selective_HE_inference | Mixed plaintext/HE inference |
| **Config** | SelectiveHEConfig | Encryption strategy configuration |
| **Activation** | gelu_approx | Polynomial GELU approximation |
| **Activation** | softmax_approx | Polynomial softmax approximation |
| **Activation** | compute_mse_gelu | Measure GELU approximation error |
| **Activation** | compute_mse_softmax | Measure softmax approximation error |
| **Data** | chunk_sequences | Split flat tokens into fixed-length sequences |
| **Data** | load_tokenizer | Load/cache GPT-2 tokenizer |
| **Benchmark** | time_inference | Measure inference latency |
| **Benchmark** | compute_perplexity | Compute validation perplexity |
| **Benchmark** | memory_profiling | Profile memory usage |
| **Training** | TokenDataset | Next-token prediction dataset |
| **Training** | evaluate | Compute validation loss/perplexity |
| **Training** | save_checkpoint | Save training checkpoint |
| **Training** | get_lr | Compute learning rate schedule |
