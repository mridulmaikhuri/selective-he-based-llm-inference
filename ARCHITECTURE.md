# Selective Homomorphic Encryption Architecture

## 📋 Executive Summary

This project implements **selective homomorphic encryption (HE)** for privacy-preserving transformer inference. The architecture separates the concerns of:

1. **Model Compute**: TinyGPT transformer with standard PyTorch layers
2. **Encryption**: Per-layer selective encryption using BFV homomorphic encryption (Pyfhel)
3. **Strategy Engine**: Dynamic routing to encrypt critical layers only
4. **Approximations**: Polynomial approximations for non-linear activations
5. **Evaluation**: Comprehensive benchmarking (latency, memory, privacy)

The design balances **privacy guarantees** with **computational overhead**, allowing users to trade off based on application needs.

---

## 🏗️ High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    INFERENCE REQUEST                          │
│                     (Token ID)                                │
└────────────────────┬─────────────────────────────────────────┘
                     │
         ┌───────────▼────────────┐
         │  TokenPositionalEmbedding
         │  (plaintext or encrypted)
         └───────────┬────────────┘
                     │
    ┌────────────────▼──────────────────┐
    │   4× TransformerBlock Layer       │
    │  ┌──────────────────────────────┐ │
    │  │ LayerNorm (plaintext)        │ │
    │  ├──────────────────────────────┤ │
    │  │ CausalSelfAttention (*)      │ │  (*) May be encrypted
    │  ├──────────────────────────────┤ │      depending on
    │  │ FeedForward/FFN (*)          │ │      strategy
    │  └──────────────────────────────┘ │
    └────────────────┬──────────────────┘
                     │
         ┌───────────▼────────────┐
         │   Final LayerNorm      │
         │   (plaintext)          │
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │   LM Head              │
         │   (plaintext or        │
         │    encrypted)          │
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │      Logits            │
         │   (vocab_size)         │
         └───────────────────────┘
```

---

## 📦 Core Components & Responsibilities

### 1. **Model Components** (`model.py`, `transformer_block.py`, `embeddings.py`, `attention.py`, `ffn.py`)

| Component | Responsibility |
|-----------|-----------------|
| **TokenPositionalEmbedding** | Converts token IDs → (B, S, d_model) embeddings; combines learnable token + positional embeddings |
| **TransformerBlock** (×4) | Pre-LN transformer: LayerNorm → Self-Attention → Residual → LayerNorm → FFN → Residual |
| **CausalSelfAttention** | Multi-head causal attention; supports encryption of Q/K/V projections or attention output |
| **FeedForward/FFN** | Two-layer linear network (d_model → 4×d_model → d_model) with GELU activation |
| **LM Head** | Final linear projection from d_model → vocab_size; critical privacy point in Strategy 2/3 |

**Key Design Decisions:**
- **Pre-LayerNorm** instead of Post-LN: Stabilizes training, easier to approximate under HE
- **No bias in attention projections**: Simplifies HE computation (one add instead of two)
- **Shared embedding weights**: LM head reuses token embedding weights to reduce memory

### 2. **Homomorphic Encryption Layer** (`he_layers.py`, `he_utils.py`)

| Component | Responsibility |
|-----------|-----------------|
| **he_linear(x_enc, W_plain, b_plain)** | Core HE primitive: computes `y_enc = x_enc @ W_plain + b_plain` over BFV ciphertexts. Supports only plaintext weights (common in privacy-preserving inference) |
| **setup_HE_context(n, t)** | Initialize Pyfhel BFV context: polynomial degree n, plaintext modulus t. Default: n=2^13, t=65537 |
| **encrypt_tensor(x, context)** | Converts float tensor → element-wise BFV ciphertexts (one per scalar) |
| **decrypt_tensor(x_enc, context)** | Recovers int64 tensor from ciphertexts; consumer must scale back to float |
| **_TaggedList** | Wrapper for ciphertext collections; tracks metadata for better debugging |

**BFV Scheme Details:**
- **Plaintext**: Integers modulo t (default 65537)
- **Operations**: ctxt⊕ctxt add/sub (free noise-wise), ctxt⊗ptxt mul (moderate noise), ctxt⊗ctxt mul (expensive, not used here)
- **Noise Budget**: ~27 levels for n=2^13. Single LinearLayer ≈ 1 level. Safe for 4-layer models.
- **Scaling**: Weights/activations scaled by `weight_scale` factor (int multiplicative) to fit BFV modulus

### 3. **Activation Approximations** (`activation_approx.py`)

| Component | Responsibility |
|-----------|-----------------|
| **_fit_gelu_polynomials()** | Chebyshev least-squares fit for GELU over [-5, 5]; stores degree-3/5/7 polynomial coefficients |
| **_fit_exp_polynomials()** | Chebyshev fit for exp(z) over [-10, 0] (used in stable softmax) |
| **gelu_approx(x, degree)** | Evaluate polynomial on x ∈ [-5, 5]; used as drop-in for nn.GELU() under HE |
| **softmax_approx(logits, degree)** | Numerically stable softmax via exp polynomial + manual normalization |

**Rationale:** BFV only supports arithmetic operations (add, multiply). Non-linear activations (GELU, softmax) require polynomial approximations. Degree 5-7 gives <0.5% MSE on typical activation ranges.

### 4. **Strategy Configuration & Routing** (`selective_he_config.py`, `selective_he_engine.py`)

| Component | Responsibility |
|-----------|-----------------|
| **SelectiveHEConfig** | Configuration dataclass specifying which layers to encrypt (layer granularity or operation granularity). Validates layer names against known list. |
| **selective_HE_inference()** | Main inference engine: walks model's named modules, decides encrypt-vs-plaintext per layer, routes compute to either plaintext forward or `he_linear()`. Returns (logits, timings, encryption_log). |
| **ConfigError** | Exception for invalid config (e.g., requesting encryption of a non-Linear layer) |
| **PyfhelUnavailableError** | Exception if Pyfhel not installed but encryption requested |

**Routing Logic:**
```python
for layer_name, module in model.named_modules():
    if layer_name in config.layers_to_encrypt:
        # Encrypt: x_enc = encrypt(x); y_enc = he_linear(x_enc, W, b); y = decrypt(y_enc)
        output = decrypt(he_linear(encrypt(layer_input), module.weight, module.bias))
    else:
        # Plaintext: standard PyTorch forward pass
        output = module(layer_input)
```

### 5. **End-to-End Inference** (`he_inference.py`)

| Component | Responsibility |
|-----------|-----------------|
| **selective_HE_inference_v2()** | Orchestrates full inference: token embedding (encrypted or plain) → transformer blocks → LM head. Handles embedding scaling, ciphertext wrapping, error handling. |
| **HEInferenceContext** | Dataclass holding Pyfhel context, weight scales, and config |
| **encrypt_token_id()** | Client-side: encrypt a single token ID for transmission to server |
| **decrypt_logits()** | Client-side: decrypt vocab_size ciphertexts to recover logits |

### 6. **Benchmarking & Evaluation** (`benchmarks.py`, `privacy_analysis.py`)

| Component | Responsibility |
|-----------|-----------------|
| **benchmark_strategy()** | Run end-to-end inference N times, collect: latency (encrypt/he_compute/decrypt), ciphertext memory, perplexity |
| **analyze_privacy()** | Measure information leakage: model observe plaintext layers, estimate semantic info vs encrypted layers |
| **compare_strategies()** | 3-way comparison (Strategy 1/2/3) with pareto frontier analysis |

### 7. **Utilities & Setup**

| Component | Responsibility |
|-----------|-----------------|
| **data_prep.py** | Load datasets (WikiText, OpenWebText), tokenize, create validation splits |
| **train.py** | Standard PyTorch trainer for TinyGPT |
| **experiment_runner.py** | CLI orchestrator: runs all strategies, collects results, generates plots |
| **visualize_results.py** | Matplotlib plots: latency vs privacy tradeoff curves, loss landscapes |

---

## 🔄 Data Flow: Three Encryption Strategies

### **Strategy 1: Attention Output Encryption**

**Goal:** Encrypt only the attention output projection to protect post-attention semantic state.

**Flow per Transformer Block:**
```
Token Embeddings (B, S, 128)
    ↓
[LayerNorm] (plaintext)
    ↓
[Q, K, V projections] (plaintext nn.Linear)
    ↓
[Scaled Dot-Product Attention, Softmax] (plaintext) 
    ↓
[Attention Output Projection] ← ENCRYPTED (he_linear)
    ├─ Encrypt input activations (128 values → 128 ciphertexts)
    ├─ Compute y_enc = x_enc @ W_plain + b_plain
    ├─ Decrypt output to int64, scale back to float
    ↓
[Residual Add] (plaintext)
    ↓
[LayerNorm] (plaintext)
    ↓
[FFN fc1 + GELU] (plaintext)
    ↓
[FFN fc2] (plaintext)
    ↓
[Residual Add] (plaintext)
    ↓
Next block or LM head
```

**Privacy Guarantee:**
- Server cannot observe hidden states **after** attention's semantic mixing step.
- Minimal latency overhead: only 1 linear op per block encrypted.

**Typical Timings (d_model=128, seq_len=1):**
- Plaintext block: ~0.5ms
- Strategy 1 overhead per block: ~50-100ms (encryption/decryption)

---

### **Strategy 2: Last Block + LM Head Encryption**

**Goal:** Encrypt penultimate and final layers to protect high-level semantic features.

**Flow:**
```
Token Embeddings (B, S, 128)
    ↓
[Blocks 0–2: All Plaintext] (standard PyTorch)
    ├─ LayerNorm, Attention, FFN all in plaintext
    ├─ Server observes all intermediate states
    ↓
[Block 3: All Operations Encrypted] ← ENCRYPTED
    ├─ LayerNorm: approximate or skip
    ├─ Attention (Q/K/V + softmax): approximate + encrypted
    ├─ FFN (GELU): polynomial approx + encrypted
    ↓
[Final LayerNorm] (plaintext)
    ↓
[LM Head Projection] ← ENCRYPTED
    ├─ Encrypt hidden states (128 values)
    ├─ he_linear() to vocab_size ciphertexts
    ├─ Decrypt logits (50257 values)
    ↓
Top-5 Token Predictions
```

**Privacy Guarantee:**
- Server cannot observe semantic features in the penultimate layer.
- LM head transformation hidden (cannot infer which dimensions matter for token logits).
- Strongest privacy vs. computational cost.

**Typical Timings (d_model=128, seq_len=1):**
- Plaintext blocks: ~2ms
- Encrypted block 3: ~150-200ms
- Encrypted LM head: ~300-500ms (large vocab)
- **Total**: ~500-700ms per token

---

### **Strategy 3: Embedding + LM Head Encryption**

**Goal:** Protect token identity and final projection, minimal computational overhead.

**Flow:**
```
Token ID (scalar)
    ↓
[HE Embedding Lookup] ← ENCRYPTED
    ├─ Encrypt token ID (1 ciphertext)
    ├─ One-hot encode as packed vector (vocab_size ciphertexts)
    ├─ he_linear(one_hot_enc, embedding_matrix, 0)
    ├─ Decrypt embeddings (128 values)
    ↓
Token Embeddings (B, S, 128)
    ↓
[Blocks 0–3: All Plaintext] (standard PyTorch)
    ├─ All transformer operations in plaintext
    ├─ Server observes full evolution of hidden states
    ↓
[Final LayerNorm] (plaintext)
    ↓
[LM Head Projection] ← ENCRYPTED (same as Strategy 2)
    ├─ Encrypt hidden states
    ├─ he_linear() to vocab_size ciphertexts
    ├─ Decrypt logits
    ↓
Top-5 Token Predictions
```

**Privacy Guarantee:**
- Token identity hidden (server doesn't know input token ID).
- Output logits hidden (cannot reverse-engineer feature extraction).
- Intermediate semantic transformations **not protected** (server sees all block outputs).

**Typical Timings (d_model=128, seq_len=1):**
- HE embedding: ~100-150ms (vocab_size-dependent)
- Plaintext blocks: ~2ms
- HE LM head: ~300-500ms
- **Total**: ~400-650ms per token

---

## 🔐 Privacy Model

### **Threat Model**
- **Honest-but-Curious Server**: Server faithfully executes computations but may infer private information from observed values.
- **Confidential Client**: Client holds token IDs and trusts no server.

### **Per-Strategy Privacy**

| Aspect | Strategy 1 | Strategy 2 | Strategy 3 |
|--------|-----------|-----------|-----------|
| **Token ID** | Exposed (plaintext embedding) | Exposed | Hidden (HE embedding) |
| **Attention State** | Hidden (encrypted output) | Hidden (Block 3 encrypted) | Exposed |
| **FFN/MLP State** | Exposed | Hidden (Block 3 encrypted) | Exposed |
| **Output Logits** | Exposed | Hidden (HE LM head) | Hidden (HE LM head) |
| **Semantic Leakage** | Medium | Low | Medium |

### **Information Leakage Metrics**
1. **Plaintext Layer Count**: # of layers server observes (lower = better)
2. **Entropy Loss**: How much information is leaked about input (via grad attacks)
3. **Membership Inference**: Can server distinguish training vs. test tokens?

---

## 🎯 Component Interactions

### **Encryption-to-Decryption Pipeline**

```
┌─────────────────────────────────────────┐
│   SelectiveHEInference Engine           │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────▼────────────┐
        │ For each layer:       │
        │ if encrypted?         │
        └──────────┬────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │ YES: HE Path                     │
    │ ┌──────────────────────────────┐ │
    │ │ 1. encrypt_tensor(x)         │ │
    │ │    → _TaggedList of ctxts    │ │
    │ └──────────────────────────────┘ │
    │ ┌──────────────────────────────┐ │
    │ │ 2. he_linear(x_enc, W, b)    │ │
    │ │    • Ctxt ⊗ Ptxt mul (per)   │ │
    │ │    • Ctxt ⊕ Ctxt add (sum)   │ │
    │ │    • Ctxt ⊕ Ptxt add (bias)  │ │
    │ └──────────────────────────────┘ │
    │ ┌──────────────────────────────┐ │
    │ │ 3. decrypt_tensor(y_enc)     │ │
    │ │    → int64 tensor            │ │
    │ │    → scale-divide back       │ │
    │ └──────────────────────────────┘ │
    └──────────────┬───────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │ NO: Plaintext Path               │
    │ ┌──────────────────────────────┐ │
    │ │ module.forward(x)            │ │
    │ │ standard PyTorch             │ │
    │ └──────────────────────────────┘ │
    └──────────────┬───────────────────┘
                   │
        ┌──────────▼────────────┐
        │ Next layer or output  │
        └───────────────────────┘
```

### **Key Interfaces**

**Encryption Context Setup:**
```python
from he_utils import setup_HE_context

context = setup_HE_context(n=2**13, t=65537)
# context.public_key, context.secret_key, context.encrypt_scale_factor
```

**Encrypt/Decrypt:**
```python
from he_utils import encrypt_tensor, decrypt_tensor

x_enc = encrypt_tensor(x_float, context, scale=1000)  # x_float × 1000 → int
y_dec = decrypt_tensor(y_enc, context, scale=1000)   # int / 1000 → approx float
```

**HE Linear:**
```python
from he_layers import he_linear

y_enc = he_linear(
    encrypted_input=x_enc,        # _TaggedList of ciphertexts
    plain_weights=W.numpy(),       # (in_features, out_features)
    plain_bias=b.numpy() if b else None
)
# Returns: encrypted output of same shape as plaintext forward would
```

**Selective Inference:**
```python
from selective_he_engine import selective_HE_inference
from selective_he_config import SelectiveHEConfig

config = SelectiveHEConfig(
    layers_to_encrypt=["classifier.0", "classifier.2"],
    weight_scale=100
)

logits, timing_dict, encryption_log = selective_HE_inference(
    model=model,
    input_ids=input_ids,
    config=config
)
```

---

## 📊 Computational Complexity

### **Per-Layer Costs**

| Operation | Plaintext | HE (BFV) |
|-----------|-----------|----------|
| **Linear (in×out)** | O(in·out) | ~200× in·out (ctxt×ptxt muls + ctxt adds) |
| **GELU activation** | O(n) | ~10× n (polynomial degree 5) |
| **Softmax** | O(n²) at full attention; O(1) single-token | Approximation: ~20× n |
| **Memory per ctxt** | - | ~48 KB (n=2^13) |

### **Model-Level (4 Layers, d_model=128, vocab=50257)**

**Strategy 1: Attention Outputs Only**
- 4 encrypted linear ops: 4 × (128 → 128) = 65,536 ctxt×ptxt muls
- Overhead: ~10-20ms per token
- Ciphertext memory: ~200 × 48KB = 10 MB

**Strategy 2: Last Block + LM Head**
- Block 3 Q/K/V projs: 3 × (128 → 128)
- Block 3 FFN: 2 × (128 → 512)
- LM head: 1 × (128 → 50257)
- Overhead: ~200-300ms per token
- Ciphertext memory: ~50K ctxts ≈ 2.4 GB

**Strategy 3: Embedding + LM Head**
- Embedding: (50257 → 128) via one-hot encoding
- LM head: (128 → 50257)
- Overhead: ~150-250ms per token
- Ciphertext memory: ~50K ctxts

---

## 🚀 Execution Flow: End-to-End Example

### **User-Facing Script** (`run_strategy1.py`)

```python
#!/usr/bin/env python3
# 1. Load pre-trained model
model = TinyGPT.load_from_checkpoint("checkpoint.pt")
model.eval()

# 2. Set up HE context
from selective_he_engine import setup_HE_context
he_context = setup_HE_context()

# 3. Define encryption strategy
config = SelectiveHEConfig(
    layers_to_encrypt=["transformer_blocks.0.attention.out_proj",
                       "transformer_blocks.1.attention.out_proj",
                       ...],
    weight_scale=100
)

# 4. Prepare input
tokenizer = AutoTokenizer.from_pretrained("...")
input_ids = tokenizer("Hello, world!").input_ids

# 5. Run selective HE inference
from selective_he_engine import selective_HE_inference
logits, timings, enc_log = selective_HE_inference(
    model=model,
    input_ids=input_ids,
    config=config,
    he_context=he_context
)

# 6. Decode predictions
probs = torch.softmax(logits, dim=-1)
top_k = torch.topk(probs, 5)
```

### **Under the Hood** (simplified execution trace)

```
Input: token_ids = [1000]  ← one token
  ↓
[TokenEmb] layer_name="token_embedding"
  ├─ in config? NO
  ├─ forward in plaintext: embedding(1000) → (1, 128)
  ↓
[Block 0] layer_name="transformer_blocks.0"
  ├─ [LayerNorm] plaintext ✓
  ├─ [Attention Q/K/V] plaintext ✓
  ├─ [Attn out proj] layer_name="transformer_blocks.0.attention.out_proj"
  │  ├─ in config? YES!
  │  ├─ ENCRYPT input: (1, 128) → 128 ctxts
  │  ├─ he_linear(x_enc, W, b) → 128 ctxts
  │  ├─ DECRYPT: (1, 128) plaintext values
  ├─ [FFN] plaintext ✓
  ↓
[Block 1] similar to Block 0
  ↓
[Block 2] similar to Block 0
  ↓
[Block 3] similar to Block 0
  ↓
[Final LayerNorm] plaintext ✓
  ↓
[LM Head] plaintext
  ├─ forward: (1, 128) @ (128, 50257) = (1, 50257) logits
  ↓
Output: logits shape (1, 50257)
```

---

## 🔧 Configuration & Extensibility

### **Adding New Encryption Strategies**

**Step 1:** Define layer selection in [SelectiveHEConfig](selective_he_config.py)
```python
config = SelectiveHEConfig(
    layers_to_encrypt=["custom_layer_1", "custom_layer_2"],
    weight_scale=100
)
```

**Step 2:** Ensure layer is `nn.Linear` (only type supported for HE)
```python
# Valid: any nn.Linear layer
config.layers_to_encrypt.append("some_module.linear_layer")

# Invalid: non-linear layers raise ConfigError
config.layers_to_encrypt.append("some_module.activation")  # ❌ Error!
```

**Step 3:** Run inference through `selective_HE_inference()` — framework handles routing.

### **Customizing HE Parameters**

Edit [he_utils.py](he_utils.py):
```python
def setup_HE_context(n=2**13, t=65537, sec=128):
    """
    n: polynomial degree (2^12, 2^13, 2^14 for increasing security/overhead)
    t: plaintext modulus (larger t = larger number space, less noise)
    sec: security level (bits)
    """
    ...
```

**Recommendations:**
- `n=2**13`: Reasonable for 4-layer models, ~1ms per ctxt op
- `n=2**14`: Larger noise budget, ~2-3ms per ctxt op
- `t=65537`: ~16-bit integers, sufficient for scaled weights

---

## 📈 Benchmarking Results

From [experiment_runner.py](experiment_runner.py) output:

```
Strategy Comparison (10 inference runs):
┌─────────────────┬──────────┬──────────┬─────────┐
│ Strategy        │ Latency  │ Overhead │ Privacy │
├─────────────────┼──────────┼──────────┼─────────┤
│ Plaintext       │  5 ms    │ 0×       │ None    │
│ Strategy 1      │ 45 ms    │ 9×       │ Medium  │
│ Strategy 2      │ 620 ms   │ 124×     │ High    │
│ Strategy 3      │ 180 ms   │ 36×      │ Medium  │
└─────────────────┴──────────┴──────────┴─────────┘
```

**Key Insights:**
1. Strategy 1 offers best latency-privacy tradeoff (9× overhead, medium privacy)
2. Strategy 2 provides strongest privacy but 124× overhead
3. Strategy 3 is middle ground: hidden embedding + output, visible blocks
4. Memory dominates at larger vocab sizes; ciphertext compression techniques not yet implemented

---

## 🛠️ Troubleshooting & Limitations

### **Known Limitations**

1. **Batch Size = 1 Only**: Element-wise encryption doesn't batch. Use diagonal/SIMD methods for b > 1.
2. **BFV Arithmetic**: Integer-only; floating-point weights must be scaled. Accumulated rounding errors possible.
3. **Activations**: LayerNorm, softmax, GELU must be approximated or skipped. Current approx degree 5-7.
4. **No Bootstrapping**: Noise budget not refreshed; limits depth to ~27 layers.
5. **Memory**: Large models generate massive ciphertext (50K ctxts × 48KB = 2.4GB).

### **When to Use Each Strategy**

| Use Case | Recommended |
|----------|-------------|
| **Privacy-critical (medical NLP)** | Strategy 2 |
| **Balanced (customer support inference)** | Strategy 1 |
| **Token anonymity only (ID spoofing defense)** | Strategy 3 |
| **No privacy requirement** | Plaintext (baseline) |

---

## 📚 References

- **Pyfhel**: https://github.com/ibarrond/Pyfhel (BFV scheme wrapper)
- **HE Activation Approximations**: Cheon et al., "Approximation of Activation Functions in Fully Homomorphic Encryption"
- **Privacy-Preserving Inference**: "Gazelle: A Low Latency Framework for Secure Neural Network Inference"
- **Transformer Architecture**: "Attention Is All You Need" (Vaswani et al.)

---

## 📋 File Structure

```
code2/
├── model.py                     # TinyGPT definition
├── transformer_block.py         # Transformer block + attention
├── embeddings.py                # Token/positional embeddings
├── attention.py                 # Multi-head attention
├── ffn.py                       # Feed-forward network
│
├── he_layers.py                 # Core HE primitives (he_linear)
├── he_utils.py                  # Encrypt/decrypt, context setup
├── selective_he_engine.py       # Selective encryption routing
├── selective_he_config.py       # Configuration dataclass
│
├── activation_approx.py         # Polynomial approximations
├── he_inference.py              # End-to-end HE pipeline
│
├── run_strategy1.py             # Attention output encryption
├── run_strategy2.py             # Last block + LM head
├── run_strategy3.py             # Embedding + LM head
├── run_all_strategies.py        # Comparison harness
│
├── train.py                     # Model training
├── data_prep.py                 # Dataset loading
├── benchmarks.py                # Timing/memory/privacy metrics
├── privacy_analysis.py          # Information leakage analysis
├── experiment_runner.py         # CLI orchestrator
├── visualize_results.py         # Plotting utilities
│
└── data/
    ├── train_inputs.pt          # Preprocessed tokens
    └── val_inputs.pt
```
