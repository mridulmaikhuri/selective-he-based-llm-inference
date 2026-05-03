# Data Flow: In-Depth Guide

## Complete Inference Pipeline Walkthrough

### Scenario: Strategy 1 (Attention Output Encryption)

```
Initial Setup
═════════════

1. Model loaded: TinyGPT with 4 TransformerBlocks
   ├─ State: float32 weights, ready for inference
   └─ Location: GPU memory (if cuda available)

2. HE Context initialized:
   ├─ BFV scheme: n=2^13, t=65537
   ├─ Public/secret keys generated
   └─ Ready for encryption/decryption

3. Configuration:
   └─ layers_to_encrypt = ["transformer_blocks.0.attention.out_proj",
                          "transformer_blocks.1.attention.out_proj",
                          "transformer_blocks.2.attention.out_proj",
                          "transformer_blocks.3.attention.out_proj"]


Input: "What is AI?"  (one sentence)
═══════════════════

Step 0: Tokenization
─────────────────────
Input text: "What is AI?"
    │
    ├─ Tokenizer.encode()
    │
    ▼
Output: token_ids = [1001, 307, 16589]  (3 tokens)
   └─ These are integer IDs from vocabulary
```

---

### **Step 1: Token Embedding (Plaintext)**

```
Input: token_ids = [1001, 307, 16589]
       shape: (batch=1, seq_len=3)
       device: CPU/GPU
       dtype: torch.long

Processing:
───────────
model.embedding(token_ids)
    ↓
self.token_emb(token_ids) + self.pos_emb(positions)
    ├─ Token embedding lookup: [1001, 307, 16589]
    │  └─ Returns: (1, 3, 128) embeddings for each token
    ├─ Position embedding lookup: [0, 1, 2]
    │  └─ Returns: (1, 3, 128) positional encodings
    ├─ Add them element-wise: token_emb + pos_emb
    └─ Apply dropout (inference mode → no-op)

Output: embeddings
        shape: (1, 3, 128)
        dtype: float32
        values: ≈ [-0.5, 0.2, 1.3, ...] (per-token hidden states)

Key point: This layer is PLAINTEXT in Strategy 1
           Server sees: actual embeddings
```

---

### **Step 2: Block 0 Forward Pass (Plaintext)**

```
Input: hidden_states = (1, 3, 128)
       ├─ Sequence dimension: 3 tokens
       └─ Feature dimension: 128 (d_model)

Processing Block 0:
───────────────────

2a. LayerNorm (plaintext)
    ├─ Compute mean over feature dim: (1, 3, 1)
    ├─ Compute variance: (1, 3, 1)
    ├─ Normalize: (x - mean) / sqrt(var + eps)
    ├─ Scale and shift by learned γ, β
    └─ Output: (1, 3, 128) normalized activations

2b. CausalSelfAttention (plaintext)
    │
    ├─ Compute Q = x @ W_q: (1, 3, 128) @ (128, 128) = (1, 3, 128)
    ├─ Compute K = x @ W_k: (1, 3, 128) @ (128, 128) = (1, 3, 128)
    ├─ Compute V = x @ W_v: (1, 3, 128) @ (128, 128) = (1, 3, 128)
    │  └─ Split into heads: (batch=1, heads=4, seq=3, d_head=32)
    │
    ├─ Scaled dot-product attention:
    │  ├─ scores = Q @ K^T / sqrt(d_head)
    │  │  └─ (1, 4, 3, 3) matrix (3×3 attention grid)
    │  ├─ Apply causal mask: future tokens set to -inf
    │  │  └─ Prevents position 0 from attending to position 1, 2
    │  ├─ Softmax(scores): (1, 4, 3, 3)
    │  │  └─ Each row sums to 1.0 (attention weights)
    │  ├─ Weighted sum: attention_weights @ V
    │  │  └─ (1, 4, 3, 3) @ (1, 4, 3, 32) = (1, 4, 3, 32)
    │  └─ Merge heads: (1, 3, 128)
    │
    └─ Output projection (THIS IS ENCRYPTED IN STRATEGY 1!)
       └─ See Step 2c below

2c. Attention Output Projection (ENCRYPTED)
    ──────────────────────────────────────
    Input from attention: attn_out = (1, 3, 128) in plaintext
    
    ┌─ ENCRYPTION GATE ACTIVATED ─────────────────────┐
    │                                                  │
    │ ENCRYPT:                                         │
    │ ────────                                         │
    │ attn_out_float = (1, 3, 128) float32            │
    │      ├─ Scale by weight_scale = 100             │
    │      └─ attn_out_scaled = attn_out × 100        │
    │           = (1, 3, 128) int values: [50, -20, 130, ...]
    │      │                                          │
    │      └─> BFV Encryption (parallel)             │
    │           For each scalar (1×3×128 = 384):     │
    │           ├─ Sample randomness r, e             │
    │           ├─ Compute ctxt = pk·r + e + m       │
    │           │  (where m is the plaintext scalar)  │
    │           └─ Store as PyCtxt object (~48 KB)   │
    │                                                  │
    │ Result: attn_out_enc = TaggedList[ 384 PyCtxt ] │
    │         Total size: 384 × 48 KB ≈ 18 MB        │
    │                                                  │
    │ HE LINEAR COMPUTATION:                          │
    │ ──────────────────────                          │
    │ Goal: compute y_enc = attn_out_enc @ W + b     │
    │                       (where W and b are plain) │
    │                                                  │
    │ W_out: (128, 128) float32 weight matrix         │
    │        └─ Rounded to integers (no scale needed) │
    │ b_out: (128,) float32 bias                      │
    │                                                  │
    │ For each output neuron j ∈ {0..127}:            │
    │   y_j_enc := 0_enc  (additive identity)        │
    │   for i ∈ {0..127}:                            │
    │     y_j_enc += attn_out_enc[i] ⊗ W[i, j]       │
    │              ┌─────────────────────────────────┤
    │              │ Ciphertext × Plaintext mul      │
    │              │ Cost: moderate noise             │
    │              └─────────────────────────────────┤
    │   y_j_enc += b[j]  (add bias)                  │
    │              ┌─────────────────────────────────┤
    │              │ Ciphertext + Plaintext add      │
    │              │ Cost: negligible noise          │
    │              └─────────────────────────────────┤
    │                                                  │
    │ Result: y_enc = (1, 3, 128) encrypted output   │
    │                 (384 ciphertexts total)        │
    │                                                  │
    │ DECRYPT:                                        │
    │ ────────                                        │
    │ y_enc  (384 ciphertexts)                        │
    │    └─> BFV Decryption (parallel)               │
    │        For each ciphertext:                     │
    │        ├─ m' = (ctxt · sk) mod t               │
    │        └─ Recover: m' ∈ ℤ/tℤ                  │
    │                                                  │
    │ y_int = [50, -20, 130, ...] (integers)         │
    │    └─> Scale back by 1/100                     │
    │ y_float = [0.50, -0.20, 1.30, ...] (float32)  │
    │                                                  │
    │ Result: attn_out_proj = (1, 3, 128) float32   │
    │         (recovered back to plaintext)           │
    │                                                  │
    └────────────────────────────────────────────────┘

2d. Residual Connection + FFN (plaintext)
    ├─ residual_out = hidden_states + attn_out_proj
    │  └─ Element-wise add, both plaintext
    ├─ Normalize: LayerNorm(residual_out)
    ├─ FFN:
    │  ├─ fc1: (1, 3, 128) @ (128, 512) = (1, 3, 512)
    │  ├─ GELU activation: (1, 3, 512)
    │  ├─ fc2: (1, 3, 512) @ (512, 128) = (1, 3, 128)
    │  └─ Output: (1, 3, 128)
    └─ Residual: block_out = residual + ffn_out

Output: Block 0 output = (1, 3, 128) plaintext
        Hidden states for Block 1 input
```

---

### **Step 3: Blocks 1-3 (Repeated Pattern)**

```
Each block 1, 2, 3 repeats the same flow as Block 0:

┌────────────────────────────────┐
│ Block N:                       │
├────────────────────────────────┤
│ 1. LayerNorm (plaintext)       │
│ 2. Attention (plaintext, Q/K/V)│
│ 3. Attn out proj (ENCRYPTED)   │  ← Same HE path as Block 0
│ 4. Residual + FFN (plaintext)  │
│ 5. Output → Block N+1          │
└────────────────────────────────┘

After Block 3:
└─ hidden_states = (1, 3, 128) plaintext
```

---

### **Step 4: Final LayerNorm (Plaintext)**

```
Input: hidden_states from Block 3 = (1, 3, 128)

Processing:
───────────
Normalize over feature dimension:
├─ mean = hidden_states.mean(dim=-1, keepdim=True)
├─ var = hidden_states.var(dim=-1, keepdim=True)
├─ normalized = (hidden_states - mean) / sqrt(var + eps)
├─ scaled = normalized × γ + β  (learned parameters)
└─ Output: (1, 3, 128)

No encryption here. Server sees: final normalized features.
```

---

### **Step 5: LM Head (Plaintext)**

```
Input: final_hidden_states = (1, 3, 128)

Processing:
───────────
Linear projection: (1, 3, 128) @ (128, vocab_size)
                 = (1, 3, 50257)

y = x @ W + b
where:
  x: (1, 3, 128) hidden states
  W: (128, 50257) weight matrix
  b: (50257,) bias vector

Output: logits = (1, 3, 50257)
        One logit per token, per vocabulary entry

Key point: In Strategy 1, LM head is plaintext
           Server sees: token logits (can infer next tokens)
```

---

### **Step 6: Softmax & Top-K Decoding**

```
Input: logits = (1, 3, 50257)

Processing:
───────────
Softmax (per token):
├─ For position 0:
│  ├─ scores = exp(logits[0, :])
│  ├─ probs = scores / sum(scores)
│  └─ result[0] = probabilities over vocab

Top-K selection (k=5):
├─ For each position, find top 5 probability entries
├─ position 0: [(token_id, prob), ...]
├─ position 1: [(token_id, prob), ...]
├─ position 2: [(token_id, prob), ...]
└─ Return top-k token IDs

Example output:
    Position 0 predictions: [" I", " we", " the", " you", " that"]
    Position 1 predictions: [" am", " are", " is", " have", " have"]
    Position 2 predictions: [" a", " very", " good", " an", " the"]

Server returns this to client.
```

---

## Data Movement Summary for Strategy 1

```
┌─────────────────────────────────────────────────────┐
│  INPUT: "What is AI?"                               │
└─────────────────────────────────────────────────────┘
                     │
                     ▼ (plaintext)
┌─────────────────────────────────────────────────────┐
│  Tokenization: "What"→1001, "is"→307, "AI?"→16589  │
└─────────────────────────────────────────────────────┘
                     │
                     ▼ (plaintext)
┌─────────────────────────────────────────────────────┐
│  Embedding:        (1, 3, 128)                      │
│  Server observes:  ✓ token embeddings              │
└─────────────────────────────────────────────────────┘
                     │
    ┌────────────────┴────────────────┐
    │                                 │
    ▼ (plaintext)                     ▼ (plaintext)
┌─────────────────┐            ┌──────────────────┐
│ Block 0-3       │            │ Block 0-3        │
│ Attention Q/K/V │            │ Attention Q/K/V  │
│ Server sees:    │            │ Server sees:     │
│ ✓ these values  │            │ ✓ these values   │
└─────────────────┘            └──────────────────┘
    │  ▲                           │  ▲
    │  │ ┌──────────────────────────┘  │
    │  │ │ (3 times: Blocks 1-3)       │
    └──┼─┘                             │
       │                               │
       ▼ ENCRYPTED:                    │
    ┌─────────────────────────────────┐│
    │ Attention Output Projection      ││
    │ ├─ Encrypt input: 128 ctxts      ││
    │ ├─ he_linear compute:  encrypted ││
    │ ├─ Decrypt output:   plaintext   ││
    │ Server observes:    ✗ HIDDEN    ││
    └─────────────────────────────────┘│
       │                               │
       └───────────────────────────────┘
                     │
                     ▼ (plaintext)
┌─────────────────────────────────────────────────────┐
│  FFN:          (1, 3, 128) → (1, 3, 512) → (1, 3, 128)
│  Server sees:  ✓ all intermediate values            │
└─────────────────────────────────────────────────────┘
                     │
    ┌────────────────┴────────────────┐
    │                                 │
    ▼ (plaintext)                     ▼ (plaintext)
    (repeated for all blocks)
    
                     │
                     ▼ (plaintext)
┌─────────────────────────────────────────────────────┐
│  Final LayerNorm + LM Head:  (1, 3, 50257)          │
│  Server observes:  ✓ all logits                     │
└─────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  OUTPUT: Top-5 token predictions per position       │
└─────────────────────────────────────────────────────┘


SERVER OBSERVES:
┌────────────────────────────────┐
│ ✓ Token embeddings             │
│ ✓ Attention Q/K/V projections  │
│ ✗ Attention output projections │  ← Protected
│ ✓ FFN outputs                  │
│ ✓ Final logits                 │
└────────────────────────────────┘
```

---

## Comparison: Plaintext vs. Strategy 1 vs. Strategy 2

```
PLAINTEXT (No Encryption)
═════════════════════════

Embedding          (1, 3, 128)  ← EXPOSED
  ↓
Blocks 0-3         All visible  ← EXPOSED
  ├─ Attention
  ├─ Attention out proj
  └─ FFN
  ↓
LM Head            (1, 3, 50257) ← EXPOSED

Privacy: NONE
Latency: 5 ms


STRATEGY 1 (Attention Output Only)
═══════════════════════════════════

Embedding          (1, 3, 128)  ← EXPOSED
  ↓
Blocks 0-3
  ├─ Attention     EXPOSED
  ├─ Attn out proj ← ENCRYPTED  (hidden from server)
  └─ FFN           EXPOSED
  ↓
LM Head            (1, 3, 50257) ← EXPOSED

Privacy: MEDIUM (4 linear layers protected out of ~20)
Latency: 45 ms (9× overhead)


STRATEGY 2 (Last Block + LM Head)
═════════════════════════════════

Embedding          (1, 3, 128)  ← EXPOSED
  ↓
Blocks 0-2         All visible  ← EXPOSED
  ├─ Attention
  ├─ Attention out proj
  └─ FFN
  ↓
Block 3            ← ENCRYPTED  (entire block hidden)
  ├─ Attention
  ├─ Attention out proj
  └─ FFN
  ↓
LM Head            (1, 3, 50257) ← ENCRYPTED

Privacy: HIGH (semantic features in penultimate layer hidden)
Latency: 620 ms (124× overhead)


STRATEGY 3 (Embedding + LM Head)
════════════════════════════════

Embedding          ← ENCRYPTED  (token ID hidden)
  ↓
Blocks 0-3         All visible  ← EXPOSED
  ├─ Attention
  ├─ Attention out proj
  └─ FFN
  ↓
LM Head            ← ENCRYPTED  (logits hidden)

Privacy: MEDIUM (token ID and output hidden, not internals)
Latency: 180 ms (36× overhead)
```

---

## Memory Flow: Ciphertext Creation & Destruction

```
ENCRYPTION POINT IN Block 0 Attention Output:
══════════════════════════════════════════════

[BEFORE ENCRYPTION]
memory: attn_out_float ≈ 1.5 MB  (1×3×128 float32)

[DURING ENCRYPTION]
├─ Create 384 ciphertexts (1×3×128 values)
├─ Each ciphertext ≈ 48 KB
│
memory: ≈ 18 MB active
│       attn_out_float (keep it around)
│       + ciphertexts (384 PyCtxt objects)
│
├─ he_linear() runs on ciphertexts
├─ Creates 384 output ciphertexts
│
memory: ≈ 36 MB peak
│       (input ciphers + output ciphers)
│
├─ Decryption of 384 ciphertexts
├─ Produces: y_int (1×3×128 int64) ≈ 3 MB
│
memory: ≈ 4.5 MB
│       y_int (int64) + y_float (float32)
│       ciphertexts can be garbage-collected

[AFTER DECRYPTION]
memory: attn_out_float ≈ 1.5 MB (overwritten)
        y_float (same size) ≈ 1.5 MB


MEMORY FOOTPRINT OVER TIME:
───────────────────────────
│
│ 40 MB │     ╭──╮
│       │  ╭──┤  ├──╮
│ 30 MB │  │  │  │  │
│       │  │  │  │  │
│ 20 MB ├──┤  │  │  ├──
│       │  │  │  │  │
│ 10 MB ├──┤  │  │  │
│       │  │  │  │  │
│  0 MB └──┴──┴──┴──┘
│       │  │  │  │
│       Enc │HC│Dis
│          │rt│
│          │  │
└─────────────────────────

(Encrypt → HE Compute → Decrypt)
```

---

## Error Flow: What Happens if Something Goes Wrong

```
SCENARIO 1: Incorrect weight_scale
═════════════════════════════════

Original weights: W ∈ [-0.1, 0.1]
weight_scale = 1 (too small!)

Scaled: W_scaled = W × 1 = [-0.1, 0.1]
        └─> Rounds to: 0, 0, 0  (all zeros!)
        └─> BFV sees: all plaintext entries are 0
        └─> Computation: y = 0 × x + b = [b, b, b, ...]
        └─> Incorrect output!

Fix: Set weight_scale ≥ 100


SCENARIO 2: Noise Budget Exceeded
═════════════════════════════════

Model has 8 layers (each consumes 1 level):
├─ Budget: 27 levels (n=2^13)
├─ Usage: 8 levels
├─ Remaining: 19 levels ✓ OK

Model has 30 layers:
├─ Budget: 27 levels
├─ Usage: 30 levels
├─ Remaining: -3 levels ✗ FAIL!
└─> Decryption errors
└─> Can't recover plaintext from ciphertext

Fix: Reduce model depth (encrypt fewer layers)
     or use larger n (2^14, but slower)


SCENARIO 3: Non-Linear Layer in Encryption List
════════════════════════════════════════════════

config = SelectiveHEConfig(
    layers_to_encrypt=["relu_activation"]  # ← Problem!
)

At runtime:
├─ Engine tries: he_linear(x_enc, W, b)
├─ But ReLU has no weight matrix (it's not nn.Linear)
├─ Raises: ConfigError("Layer relu_activation is not nn.Linear")
└─> User must fix config

Fix: Only encrypt nn.Linear layers
     Activations must run plaintext (or use polynomial approx)
```

---

## Data Type Journey

```
Token ID (int)
    "What" → 1001
            │
            ▼
Embedding (float32)
    token_emb[1001] → [-0.5, 0.2, 1.3, ..., 0.1]  shape: (128,)
            │
            ▼
Hidden States (float32, multiple layers)
    Block 0 output → (1, 3, 128) float32
    Block 1 output → (1, 3, 128) float32
    Block 2 output → (1, 3, 128) float32
    Block 3 output → (1, 3, 128) float32
            │
            ▼
[ENCRYPTION GATE]  ← Strategy 1 Attention Output
            │
    Scaling: float32 × int
    [-0.5, 0.2, 1.3, ...] × 100 → [-50, 20, 130, ...]  (int)
            │
            ▼
BFV Plaintext (ℤ/tℤ)
    integers mod 65537: [65487, 20, 130, ...] (65537 = t)
            │
            ▼
BFV Ciphertext (polynomial ring)
    Encrypted representation: PyCtxt objects (~48 KB each)
            │
            ▼
HE Computation (ctxt × ptxt + ctxt + ctxt)
    Still in encrypted domain
            │
            ▼
BFV Decryption
    Recover plaintext: [65487, 20, 130, ...] mod 65537
            │
            ▼
Unscaling: int / int
    [65487, 20, 130, ...] / 100 → [-0.49, 0.20, 1.30, ...]  (float32)
            │
            ▼
Hidden States (float32, recovered)
    (slight rounding error from int conversion, but small)
            │
            ▼
Next block or LM head (float32 input)
            │
            ▼
Final Logits (float32)
    (1, 3, 50257) logits
            │
            ▼
Softmax → Probabilities
    (1, 3, 50257) probabilities ∈ [0, 1]
            │
            ▼
Top-K → Token IDs (int)
    [(word_id, prob), ...]
            │
            ▼
Tokenizer.decode() → Output Text
    [16589, 8, 12345, ...] → "AI is great..."
```
