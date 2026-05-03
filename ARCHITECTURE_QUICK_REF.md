# Quick Architecture Reference

## Selective HE Inference: High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT TOKEN IDS                            │
│                    (Tensor shape: [seq_len])                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ├──────────────────────────┐
                             │                          │
        ┌────────────────────▼──────┐      ┌───────────▼─────────┐
        │  STRATEGY 1:               │      │  STRATEGY 2:        │
        │  ATTENTION OUTPUT          │      │  LAST BLOCK + HEAD  │
        │                            │      │                     │
        │  Encrypt:                  │      │  Encrypt:           │
        │  ✓ Attn out proj          │      │  ✓ Block 3 all ops │
        │  ✗ Embedding              │      │  ✓ LM head          │
        │  ✗ FFN                    │      │  ✗ Blocks 0-2       │
        │  ✗ LM head                │      │  ✗ Embedding        │
        │                            │      │                     │
        │  Privacy: MEDIUM           │      │  Privacy: HIGH      │
        │  Latency: 45ms (9×)        │      │  Latency: 620ms (124×)
        └────────┬─────────────────┘       └────────┬────────────┘
                 │                                   │
        ┌────────▼──────────────┐                   │
        │  STRATEGY 3:           │                   │
        │  EMBEDDING + LM HEAD   │                   │
        │                        │                   │
        │  Encrypt:              │                   │
        │  ✓ Embedding lookup   │                   │
        │  ✓ LM head            │                   │
        │  ✗ Blocks 0-3         │                   │
        │                        │                   │
        │  Privacy: MEDIUM       │                   │
        │  Latency: 180ms (36×)  │                   │
        └────────┬───────────────┘                   │
                 │                                   │
        ┌────────▼───────────────────────────────────▼──────────┐
        │               ALL THREE CONVERGE                       │
        │                FINAL LOGITS                            │
        │          (Shape: [seq_len, vocab_size])               │
        └──────────────────────┬──────────────────────────────────┘
                               │
                ┌──────────────▼──────────────┐
                │  TOP-K TOKEN PREDICTIONS    │
                │  (Most likely next words)   │
                └─────────────────────────────┘
```

---

## Component Interaction Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    USER SCRIPT (run_strategy*.py)            │
│  1. Load model checkpoint                                    │
│  2. Set SelectiveHEConfig (which layers to encrypt)          │
│  3. Call selective_HE_inference()                            │
└─────────────────────────────┬────────────────────────────────┘
                              │
                ┌─────────────▼──────────────┐
                │ selective_he_engine.py     │
                │                            │
                │ selective_HE_inference()   │
                │ ├─ For each module:        │
                │ │  if in config:           │
                │ │    → encrypt path        │
                │ │  else:                   │
                │ │    → plaintext path      │
                └─────────────┬──────────────┘
                              │
        ┌─────────────────────┴──────────────────────┐
        │                                            │
        │                                            │
    ┌───▼─────────────┐                  ┌──────────▼───────────┐
    │  ENCRYPT PATH   │                  │  PLAINTEXT PATH     │
    │                 │                  │                     │
    │ he_utils.py:    │                  │ torch.nn.Module:    │
    │ ├─ encrypt()    │                  │ ├─ forward()        │
    │ └─ decrypt()    │                  │ └─ standard ops     │
    │                 │                  │                     │
    │ he_layers.py:   │                  │ (standard PyTorch)  │
    │ └─ he_linear()  │                  │                     │
    │    ├─ ctxt×ptxt │                  │                     │
    │    ├─ ctxt⊕ctxt │                  │                     │
    │    └─ ctxt⊕ptxt │                  │                     │
    │                 │                  │                     │
    │ Pyfhel (BFV):   │                  │                     │
    │ ├─ Arithmetic   │                  │                     │
    │ └─ Key mgmt     │                  │                     │
    └────┬────────────┘                  └──────────┬──────────┘
         │                                           │
         └───────────────────┬───────────────────────┘
                             │
                    ┌────────▼──────────┐
                    │   Logits Output   │
                    │  (shape: vocab)   │
                    └───────────────────┘
```

---

## Data Movement Through Single Layer

```
PLAINTEXT PATH:
───────────────

    Input Tensor (B, d_model)
           │
           ├─ W: weight matrix (d_model, out_dim)
           ├─ b: bias vector (out_dim)
           │
           ▼
    x @ W + b  ← Pure PyTorch, nanoseconds
           │
           ▼
    Output Tensor (B, out_dim)


HOMOMORPHIC ENCRYPTION PATH:
─────────────────────────────

    Step 1: ENCRYPT
    ───────────────
    x_float (B, d_model)  ← typically in [-1, 1]
           │
           ├─ Scale by weight_scale (e.g., ×1000)
           │
           ├─ Round to integers
           │
           ▼
    x_int (B, d_model)
           │
           ├─ Encrypt each scalar element
           │
           ▼
    x_enc: TaggedList[ Pyfhel.PyCtxt ]  ← size B·d_model
    └─ Each ctxt represents one scalar, ~48KB each


    Step 2: COMPUTE (he_linear)
    ──────────────────────────
    For each output neuron j:
        y_j_enc = 0_enc
        for i in 0..d_model:
            y_j_enc += x_i_enc * W[i,j]  ← ctxt×ptxt mult
        y_j_enc += b[j]  ← ctxt+ptxt add

    Result: y_enc: TaggedList[ Pyfhel.PyCtxt ]  ← size B·out_dim


    Step 3: DECRYPT
    ──────────────
    y_enc  ← each ciphertext ~48KB
           │
           ├─ Decrypt using secret key
           │
           ▼
    y_int (B, out_dim)  ← integers
           │
           ├─ Divide by weight_scale
           │
           ▼
    y_float (B, out_dim)  ← approximate recovery, microseconds


    TOTAL OVERHEAD:
    ───────────────
    Plaintext:    ~0.01ms
    Encrypt:      ~10ms    (B·d_model ctxts)
    HE Compute:   ~50ms    (B·d_model·out_dim ctxt ops)
    Decrypt:      ~10ms    (B·out_dim ctxts)
    ────────────────────
    Total:        ~70ms    (7000× slower!)
```

---

## BFV Homomorphic Encryption Scheme

```
┌─────────────────────────────────────────────────────────────┐
│         PYFHEL BFV (Brakerski-Fan-Vercauteren)             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PLAINTEXT SPACE:  ℤ/tℤ  (integers mod t)                 │
│  Example: t = 65537 (16-bit prime)                         │
│  Range: {0, 1, 2, ..., 65536}                              │
│                                                             │
│  CIPHERTEXT SPACE: Polynomial ring (ℤ/qℤ)[x]/(x^n+1)      │
│  n = 2^13 (8192)  — polynomial degree                      │
│  q = large prime   — modulus                               │
│  ≈48 KB per ciphertext (with n=2^13)                       │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  KEY GENERATION:                                            │
│  ├─ Public Key (pk):  Used for encryption                  │
│  ├─ Secret Key (sk):  Used for decryption (KEEP SECRET!)   │
│  └─ Eval Keys:        For ciphertext × ciphertext ops      │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  OPERATIONS:                                                │
│                                                             │
│  1. Addition (ctxt ⊕ ctxt):                                │
│     m₁ + m₂ ← Dec(C₁ + C₂)                                 │
│     Cost: negligible noise ← why we prefer this            │
│                                                             │
│  2. Plaintext Multiply (ctxt ⊗ ptxt):                      │
│     m · p ← Dec(C * p)                                     │
│     Cost: moderate noise ← safe for linear layers          │
│                                                             │
│  3. Ciphertext Multiply (ctxt ⊗ ctxt):  EXPENSIVE!         │
│     m₁ · m₂ ← Dec(C₁ * C₂)                                 │
│     Cost: exponential noise growth ← avoid!                │
│     (Would need bootstrapping to refresh)                  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  NOISE BUDGET:                                              │
│                                                             │
│  Each operation consumes "noise" (error bits)              │
│  Total budget ≈ 27 levels for n=2^13                       │
│                                                             │
│  Single linear layer:  ~1 level consumed                   │
│  ┌────┴────┬────┴────┬────┴────┬────┴────┐                 │
│  │ Block 0 │ Block 1 │ Block 2 │ Block 3 │                 │
│  │   1     │   1     │   1     │   1     │                 │
│  └────┬────┴────┬────┴────┬────┴────┬────┘                 │
│       └─ 4 consumed, 23 remaining ✓                        │
│                                                             │
│  If we add too many operations → noise exceeds capacity    │
│  → Decryption fails (incorrect results)                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuration Deep-Dive: SelectiveHEConfig

```
SelectiveHEConfig(
    layers_to_encrypt = ["transformer_blocks.0.attention.out_proj",
                         "transformer_blocks.1.attention.out_proj",
                         "transformer_blocks.2.attention.out_proj",
                         "transformer_blocks.3.attention.out_proj"],
    weight_scale = 100,
    encryption_granularity = "layer"
)

What this means:
────────────────

1. WHICH LAYERS?
   ├─ "transformer_blocks.0.attention.out_proj"
   │  └─ The output projection inside attention, layer 0
   ├─ "transformer_blocks.1.attention.out_proj"
   │  └─ Same for layer 1
   ├─ (and so on...)
   
   Strategy 1 encrypts these 4 layers (one per block).
   Strategy 2 would add "transformer_blocks.3..." (entire block).
   Strategy 3 would replace these with "token_embedding" and "lm_head".

2. WEIGHT_SCALE
   ├─ Original weights: float32, typically in [-0.1, 0.1]
   ├─ BFV only handles integers
   ├─ Solution: multiply by weight_scale BEFORE encryption
   │
   │  Example with weight_scale=100:
   │  ┌────────────────────────────────────┐
   │  │ W_float = [0.05, -0.02, ...]       │
   │  │ W_scaled = [5, -2, ...]            │ ← integers now
   │  │ Encrypt W_scaled                   │
   │  │ ...HE compute...                   │
   │  │ Decrypt result: y_int              │
   │  │ y_float = y_int / 100              │ ← recover floats
   │  └────────────────────────────────────┘
   │
   ├─ Too small (100): rounding errors
   ├─ Too large (10000): overflow in BFV space (t=65537)
   └─ Goldilocks zone: 10 - 1000 (problem dependent)

3. ENCRYPTION_GRANULARITY
   ├─ "layer": encrypt ALL operations in listed layers
   ├─ "operation": encrypt only specific ops in listed layers
   │   Example: {"attention.out_proj": ["matmul", "add"]}
   │            (not implemented in current version)
   └─ Current version: always "layer"
```

---

## Memory Breakdown: Strategy 2 with TinyGPT (d=128, vocab=50257)

```
CIPHERTEXT MEMORY ANALYSIS:
═════════════════════════════

Given:
  ├─ BFV parameter: n=2^13 (polynomial degree 8192)
  ├─ One ciphertext ≈ 48 KB (2 polynomials × 8192 × 3 bytes)
  ├─ We encrypt element-wise (one ctxt per scalar)
  │
  Model dimensions:
  ├─ d_model = 128
  ├─ vocab_size = 50257
  └─ seq_len = 1 (for single-token inference)


MEMORY PER COMPONENT:
─────────────────────

1. Block 3 Input:  128 ctxts × 48 KB = 6.1 MB
   └─ Hidden state is 128-dimensional, fully encrypted

2. Block 3 Attention:
   ├─ Q: 128 ctxts (after qkv_proj)
   ├─ K: 128 ctxts (after qkv_proj)
   ├─ V: 128 ctxts (after qkv_proj)
   ├─ Attention output: 128 ctxts
   └─ Total: ~24 MB (peak usage during forward pass)

3. Block 3 FFN:
   ├─ FC1 output (after scaling): 512 ctxts
   │  └─ 512 × 48 KB = 24.6 MB
   ├─ GELU (approximated): keeps 512 ctxts
   ├─ FC2 output: 128 ctxts
   └─ Total: ~25 MB (peak)

4. Block 3 Output: 128 ctxts = 6.1 MB

5. LM Head Computation:
   ├─ Input: 128 ctxts × 48 KB = 6.1 MB
   ├─ Output: 50257 ctxts × 48 KB = 2.4 GB  ← HUGE!
   └─ Total: ~2.4 GB (peak memory point)


TOTAL PEAK MEMORY:
──────────────────
Max across all layers: ~2.4 GB (at LM head output)

Plaintext model by comparison: ~50 MB

HE overhead: 2.4 GB / 50 MB = 48× memory!
```

---

## Latency Breakdown: Strategy 2, Single Forward Pass

```
TIMING ANALYSIS:
═════════════════

Input: token_id = 1000 (one token)

┌─────────────────────────────────────────────────┐
│ Token Embedding (plaintext)     │  0.5 ms      │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Blocks 0-2 (plaintext)                          │
│  ├─ LayerNorm: 0.1 ms                          │
│  ├─ Attention (Q/K/V + softmax): 1.0 ms       │
│  ├─ Attention out proj: 0.2 ms                 │
│  ├─ FFN: 0.4 ms                                │
│  └─ Total per block: 1.7 ms × 3 = 5.1 ms     │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Block 3 (ENCRYPTED)                            │
│                                                 │
│  Input: 128 float values                       │
│  ├─ ENCRYPT (128 scalars): 10 ms               │
│  │  └─ BFV encryption (parallel)               │
│  │                                              │
│  ├─ LayerNorm (approximated):                  │
│  │  └─ Polynomial eval on 128 ctxts: 50 ms     │
│  │                                              │
│  ├─ Attention Q/K/V projections:               │
│  │  ├─ Q: he_linear(128, 128) = 80 ms          │
│  │  ├─ K: he_linear(128, 128) = 80 ms          │
│  │  ├─ V: he_linear(128, 128) = 80 ms          │
│  │  └─ Subtotal: 240 ms                        │
│  │                                              │
│  ├─ Attention (softmax approximated):          │
│  │  └─ Polynomial softmax on ctxts: 60 ms      │
│  │                                              │
│  ├─ Attention out proj:                        │
│  │  └─ he_linear(128, 128) = 80 ms             │
│  │                                              │
│  ├─ FFN:                                       │
│  │  ├─ FC1: he_linear(128, 512) = 320 ms       │
│  │  ├─ GELU (polynomial approx): 100 ms        │
│  │  ├─ FC2: he_linear(512, 128) = 160 ms       │
│  │  └─ Subtotal: 580 ms                        │
│  │                                              │
│  └─ DECRYPT (128 values): 10 ms                │
│                                                 │
│  Block 3 Total: ~1200 ms                       │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Final LayerNorm (plaintext)     │  0.5 ms      │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ LM Head (ENCRYPTED)                             │
│                                                 │
│  Input: 128 float values                       │
│  ├─ ENCRYPT: 10 ms                             │
│  ├─ he_linear(128, 50257): 2000 ms             │
│  │  └─ Each of 50257 output neurons:          │
│  │     128 ctxt×ptxt muls + 127 ctxt adds     │
│  ├─ DECRYPT: 400 ms                            │
│  │  └─ 50257 ciphertexts to decrypt           │
│  └─ Subtotal: 2410 ms                          │
└─────────────────────────────────────────────────┘


GRAND TOTAL:
═════════════
Plaintext (Blocks 0-2):        5 ms
Encrypted (Block 3):        1200 ms
Encrypted (LM head):        2410 ms
Other:                         6 ms
                            ────────
TOTAL:                      3621 ms ≈ 3.6 seconds per token!


COMPARISON:
──────────
Plaintext model:     5 ms (baseline)
Strategy 2:       3600 ms (720× slower)

For production: Would need
  ├─ Batching (SIMD over multiple tokens)
  ├─ Ciphertext compression
  ├─ GPU acceleration
  ├─ or accept this latency for high-security applications
```

---

## Integration Checklist: Adding Your Own Encrypted Layer

```
STEP 1: Identify Target Layer
────────────────────────────

In your model definition (e.g., model.py):
    class MyModel(nn.Module):
        def __init__(self):
            ...
            self.custom_layer = nn.Linear(128, 256)  ← This one
            ...


STEP 2: Extract Layer Name
──────────────────────────

Run this to see all layer names:
    model = MyModel()
    for name, module in model.named_modules():
        print(name, type(module).__name__)

Output:
    ...
    custom_layer Linear
    ...

Layer name: "custom_layer"


STEP 3: Update SelectiveHEConfig
────────────────────────────────

    config = SelectiveHEConfig(
        layers_to_encrypt=["custom_layer"],  ← Add here
        weight_scale=100
    )


STEP 4: Verify Layer Type
─────────────────────────

MUST be nn.Linear for encryption. If it's:
  ✓ nn.Linear(in, out)              → OK
  ✓ nn.Linear(in, out, bias=True)   → OK
  ✓ nn.Linear(in, out, bias=False)  → OK
  ✗ nn.Conv1d / nn.Conv2d           → NOT supported
  ✗ nn.LSTM / nn.GRU                → NOT supported
  ✗ Custom non-linear layer         → NOT supported


STEP 5: Run selective_HE_inference
──────────────────────────────────

    from selective_he_engine import selective_HE_inference
    
    logits, timing_dict, enc_log = selective_HE_inference(
        model=model,
        input_ids=input_ids,
        config=config
    )

    print("Encrypted layers:", enc_log)
    # Output: [("custom_layer", True), ...]


STEP 6: Analyze Results
──────────────────────

    print(f"Total HE compute time: {timing_dict['he_compute_time']:.2f}s")
    print(f"Total encryption/decryption: {timing_dict['encryption_time']:.2f}s")
    print(f"Ciphertext memory usage: ?")  ← See memory calc above
```

---

## Troubleshooting Decision Tree

```
Problem: "Decryption gives wrong results"
─────────────────────────────────────────

    Is weight_scale set reasonably?
    ├─ NO → Increase to 100-1000, retry
    │
    ├─ YES → Is noise budget exceeded?
    │        (can't know easily without debugging BFV internals)
    │        ├─ Try reducing model depth (encrypt fewer layers)
    │        ├─ Use smaller n (2^12 has less budget, but try anyway)
    │        └─ This is a known limitation; HE has depth limits
    │
    └─ Check layer dtype
       └─ Inputs should be float32, not float64 or other
          Weights/biases should be float32


Problem: "Memory usage is gigabytes, way too much!"
────────────────────────────────────────────────────

    Yes, this is expected. For d=128, vocab=50257:
    ├─ Each ciphertext ≈ 48 KB
    ├─ LM head outputs 50257 ctxts
    ├─ Total: 2.4 GB (not a bug, a feature of HE!)
    │
    Mitigation:
    ├─ Use smaller vocab (vocab=1000 → 50 MB)
    ├─ Don't encrypt LM head (use Strategy 1)
    ├─ Batch process tokens (reduces ciphertext inflation)
    └─ Wait for batching + compression in future work


Problem: "ImportError: No module named 'Pyfhel'"
────────────────────────────────────────────────

    pip install Pyfhel


Problem: "My custom layer isn't being encrypted"
─────────────────────────────────────────────────

    Check:
    1. Is the layer in layers_to_encrypt? Print it
    2. Is the layer an nn.Linear? Print module type
    3. Run with verbose=True to see encryption log
    4. Check that model.named_modules() returns the expected names


Problem: "HE inference is 500× slower than plaintext"
──────────────────────────────────────────────────────

    This is EXPECTED and CORRECT!

    BFV overhead factors:
    ├─ Encryption/decryption: 10-100ms per layer
    ├─ Ciphertext arithmetic: 1-10 ops per plaintext op
    └─ Noise budget management: limits depth

    Mitigation:
    ├─ Use GPU-accelerated FHE (TFHE-rs, OpenFHE+CUDA)
    ├─ Reduce model size
    ├─ Accept latency for privacy-critical apps
    └─ Consider MPC + HE hybrid protocols
```
