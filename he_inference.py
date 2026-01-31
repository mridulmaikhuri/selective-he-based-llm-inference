"""
Full homomorphic encryption inference pipeline for transformer language models.

This module demonstrates end-to-end encrypted inference on a toy transformer,
showing how to encrypt inputs, run through transformer blocks, and decrypt outputs.

⚠️ CRITICAL LIMITATIONS:
- Batch size = 1 only (single sample)
- Sequence length = 1 only (single token at a time)
- Toy model parameters (small embedding dim, few layers)
- Approximate softmax (no true probability distribution)
- Integer quantization (rounding errors)
- Noise growth limits model depth to ~2-3 transformer blocks
- Memory: ciphertexts are ~1KB each; full inference ~100MB+ for typical model

Practical Use Cases:
- Research on encrypted inference feasibility
- Privacy-preserving single-token generation (e.g., first word only)
- Educational demonstration of HE in ML
- NOT suitable for production inference

Next Steps to Production:
1. Batching: Use SIMD packing in Pyfhel slots (CKKS or BGV)
2. Approximate softmax: Polynomial approximation or use CKKS for real numbers
3. Reduce noise: Better parameter selection, modulus switching
4. Optimize matrix products: Block-wise multiplication, sparsity
5. Hardware acceleration: GPU-optimized HE libraries (HEAX, CuPy)
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Tuple, List, Dict, Any
from Pyfhel import Pyfhel

from he_utils import setup_HE_context, encrypt_tensor, decrypt_tensor
from he_layers import he_linear, he_attention_approx


class ToyTransformer(nn.Module):
    """
    Minimal transformer model for demonstration.
    
    Parameters:
    - vocab_size: vocabulary size (default 256 for small toy model)
    - embed_dim: embedding dimension (default 16 for efficiency)
    - num_layers: number of transformer blocks (default 1)
    - intermediate_dim: FFN hidden dimension (default 32)
    """
    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 16,
        num_layers: int = 1,
        intermediate_dim: int = 32
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.intermediate_dim = intermediate_dim
        
        # Embedding layer
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            nn.ModuleDict({
                'self_attn_q': nn.Linear(embed_dim, embed_dim, bias=False),
                'self_attn_k': nn.Linear(embed_dim, embed_dim, bias=False),
                'self_attn_v': nn.Linear(embed_dim, embed_dim, bias=False),
                'attn_out': nn.Linear(embed_dim, embed_dim, bias=True),
                'ffn_in': nn.Linear(embed_dim, intermediate_dim, bias=True),
                'ffn_out': nn.Linear(intermediate_dim, embed_dim, bias=True),
            })
            for _ in range(num_layers)
        ])
        
        # LM head (logits over vocabulary)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=True)
        
        # Initialize weights to small values for stability
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (plaintext, for reference).
        
        Args:
            input_ids: shape (batch_size, seq_len)
        
        Returns:
            logits: shape (batch_size, seq_len, vocab_size)
        """
        # Embedding
        hidden = self.embeddings(input_ids)  # (batch, seq_len, embed_dim)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            # Self-attention (simplified: no multi-head)
            Q = block['self_attn_q'](hidden)  # type: ignore  # (batch, seq_len, embed_dim)
            K = block['self_attn_k'](hidden)  # type: ignore
            V = block['self_attn_v'](hidden)  # type: ignore
            
            # Scaled dot-product attention (plaintext)
            scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.embed_dim ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn_weights, V)
            
            # Attention projection
            hidden = block['attn_out'](attn_out)  # type: ignore
            
            # Feed-forward
            ffn_hidden = torch.relu(block['ffn_in'](hidden))  # type: ignore
            hidden = block['ffn_out'](ffn_hidden)  # type: ignore
        
        # LM head
        logits = self.lm_head(hidden)  # (batch, seq_len, vocab_size)
        return logits


def full_he_inference(
    model: ToyTransformer,
    input_ids: torch.Tensor,
    HE: Pyfhel
) -> Tuple[List[int], Dict[str, float], Dict[str, Any]]:
    """
    Perform full encrypted inference on a transformer language model.

    Parameters
    ----------
    model : ToyTransformer
        Pre-trained transformer model with frozen weights.
    
    input_ids : torch.Tensor
        Input token IDs, shape (1, 1) for batch_size=1, seq_len=1.
    
    HE : Pyfhel
        Initialized homomorphic encryption context.

    Returns
    -------
    top_5_tokens : list of int
        Top-5 most likely next tokens.
    
    timings : dict
        Timing breakdown:
        - 'embedding': time for encrypted embedding lookup (ms)
        - 'transformer_blocks': list of timings per block (ms each)
        - 'lm_head': time for encrypted LM head (ms)
        - 'decryption': time to decrypt logits (ms)
        - 'total': total inference time (ms)
    
    memory_info : dict
        Memory profiling information:
        - 'num_ciphertexts': total ciphertexts used
        - 'est_ciphertext_size_mb': estimated memory in MB
        - 'peak_ciphertexts': maximum simultaneous ciphertexts

    Raises
    ------
    ValueError
        If input shape is not (1, 1) for batch_size=1, seq_len=1.
    
    Notes
    -----
    Constraints:
    - Only supports batch_size=1 and seq_len=1
    - Uses approximate softmax (scaling by max, not true softmax)
    - Integer quantization causes rounding errors
    - Noise growth limits to ~2-3 transformer blocks before failure
    - Memory: Each ciphertext ~1KB; full model can use 100MB+
    
    Algorithm:
    1. Encrypt embedding lookup result
    2. For each transformer block:
       a. Compute encrypted Q, K, V via he_linear
       b. Compute encrypted attention (Q@K^T, softmax approx, @V)
       c. Attention projection via he_linear
       d. FFN via two he_linear calls (no activation in HE)
    3. Decrypt hidden state, apply ReLU, re-encrypt for next block (expensive!)
    4. Decrypt final hidden, run LM head in plaintext
    5. Return top-5 tokens by logits
    """
    batch_size, seq_len = input_ids.shape
    
    if batch_size != 1 or seq_len != 1:
        raise ValueError(
            f"Only batch_size=1, seq_len=1 supported. Got ({batch_size}, {seq_len})"
        )
    
    timings = {}
    memory_info: Dict[str, Any] = {
        'num_ciphertexts': 0,
        'est_ciphertext_size_mb': 0.0,
        'peak_ciphertexts': 0,
    }
    
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    print("\n" + "=" * 80)
    print("Full HE Inference Pipeline (batch_size=1, seq_len=1)")
    print("=" * 80)
    
    # ========== STEP 1: Embedding ==========
    print("\n[1] Encrypted Embedding Lookup")
    start = time.perf_counter()
    
    token_id = int(input_ids[0, 0].item())
    print(f"  Input token ID: {token_id}")
    
    # Get plaintext embedding
    embedding_vec = model.embeddings.weight[token_id]  # shape (embed_dim,)
    print(f"  Embedding shape: {embedding_vec.shape}")
    print(f"  Embedding sample values: {embedding_vec[:4].tolist()}")
    
    # Encrypt embedding (scale larger to reduce rounding error)
    SCALE_FACTOR = 10000  # Scale embeddings and weights for integer arithmetic
    embedding_long = (embedding_vec * SCALE_FACTOR).round().long()
    enc_embedding = encrypt_tensor(embedding_long, HE, batch=False)
    memory_info['num_ciphertexts'] += len(enc_embedding)
    memory_info['peak_ciphertexts'] = max(memory_info['peak_ciphertexts'], len(enc_embedding))
    
    embedding_time = (time.perf_counter() - start) * 1000
    timings['embedding'] = embedding_time
    print(f"  Embedding encrypted in {embedding_time:.2f} ms")
    print(f"  Ciphertexts used: {len(enc_embedding)}")
    
    # ========== STEP 2: Transformer Blocks ==========
    print("\n[2] Transformer Blocks")
    
    enc_hidden = enc_embedding
    timings['transformer_blocks'] = []
    
    for block_idx, block in enumerate(model.transformer_blocks):
        print(f"\n  Block {block_idx}:")
        block_start = time.perf_counter()
        
        # ===== Self-Attention =====
        print(f"    [Self-Attention]")
        attn_start = time.perf_counter()
        
        # Get Q, K, V projections (plaintext weights, encrypted hidden)
        W_q = block['self_attn_q'].weight.detach().cpu().numpy().astype(np.float32)  # type: ignore  # Keep as float
        W_k = block['self_attn_k'].weight.detach().cpu().numpy().astype(np.float32)  # type: ignore
        W_v = block['self_attn_v'].weight.detach().cpu().numpy().astype(np.float32)  # type: ignore
        
        # Scale weights for integer multiplication
        W_q = (W_q * SCALE_FACTOR).astype(np.int32)
        W_k = (W_k * SCALE_FACTOR).astype(np.int32)
        W_v = (W_v * SCALE_FACTOR).astype(np.int32)
        
        # Q = hidden @ W_q^T
        enc_Q = he_linear(enc_hidden, torch.from_numpy(W_q).T, None, HE)
        enc_K = he_linear(enc_hidden, torch.from_numpy(W_k).T, None, HE)
        enc_V = he_linear(enc_hidden, torch.from_numpy(W_v).T, None, HE)
        
        # Attention (simplified: single sequence position, so attention is 1x1)
        # For seq_len=1, Q@K^T is just Q[0]*K[0] (scalar), weights are [1.0], output is V[0]
        # But we'll use he_attention_approx anyway for demonstration
        enc_attn_out = he_attention_approx(enc_Q, enc_K, enc_V, HE, causal_mask=False)
        
        # Attention output projection
        W_attn_out = block['attn_out'].weight.detach().cpu().numpy().astype(np.float32)  # type: ignore
        b_attn_out = block['attn_out'].bias.detach().cpu().numpy().astype(np.float32)  # type: ignore
        W_attn_out = (W_attn_out * SCALE_FACTOR).astype(np.int32)
        b_attn_out = (b_attn_out * SCALE_FACTOR).astype(np.int32)
        enc_attn_proj = he_linear(enc_attn_out, torch.from_numpy(W_attn_out).T, 
                                   torch.from_numpy(b_attn_out), HE)
        
        attn_time = (time.perf_counter() - attn_start) * 1000
        print(f"      Attention: {attn_time:.2f} ms")
        
        # Residual (approximate: skip encrypted residual due to complexity)
        # In practice, would need to re-encrypt hidden after decryption
        enc_hidden = enc_attn_proj
        
        # ===== Feed-Forward Network =====
        print(f"    [FFN]")
        ffn_start = time.perf_counter()
        
        W_ffn_in = block['ffn_in'].weight.detach().cpu().numpy().astype(np.float32)  # type: ignore
        b_ffn_in = block['ffn_in'].bias.detach().cpu().numpy().astype(np.float32)  # type: ignore
        W_ffn_out = block['ffn_out'].weight.detach().cpu().numpy().astype(np.float32)  # type: ignore
        b_ffn_out = block['ffn_out'].bias.detach().cpu().numpy().astype(np.float32)  # type: ignore
        
        W_ffn_in = (W_ffn_in * SCALE_FACTOR).astype(np.int32)
        b_ffn_in = (b_ffn_in * SCALE_FACTOR).astype(np.int32)
        W_ffn_out = (W_ffn_out * SCALE_FACTOR).astype(np.int32)
        b_ffn_out = (b_ffn_out * SCALE_FACTOR).astype(np.int32)
        
        # FFN layer 1: hidden @ W_in + b_in (no ReLU in encrypted domain)
        enc_ffn = he_linear(enc_hidden, torch.from_numpy(W_ffn_in).T, 
                            torch.from_numpy(b_ffn_in), HE)
        
        # NOTE: Cannot apply ReLU in encrypted domain
        # Would require: comparison/select → complex HE operations
        # Workaround: decrypt, apply ReLU, re-encrypt (breaks privacy but maintains correctness)
        print(f"      WARNING: Decrypting for ReLU activation (privacy loss)")
        ffn_hidden_plain = decrypt_tensor(enc_ffn, HE)
        ffn_hidden_plain = torch.relu(ffn_hidden_plain.float())
        ffn_hidden_long = ffn_hidden_plain.long()  # Keep in scaled domain
        enc_ffn = encrypt_tensor(ffn_hidden_long, HE, batch=False)
        
        # FFN layer 2: ffn @ W_out + b_out
        enc_hidden = he_linear(enc_ffn, torch.from_numpy(W_ffn_out).T,
                               torch.from_numpy(b_ffn_out), HE)
        
        ffn_time = (time.perf_counter() - ffn_start) * 1000
        print(f"      FFN: {ffn_time:.2f} ms")
        
        block_time = (time.perf_counter() - block_start) * 1000
        timings['transformer_blocks'].append(block_time)
        print(f"    Total block time: {block_time:.2f} ms")
        
        memory_info['num_ciphertexts'] += len(enc_hidden)
        memory_info['peak_ciphertexts'] = max(memory_info['peak_ciphertexts'], len(enc_hidden))
    
    # ========== STEP 3: LM Head ==========
    print("\n[3] LM Head (Language Modeling Head)")
    lm_start = time.perf_counter()
    
    # Apply LM head in encrypted domain first
    W_lm = model.lm_head.weight.detach().cpu().numpy().astype(np.float32)
    b_lm = model.lm_head.bias.detach().cpu().numpy().astype(np.float32)
    
    W_lm = (W_lm * SCALE_FACTOR).astype(np.int32)
    b_lm = (b_lm * SCALE_FACTOR).astype(np.int32)
    
    # Encrypted LM head: hidden @ W_lm^T + b_lm
    enc_logits = he_linear(enc_hidden, torch.from_numpy(W_lm).T, torch.from_numpy(b_lm), HE)
    
    # Decrypt logits (still in scaled domain, but apply softmax on them)
    logits_long = decrypt_tensor(enc_logits, HE)
    logits = logits_long.float() / (SCALE_FACTOR ** 2)  # Unscale (squared due to matrix mult)
    
    lm_time = (time.perf_counter() - lm_start) * 1000
    timings['lm_head'] = lm_time
    print(f"  LM head computed in {lm_time:.2f} ms")
    
    # ========== STEP 4: Top-5 Tokens ==========
    print("\n[4] Top-5 Token Suggestions")
    
    top_5_logits, top_5_indices = torch.topk(logits, k=5)
    top_5_tokens = top_5_indices.cpu().numpy().tolist()
    top_5_logits_vals = top_5_logits.cpu().numpy().tolist()
    
    for rank, (token_id, logit) in enumerate(zip(top_5_tokens, top_5_logits_vals), 1):
        print(f"  {rank}. Token {token_id}: logit {logit:.2f}")
    
    # ========== MEMORY PROFILING ==========
    print("\n[5] Memory Profiling")
    est_ciphertext_size_kb = 1.0  # Rough estimate: 1KB per ciphertext
    est_total_mb = (memory_info['peak_ciphertexts'] * est_ciphertext_size_kb) / 1024
    memory_info['est_ciphertext_size_mb'] = est_total_mb
    
    print(f"  Peak simultaneous ciphertexts: {memory_info['peak_ciphertexts']}")
    print(f"  Estimated peak memory (ciphertexts only): {est_total_mb:.2f} MB")
    print(f"  Note: Full model with weights would require ~50-200MB additional")
    
    # ========== TIMING SUMMARY ==========
    print("\n[6] Timing Summary")
    total_time = embedding_time + sum(timings['transformer_blocks']) + lm_time
    timings['total'] = total_time
    
    print(f"  Embedding:          {embedding_time:7.2f} ms")
    for i, block_time in enumerate(timings['transformer_blocks']):
        print(f"  Transformer Block {i}: {block_time:7.2f} ms")
    print(f"  LM Head:            {lm_time:7.2f} ms")
    print(f"  ─────────────────────────────")
    print(f"  Total Inference:    {total_time:7.2f} ms")
    
    plaintext_reference = model(input_ids).squeeze()
    ref_top_5 = torch.topk(plaintext_reference, k=5)[1].cpu().numpy().tolist()
    
    print(f"\n  Plaintext reference top-5: {ref_top_5}")
    print(f"  HE top-5:                  {top_5_tokens}")
    print(f"  Match: {top_5_tokens == ref_top_5}")
    
    # ========== LIMITATIONS & NEXT STEPS ==========
    print("\n" + "=" * 80)
    print("LIMITATIONS & NEXT STEPS")
    print("=" * 80)
    print("""
⚠️  CRITICAL LIMITATIONS:
  1. Batch size = 1 only (no parallel samples)
  2. Sequence length = 1 only (single token generation)
  3. Toy model (tiny vocab, embedding dim, layers)
  4. Approximate softmax (no true probability)
  5. Privacy loss: ReLU requires decryption (breaks end-to-end encryption)
  6. Integer quantization (rounding errors)
  7. Noise growth limits depth (~2-3 blocks before decryption failure)
  8. Memory: ~100MB+ for typical model sizes

🔧 NEXT STEPS TO PRODUCTION:
  1. BATCHING:
     - Use SIMD packing in BFV/CKKS slots (NTT optimization)
     - Pack multiple samples/tokens in single ciphertext
     - Reduce per-sample encryption overhead
  
  2. SOFTMAX & ACTIVATION:
     - Polynomial approximation (Taylor/Chebyshev)
     - Approximate exp via polynomial evaluation (Pyfhel supports)
     - Keep ReLU encrypted via sign function + multiplication
  
  3. NOISE MANAGEMENT:
     - Use CKKS (floating point) instead of BFV for better precision
     - Implement modulus switching (rescale between layers)
     - Reduce parameters: quantization, pruning
  
  4. EFFICIENCY:
     - Pre-compute encrypted weights (if server has them)
     - Use low-rank approximations for weight matrices
     - Implement fast matrix multiplication (block-wise)
  
  5. HARDWARE:
     - GPU-optimized HE libraries (HEAX, cuFHE)
     - FPGA implementations
     - Custom silicon (e.g., CryptoLab's CryptoNets)
  
  6. PROTOCOL DESIGN:
     - Client-server architecture (encrypted queries)
     - Homomorphic evaluation in cloud (oblivious inference)
     - Multi-party computation hybrid approaches

📊 SCALABILITY ESTIMATE:
     Current:    1 token, 2-3 layers, 256 vocab → 100-500 ms, ~50 MB
     With SIMD:  8 tokens, 2-3 layers, 1K vocab → 200-800 ms, ~100 MB
     Optimized:  256 tokens, 4-6 layers, 10K vocab → 1-5 sec, ~500 MB
     Production: Full transformer → ~1-10 min, ~1-10 GB (needs distributed HE)
    """)
    print("=" * 80 + "\n")
    
    return top_5_tokens, timings, memory_info


def demo_he_inference():
    """
    Run a complete HE inference demo.
    
    Creates a toy transformer, encrypts a single input token,
    performs inference, and returns top-5 next tokens.
    """
    print("\n" + "=" * 80)
    print("DEMO: Homomorphic Encryption Inference Pipeline")
    print("=" * 80)
    
    # Setup HE context
    print("\n[Setup] Creating HE context...")
    start = time.perf_counter()
    HE = setup_HE_context(n=2**14, t=65537)
    setup_time = (time.perf_counter() - start) * 1000
    print(f"  HE context ready in {setup_time:.2f} ms")
    
    # Create toy model
    print("\n[Model] Creating toy transformer...")
    model = ToyTransformer(
        vocab_size=256,       # Small vocabulary
        embed_dim=8,          # Tiny embeddings (for speed)
        num_layers=1,         # Single transformer block
        intermediate_dim=16   # Small FFN
    )
    model.eval()
    print(f"  Vocab size: {model.vocab_size}")
    print(f"  Embedding dim: {model.embed_dim}")
    print(f"  Num layers: {model.num_layers}")
    print(f"  Parameter count: {sum(p.numel() for p in model.parameters()):,}")
    
    # Sample input
    print("\n[Input] Selecting random token...")
    input_token = torch.randint(0, model.vocab_size, (1, 1))
    print(f"  Input: token {int(input_token.item())}")
    
    # Run HE inference
    print("\n[Inference] Starting encrypted inference...")
    total_start = time.perf_counter()
    
    top_5, timings, memory_info = full_he_inference(model, input_token, HE)
    
    total_time = (time.perf_counter() - total_start) * 1000
    print(f"\n✓ Inference completed in {total_time:.2f} ms")
    
    # Plaintext reference
    print("\n[Reference] Plaintext model prediction...")
    with torch.no_grad():
        plaintext_logits = model(input_token).squeeze()
    plaintext_top5 = torch.topk(plaintext_logits, k=5)[1].tolist()
    print(f"  Plaintext top-5: {plaintext_top5}")
    print(f"  HE top-5:        {top_5}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    demo_he_inference()
