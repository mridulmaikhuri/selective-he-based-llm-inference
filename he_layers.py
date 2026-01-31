"""
Homomorphic Encryption layers for neural networks using Pyfhel.

This module provides encrypted linear layer operations that perform
matrix-vector multiplication over encrypted data.

Key Computational Properties:
- Matrix-vector product with encrypted vector, plaintext weights
- Output shape: (out_features,) as encrypted vector
- Cost: O(in_features * out_features) ciphertext operations
- Ciphertext growth: One multiplication per output increases noise
- Batching: Currently handles batch_size=1 only (single vector)
  For larger batches, would need SIMD slot optimization

Performance Characteristics:
- Encryption/decryption dominates for small networks
- HE operations ~1000x slower than plaintext PyTorch
- Memory overhead: ciphertexts >> plaintext weights
"""

import torch
import numpy as np
import time
from typing import Union, List
from Pyfhel import Pyfhel


def he_linear(
    encrypted_input: Union[List, np.ndarray],
    plain_weights: Union[torch.Tensor, np.ndarray],
    plain_bias: Union[torch.Tensor, np.ndarray, None],
    HE: Pyfhel
) -> List:
    """
    Homomorphic encryption linear layer: (enc_vec @ plain_W) + plain_b.

    Performs matrix-vector multiplication where the vector is encrypted
    and weights/bias are plaintext. This allows computation over encrypted
    data without key sharing.

    Parameters
    ----------
    encrypted_input : list of ciphertexts
        Encrypted vector of shape (in_features,). Each element is a
        Pyfhel ciphertext from he_utils.encrypt_tensor().
    
    plain_weights : torch.Tensor or np.ndarray
        Weight matrix of shape (in_features, out_features).
        Each column contains weights for one output neuron.
    
    plain_bias : torch.Tensor, np.ndarray, or None
        Bias vector of shape (out_features,). Can be None to skip bias.
        If provided, added homomorphically (as plaintext addition).
    
    HE : Pyfhel
        Initialized Pyfhel object with encryption context.

    Returns
    -------
    list of ciphertexts
        Encrypted output vector of shape (out_features,).
        Each ciphertext encrypts: sum(enc_input[i] * W[i,j]) + bias[j]

    Raises
    ------
    ValueError
        If shapes are incompatible or HE not initialized.
        If encrypted_input is not a list of ciphertexts.

    Notes
    -----
    Computational Cost Analysis:
    - Ciphertext-plaintext multiplication: O(1) time per element
    - Ciphertext addition: O(log(in_features)) depth for tree reduction
    - Total: O(in_features * out_features) multiplications +
             O(out_features * log(in_features)) additions
    
    Noise Growth:
    - Each plaintext multiplication: small noise increase
    - Each ciphertext addition: noise accumulates
    - Total depth: log(in_features) for one layer
    - Multiple layers stack noise → limits network depth to ~3-4 layers

    Example for in_features=10, out_features=5:
    - 50 ciphertext-plaintext multiplications
    - 10 additions per output (could be optimized with tree reduction)
    - Typical runtime: 100-500 ms on modern CPU
    """
    if not isinstance(HE, Pyfhel):
        raise ValueError("HE must be initialized Pyfhel object")
    
    if not isinstance(encrypted_input, list):
        raise ValueError("encrypted_input must be list of ciphertexts")
    
    if len(encrypted_input) == 0:
        raise ValueError("encrypted_input cannot be empty")
    
    # Convert weights to numpy if needed
    if isinstance(plain_weights, torch.Tensor):
        plain_weights = plain_weights.cpu().numpy()
    
    in_features, out_features = plain_weights.shape
    
    if len(encrypted_input) != in_features:
        raise ValueError(
            f"encrypted_input length {len(encrypted_input)} != "
            f"weights shape[0] {in_features}"
        )
    
    # Compute matrix-vector product homomorphically
    # For each output j: result[j] = sum_i(enc_input[i] * W[i,j]) + bias[j]
    encrypted_outputs = []
    
    for j in range(out_features):
        # Get weights for this output neuron
        weights_col = plain_weights[:, j]
        
        # Find first non-zero weight to initialize accumulator
        # This avoids the "transparent ciphertext" problem from multiplying by 0
        acc = None
        for i in range(in_features):
            weight_val = int(weights_col[i])
            if weight_val != 0:
                if acc is None:
                    # Initialize accumulator with first non-zero term
                    acc = encrypted_input[i] * weight_val
                else:
                    # Add subsequent terms
                    term = encrypted_input[i] * weight_val
                    acc = acc + term
            elif acc is not None:
                # Zero weight: skip since multiply by 0 creates transparent ciphertext
                pass
        
        # Handle case where all weights are zero (rare but possible)
        if acc is None:
            # All weights are zero, result is just bias (or zero if no bias)
            if plain_bias is not None:
                bias_val = int(plain_bias[j])
                # Encrypt a dummy value then multiply by 0 and add bias
                # Actually, just encrypt the bias directly
                acc = HE.encrypt(0)
                acc = acc + int(plain_bias[j])
            else:
                # Encrypt zero
                acc = HE.encrypt(0)
        else:
            # Add bias if provided
            if plain_bias is not None:
                bias_val = int(plain_bias[j])
                acc = acc + bias_val
        
        encrypted_outputs.append(acc)
    
    return encrypted_outputs


def test_he_linear_demo() -> None:
    """
    Demonstrate he_linear() with timing comparison to plaintext PyTorch.

    Creates small encrypted linear layer, computes output, measures time,
    and verifies correctness by decryption and comparing to torch.matmul().
    """
    from he_utils import setup_HE_context, encrypt_tensor, decrypt_tensor
    
    print("=" * 80)
    print("Homomorphic Encryption Linear Layer Demo")
    print("=" * 80)
    
    # Setup
    print("\n[Setup] Initializing HE context...")
    start = time.perf_counter()
    HE = setup_HE_context(n=2**14, t=65537)
    setup_time = (time.perf_counter() - start) * 1000
    print(f"  HE context ready in {setup_time:.2f} ms")
    
    # Create test data
    in_features, out_features = 8, 4
    print(f"\nNetwork dimensions: in_features={in_features}, out_features={out_features}")
    
    # Plaintext tensors
    plain_input = torch.tensor([2, 1, 3, 2, 1, 4, 2, 1], dtype=torch.long)
    plain_weights = torch.randn(in_features, out_features).int().long()
    plain_bias = torch.randint(-5, 5, (out_features,), dtype=torch.long)
    
    print(f"\nPlaintext input: {plain_input.tolist()}")
    print(f"Plaintext weights shape: {plain_weights.shape}")
    print(f"Plaintext bias: {plain_bias.tolist()}")
    
    # Encrypt input
    print("\n[Encryption] Encrypting input...")
    start = time.perf_counter()
    encrypted_input = encrypt_tensor(plain_input, HE, batch=False)
    enc_time = (time.perf_counter() - start) * 1000
    print(f"  Encrypted {len(encrypted_input)} elements in {enc_time:.2f} ms")
    
    # Homomorphic linear layer
    print("\n[HE Linear Layer] Computing encrypted output...")
    start = time.perf_counter()
    encrypted_output = he_linear(
        encrypted_input,
        plain_weights,
        plain_bias,
        HE
    )
    he_linear_time = (time.perf_counter() - start) * 1000
    print(f"  HE linear computation: {he_linear_time:.2f} ms")
    print(f"  Operations: {in_features * out_features} multiplications, "
          f"{out_features * (in_features - 1)} additions")
    
    # Decrypt result
    print("\n[Decryption] Decrypting output...")
    start = time.perf_counter()
    decrypted_output = decrypt_tensor(encrypted_output, HE)
    dec_time = (time.perf_counter() - start) * 1000
    print(f"  Decrypted in {dec_time:.2f} ms")
    print(f"  HE result: {decrypted_output.tolist()}")
    
    # Plaintext computation for comparison
    print("\n[Plaintext] Computing PyTorch equivalent...")
    start = time.perf_counter()
    plaintext_output = plain_input @ plain_weights + plain_bias
    plaintext_time = (time.perf_counter() - start) * 1000
    print(f"  PyTorch matmul: {plaintext_time:.4f} ms")
    print(f"  Plaintext result: {plaintext_output.tolist()}")
    
    # Verify correctness
    print("\n[Verification]")
    match = torch.allclose(decrypted_output, plaintext_output)
    print(f"  Results match: {match} ✓" if match else f"  MISMATCH ✗")
    if match:
        print(f"  Decrypted output matches plaintext within tolerance")
    else:
        diff = (decrypted_output - plaintext_output).abs().max()
        print(f"  Max difference: {diff}")
    
    # Timing comparison
    print("\n[Performance Comparison]")
    total_he_time = setup_time + enc_time + he_linear_time + dec_time
    print(f"  HE total time (setup + enc + linear + dec): {total_he_time:.2f} ms")
    print(f"  PyTorch time (linear only): {plaintext_time:.4f} ms")
    slowdown = total_he_time / plaintext_time if plaintext_time > 0 else float('inf')
    print(f"  HE slowdown: {slowdown:.1f}x (including overhead)")
    print(f"  HE linear only: {he_linear_time:.2f} ms")
    linear_slowdown = he_linear_time / plaintext_time if plaintext_time > 0 else float('inf')
    print(f"  HE linear slowdown: {linear_slowdown:.1f}x")
    
    print("\n[Cost Analysis]")
    print(f"  Setup: {setup_time:.2f} ms (one-time)")
    print(f"  Encryption: {enc_time:.2f} ms (per batch)")
    print(f"  HE linear: {he_linear_time:.2f} ms (O(in*out) complexity)")
    print(f"  Decryption: {dec_time:.2f} ms (per batch)")
    print(f"  Plaintext: {plaintext_time:.4f} ms (baseline)")
    
    print("\n[Limitations & Notes]")
    print(f"  - Current batch size: 1 (handles single vector)")
    print(f"  - Ciphertext growth: 1 per output (manageable)")
    print(f"  - Max network depth: ~3-4 layers before noise overflow")
    print(f"  - Memory: {in_features} ciphertexts vs {in_features} numbers")
    print(f"  - Use case: Privacy-preserving inference on untrusted servers")
    
    print("\n" + "=" * 80)
    print("Demo Complete")
    print("=" * 80)


if __name__ == "__main__":
    test_he_linear_demo()
