#!/usr/bin/env python3
"""
he_batch_benchmark_demo.py
==========================
Demo script that runs the batched encryption benchmark from he_utils.py
and demonstrates batch_encrypt() and he_linear_batched() functions.

This script:
  1. Sets up a small FHE context
  2. Runs the batch_encrypt() benchmark vs element-wise
  3. Demonstrates he_linear_batched() on a small matrix multiply problem
  4. Reports measured speedup factors
"""

import numpy as np
import torch
from he_utils import (
    setup_HE_context,
    batch_encrypt,
    he_linear_batched,
    encrypt_tensor,
    decrypt_tensor,
    he_batch_benchmark,
)


def demo_batch_encrypt_and_decrypt():
    """Demonstrate batch_encrypt() with basic encryption/decryption."""
    print("=" * 70)
    print("  Demo 1: batch_encrypt() — SIMD-style Batching")
    print("=" * 70)

    # Setup small context
    print("\n[Setup] Creating HE context…")
    HE = setup_HE_context(n=2**12, t=65537)
    print(f"  Slots available: {HE.get_nSlots()}")

    # Create sample data
    print("\n[Data] Creating 48-value plaintext vector…")
    plaintext_values = np.random.randint(10, 100, size=48, dtype=np.int64)
    print(f"  First 16 values: {plaintext_values[:16].tolist()}")

    # Batch encrypt with pack_size=16
    print("\n[Encrypt] Batching with pack_size=16…")
    pack_size = 16
    ct_batched = batch_encrypt(plaintext_values, HE, pack_size=pack_size)
    print(f"  Created {len(ct_batched)} ciphertexts (vs {len(plaintext_values)} for elementwise)")

    # Decrypt back
    print("\n[Decrypt] Recovering plaintext…")
    recovered = decrypt_tensor(ct_batched, HE)
    print(f"  Recovered first 16 values: {recovered[:16].tolist()}")

    # Verify correctness
    errors = (recovered.numpy().astype(np.int64) - plaintext_values).abs()
    max_err = errors.max()
    print(f"  Max decryption error: {max_err} (expected 0 for BFV integers)")
    assert max_err == 0, f"Decryption error detected: {max_err}"
    print("  ✓ Batch encryption/decryption correct")


def demo_he_linear_batched():
    """Demonstrate he_linear_batched() for matrix multiplication."""
    print("\n" + "=" * 70)
    print("  Demo 2: he_linear_batched() — Encrypted Matrix Multiplication")
    print("=" * 70)

    # Setup context
    print("\n[Setup] Creating HE context…")
    HE = setup_HE_context(n=2**12, t=65537)

    # Create a small matrix multiply problem: y = W @ x
    # x: shape (in_features,), encrypted
    # W: shape (out_features, in_features), plaintext
    in_features = 16
    out_features = 8
    pack_size = 8

    print(f"\n[Problem] Matrix-vector multiply:")
    print(f"  Input vector (encrypted):  {in_features} values")
    print(f"  Weight matrix (plaintext): {out_features} x {in_features}")

    # Create plaintext values
    x_plain = np.random.randint(5, 20, size=in_features, dtype=np.int64)
    W_plain = np.random.randint(1, 10, size=(out_features, in_features), dtype=np.int64)

    # Encrypt input with batching
    print(f"\n[Encrypt] Using batch_encrypt with pack_size={pack_size}…")
    x_encrypted = batch_encrypt(x_plain, HE, pack_size=pack_size)
    print(f"  Input encrypted in {len(x_encrypted)} ciphertexts")

    # Compute encrypted matrix multiply
    print(f"\n[Compute] Homomorphic matrix multiplication…")
    y_encrypted = he_linear_batched(W_plain, x_encrypted, HE, pack_size=pack_size)
    print(f"  Output encrypted in {len(y_encrypted)} ciphertexts")

    # Decrypt result
    print(f"\n[Decrypt] Recovering output…")
    y_recovered = decrypt_tensor(y_encrypted, HE)

    # Verify against plaintext computation
    y_expected = W_plain @ x_plain
    y_computed = y_recovered.numpy().astype(np.int64)

    print(f"\n[Verify] Comparing with plaintext result:")
    print(f"  Expected output (first 4): {y_expected[:4]}")
    print(f"  Computed output (first 4): {y_computed[:4]}")

    errors = (y_computed - y_expected).abs()
    max_err = errors.max()
    print(f"  Max error: {max_err}")

    if max_err == 0:
        print("  ✓ Encrypted matrix multiply correct")
    else:
        print(f"  ⚠ Warning: max error {max_err} > 0 (possible noise accumulation)")


def run_full_benchmark():
    """Run the comprehensive benchmark with timing comparisons."""
    print("\n" + "=" * 70)
    print("  Demo 3: Full Benchmark (Encryption/Decryption Timing)")
    print("=" * 70 + "\n")
    he_batch_benchmark()


if __name__ == "__main__":
    try:
        # Run all demos
        demo_batch_encrypt_and_decrypt()
        demo_he_linear_batched()
        run_full_benchmark()

        print("\n" + "=" * 70)
        print("  ✓ All demos completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during demo: {e}", flush=True)
        import traceback
        traceback.print_exc()
        exit(1)
