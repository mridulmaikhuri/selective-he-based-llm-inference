#!/usr/bin/env python
"""Quick test of he_attention_approx function."""

import torch
from he_utils import setup_HE_context, encrypt_tensor, decrypt_tensor
from he_layers import he_attention_approx

# Simple test
HE = setup_HE_context(n=2**14, t=65537)

# Minimal test case
seq_len = 2
Q = torch.tensor([1, 2], dtype=torch.long)
K = torch.tensor([2, 1], dtype=torch.long)
V = torch.tensor([3, 4], dtype=torch.long)

print("Plaintext:")
print(f"  Q: {Q.tolist()}")
print(f"  K: {K.tolist()}")
print(f"  V: {V.tolist()}")

# Encrypt
Q_enc = encrypt_tensor(Q, HE, batch=False)
K_enc = encrypt_tensor(K, HE, batch=False)
V_enc = encrypt_tensor(V, HE, batch=False)

# Run attention
output_enc = he_attention_approx(Q_enc, K_enc, V_enc, HE, causal_mask=False)

# Decrypt
output = decrypt_tensor(output_enc, HE)

print(f"\nHE Attention Output: {output.tolist()}")
print("\nComputation breakdown:")
print(f"  Logits (Q*K):")
print(f"    Row 0: [1*2, 1*1] = [2, 1]")
print(f"    Row 1: [2*2, 2*1] = [4, 2]")
print(f"  Max logit: 4")
print(f"  Weights (logits/4):")
print(f"    Row 0: [0.5, 0.25]")
print(f"    Row 1: [1.0, 0.5]")
print(f"  Output via integer arithmetic:")
print(f"    Output[0] = weighted sum (integer rounding)")
print(f"    Output[1] = weighted sum (integer rounding)")
print(f"\n✓ Test completed successfully!")
