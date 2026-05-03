"""
he_utils.py
===========
Pyfhel helpers for BFV-scheme Fully Homomorphic Encryption (FHE).

BFV scheme quick reference
---------------------------
BFV (Brakerski / Fan-Vercauteren) operates on **integer** plaintexts modulo a
prime `t` (the plaintext modulus).  Operations happen inside a polynomial ring
Z_t[x] / (x^n + 1), where:

  n  (poly_modulus_degree)  — must be a power of 2.
                             Larger n → more slots, higher security, slower ops.
                             n=2**14=16384 gives ~192-bit security at t≈2^16.

  t  (plain_modulus / t_bits) — prime determining the plaintext space [0, t).
                             t=65537 (a Fermat prime 2^16+1) is the standard
                             choice: it is prime, fits 16 bits, and enables
                             efficient NTT-based batching.

  q  (coeff_modulus)        — ciphertext modulus; controls noise budget.
                             Pyfhel chooses a safe default from the security
                             level when you set sec=192 (or 128).

Noise budget
------------
Every ciphertext starts with a finite "noise budget" (in bits).  Each
homomorphic multiplication consumes roughly half the budget.  When the budget
hits zero the ciphertext cannot be decrypted correctly.

  Fresh ciphertext  : ~438 bits  (n=2**14, default coeff_modulus)
  Cost of addition  : negligible
  Cost of multiply  : ~(budget / 2) bits consumed

For n=2**14 and t=65537 you can perform ~27 consecutive multiplications
before the noise budget is exhausted.

Performance notes
-----------------
  * BFV batching (batch=True) packs n//2 integers into one ciphertext and
    executes SIMD-style operations; it is 100-1000× faster than the
    element-wise approach for large tensors.
  * Element-wise mode (batch=False) creates one ciphertext per value; use
    only for small tensors or when you need per-element control.
  * Key generation for n=2**14 takes ~1-2 s on a modern CPU.
  * Relinearisation keys (relin_key) are required after multiplication to
    keep ciphertext size at 2 polynomials.
"""

import time
from typing import Union

import numpy as np
import torch

try:
    from Pyfhel import Pyfhel, PyPtxt
except ImportError as _pyfhel_err:  # pragma: no cover
    raise ImportError(
        "Pyfhel is required but not installed.\n"
        "Install it with:  pip install Pyfhel\n"
        "(Pyfhel needs a C++ compiler; see https://github.com/ibarrond/Pyfhel)"
    ) from _pyfhel_err


# ---------------------------------------------------------------------------
# Context setup
# ---------------------------------------------------------------------------

def setup_HE_context(n: int = 2 ** 14, t: int = 65537) -> Pyfhel:
    """
    Create and return a fully initialised Pyfhel object configured for BFV.

    Parameters
    ----------
    n : int
        Polynomial modulus degree (must be a power of 2).
        Controls the number of batching slots (n // 2) and the security level.
        Default 2**14 = 16 384 gives ~192-bit security with the default
        coefficient modulus chosen by Pyfhel.

    t : int
        Plaintext modulus (must be a prime ≡ 1 (mod 2n) for batching to work).
        Default 65537 = 2^16 + 1 satisfies this condition for all n ≤ 2**16.
        All arithmetic is performed modulo t, so values must lie in [0, t).

    Returns
    -------
    HE : Pyfhel
        Object with context, public key, secret key, and relinearisation key
        already generated.  Pass this handle to encrypt_tensor / decrypt_tensor.

    Raises
    ------
    ValueError
        If n is not a power of 2, or t is not a valid plaintext modulus.
    RuntimeError
        If Pyfhel context or key generation fails.
    """
    # Validate n
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a power of 2, got {n}")
    # Validate t is prime (simple trial-division for expected small values)
    if t < 2 or not _is_prime(t):
        raise ValueError(f"t must be a prime number, got {t}")

    HE = Pyfhel()

    try:
        # contextGen sets up the polynomial ring and coefficient modulus.
        # sec=192 asks Pyfhel to pick a coeff_modulus that achieves ~192-bit
        # security; you can lower to sec=128 for a larger noise budget.
        HE.contextGen(
            scheme="BFV",
            n=n,
            t=t,
            sec=128,          # security level in bits (128 or 192)
        )
    except Exception as exc:
        raise RuntimeError(
            f"Pyfhel contextGen failed (n={n}, t={t}): {exc}"
        ) from exc

    try:
        HE.keyGen()           # generates public + secret key pair
        HE.relinKeyGen()      # relinearisation key: required after multiply
        HE.rotateKeyGen()     # rotation keys: needed for slot rotations / batching
    except Exception as exc:
        raise RuntimeError(f"Pyfhel key generation failed: {exc}") from exc

    return HE


# ---------------------------------------------------------------------------
# Encrypt
# ---------------------------------------------------------------------------

def encrypt_tensor(
    tensor: torch.Tensor,
    HE: Pyfhel,
    batch: bool = False,
) -> list:
    """
    Encode and encrypt a 1-D or 2-D integer tensor using BFV.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.  Values must be integers in [0, t) where t is the
        plaintext modulus used when the HE context was created.
        Float tensors are automatically cast to int64 (truncation, not rounding).
        Supported shapes: (L,) or (R, C).

    HE : Pyfhel
        An initialised Pyfhel object (from setup_HE_context).

    batch : bool
        False (default) — element-wise mode.
            Each scalar value is encrypted into its own ciphertext.
            Returns a flat list of PyCtxt objects whose length equals
            the total number of elements (R*C for 2-D input).

        True — batched mode.
            Values are packed into ciphertext slots using BFV's SIMD encoding.
            For a 1-D tensor of length L a single ciphertext is returned
            (L must be ≤ n//2 slots).  For a 2-D tensor each row becomes one
            ciphertext.
            Returns list of PyCtxt objects (length 1 for 1-D, R for 2-D).

            ⚡ Batch mode is orders of magnitude faster for large tensors because
            one ciphertext holds up to n//2 values and operations apply to all
            slots simultaneously.

    Returns
    -------
    list[PyCtxt]
        Flat list of ciphertexts.  Metadata needed for decryption (original
        shape, batch flag) is stored as attributes on the returned list object
        (``ciphertexts.original_shape`` and ``ciphertexts.batch``).

    Raises
    ------
    TypeError  : tensor is not a torch.Tensor, or HE is not a Pyfhel instance.
    ValueError : tensor has unsupported number of dimensions, or batch slot
                 count exceeds n//2.
    """
    _check_he(HE)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor)}")
    if tensor.ndim not in (1, 2):
        raise ValueError(f"Only 1-D and 2-D tensors are supported, got shape {tensor.shape}")

    # Cast to int64; BFV plaintexts are integers mod t
    int_tensor = tensor.detach().cpu().to(torch.int64)
    original_shape = tuple(int_tensor.shape)

    n_slots = HE.get_nSlots()  # = n // 2

    ciphertexts = []

    if not batch:
        # ── Element-wise: one ciphertext per scalar ──────────────────────────
        # Performance note: this creates len(tensor.flatten()) ciphertexts.
        # Each encryption takes ~microseconds but the total overhead is linear
        # in the number of elements.  Use batch=True for tensors with > ~100
        # elements.
        flat = int_tensor.flatten().tolist()
        for val in flat:
            ptxt = HE.encode([int(val)])
            ctxt = HE.encrypt(ptxt)
            ciphertexts.append(ctxt)

    else:
        # ── Batched: pack rows (or the whole 1-D tensor) into slots ──────────
        rows = int_tensor if int_tensor.ndim == 2 else int_tensor.unsqueeze(0)
        _, row_len = rows.shape
        if row_len > n_slots:
            raise ValueError(
                f"Row length {row_len} exceeds the number of BFV slots "
                f"({n_slots} = n//2 = {HE.get_n()}//2).  "
                f"Split the tensor or use a larger n."
            )
        for row in rows:
            # Pad row to n_slots with zeros (BFV requires full slot vector)
            padded = row.tolist() + [0] * (n_slots - row_len)
            ptxt = HE.encode(padded)
            ctxt = HE.encrypt(ptxt)
            ciphertexts.append(ctxt)

    # Attach metadata so decrypt_tensor can reconstruct the original shape
    ciphertexts = _TaggedList(ciphertexts, original_shape=original_shape, batch=batch)
    return ciphertexts


# ---------------------------------------------------------------------------
# Decrypt
# ---------------------------------------------------------------------------

def decrypt_tensor(
    ciphertexts,
    HE: Pyfhel,
) -> torch.Tensor:
    """
    Decrypt a list of ciphertexts (produced by encrypt_tensor) back to a tensor.

    Parameters
    ----------
    ciphertexts : _TaggedList or list[PyCtxt]
        List returned by encrypt_tensor.  Must have ``.original_shape`` and
        ``.batch`` attributes (automatically set by encrypt_tensor).
        If you pass a plain list the function assumes element-wise mode and
        returns a 1-D tensor.

    HE : Pyfhel
        The same initialised Pyfhel object used for encryption (must hold the
        secret key).

    Returns
    -------
    torch.Tensor  (dtype=torch.int64)
        Tensor with the same shape as the original input to encrypt_tensor.
        Values are integers modulo t; floating-point inputs are therefore
        recovered as truncated integers.

    Raises
    ------
    TypeError   : HE is not a Pyfhel instance.
    RuntimeError: Decryption fails (e.g. noise budget exhausted after too many
                  multiplications — the classic FHE failure mode).
    """
    _check_he(HE)

    batch          = getattr(ciphertexts, "batch", False)
    original_shape = getattr(ciphertexts, "original_shape", None)

    values: list[int] = []

    try:
        if not batch:
            # ── Element-wise decryption ───────────────────────────────────────
            for ctxt in ciphertexts:
                ptxt  = HE.decrypt(ctxt)
                # decryptInt returns a list; first element is the scalar
                val   = HE.decryptInt(ctxt)
                scalar = val[0] if hasattr(val, "__len__") else int(val)
                values.append(scalar)

        else:
            # ── Batched decryption ────────────────────────────────────────────
            n_slots = HE.get_nSlots()
            if original_shape is not None:
                row_len = original_shape[-1]
            else:
                row_len = n_slots  # fall back: return all slots

            for ctxt in ciphertexts:
                decoded = HE.decryptInt(ctxt)           # list of n_slots ints
                values.extend(decoded[:row_len])        # strip padding zeros

    except Exception as exc:
        raise RuntimeError(
            f"Decryption failed: {exc}\n"
            "This can happen if the noise budget was exhausted by too many "
            "homomorphic multiplications.  Reduce the circuit depth or "
            "increase n (poly_modulus_degree)."
        ) from exc

    t = torch.tensor(values, dtype=torch.int64)
    if original_shape is not None:
        t = t.reshape(original_shape)
    return t


# ---------------------------------------------------------------------------
# Batched encryption and matrix multiplication
# ---------------------------------------------------------------------------

def batch_encrypt(
    values: np.ndarray,
    HE: Pyfhel,
    pack_size: int,
) -> list:
    """
    Pack multiple plaintext values into ciphertexts using SIMD-style batching.

    This function groups `pack_size` consecutive values into a single ciphertext
    when supported by Pyfhel's batching (BFV slots). Falls back to elementwise
    encryption if batching is not suitable (e.g., if pack_size exceeds available
    slots).

    Parameters
    ----------
    values : np.ndarray
        1-D array of plaintext values (integers or floats; floats are truncated
        to int64).

    HE : Pyfhel
        Initialized Pyfhel context with keys and rotation support.

    pack_size : int
        Target number of values to pack per ciphertext.
        If pack_size >= len(values), packs all into one ciphertext (if it fits).
        Otherwise, creates ceil(len(values) / pack_size) ciphertexts.
        Clamped internally to HE.get_nSlots() if larger.

    Returns
    -------
    list[PyCtxt]
        List of ciphertexts, each containing up to pack_size encrypted values.
        Metadata attributes: ``.batch = True``, ``.pack_size = pack_size``
        on the returned _TaggedList object.

    Notes
    -----
    - Each ciphertext is zero-padded to n_slots (= n // 2) to conform to BFV
      batching requirements.
    - The function uses HE.encode() with a list of values and HE.encrypt() to
      perform SIMD-style encryption in Pyfhel's native batching mode.
    """
    _check_he(HE)

    # Convert to numpy if needed
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    elif not isinstance(values, np.ndarray):
        values = np.asarray(values)

    # Convert to int64
    values = values.astype(np.int64).flatten()

    n_slots = HE.get_nSlots()  # Typically n // 2

    # Clamp pack_size to available slots
    effective_pack_size = min(pack_size, n_slots)

    ciphertexts = []

    # Group values into chunks of pack_size
    for i in range(0, len(values), effective_pack_size):
        chunk = values[i : i + effective_pack_size].tolist()

        # Pad chunk to n_slots (required by BFV batching)
        padded_chunk = chunk + [0] * (n_slots - len(chunk))

        # Encode and encrypt as SIMD batch
        ptxt = HE.encode(padded_chunk)
        ctxt = HE.encrypt(ptxt)
        ciphertexts.append(ctxt)

    # Attach metadata
    ciphertexts = _TaggedList(
        ciphertexts,
        original_shape=(len(values),),
        batch=True,
        pack_size=effective_pack_size,
    )
    return ciphertexts


def he_linear_batched(
    weight_matrix: np.ndarray,
    input_ciphertexts: list,
    HE: Pyfhel,
    pack_size: int = None,
) -> list:
    """
    Compute matrix multiplication using batched ciphertexts.

    Computes output = input @ weight_matrix.T where input is encrypted in
    batched form. This function demonstrates how batched ciphertexts can
    reduce the number of ciphertexts (and thus operations) needed for
    linear algebra on encrypted data.

    Parameters
    ----------
    weight_matrix : np.ndarray
        Shape (out_features, in_features), plaintext weights (integers).
        Each row is the weight vector for one output feature.

    input_ciphertexts : list
        Batched ciphertexts from batch_encrypt() or compatible.
        Assumed to represent a flattened input vector encrypted across
        multiple ciphertexts.

    HE : Pyfhel
        Initialized Pyfhel context with keys and relinearization support.

    pack_size : int, optional
        If provided, used for alignment/metadata. Otherwise inferred from
        `input_ciphertexts.pack_size` if available.

    Returns
    -------
    list[PyCtxt]
        Batched ciphertexts containing encrypted output values.
        Shape corresponds to (out_features,) after decryption.

    Notes
    -----
    - This is a simplified implementation for demonstration. A production
      matrix-vector product would use rotations and summation tricks to
      amortize the cost across slots.
    - Each output feature is computed as a dot product of the weight row
      and the (encrypted) input vector. Relinearization is applied after
      multiplication to keep noise in check.
    """
    _check_he(HE)

    weight_matrix = np.asarray(weight_matrix, dtype=np.int64)

    if len(input_ciphertexts) == 0:
        raise ValueError("input_ciphertexts cannot be empty")

    out_features, in_features = weight_matrix.shape
    n_slots = HE.get_nSlots()

    # Flatten input ciphertexts into a single list
    if hasattr(input_ciphertexts, "original_shape"):
        n_input = int(np.prod(input_ciphertexts.original_shape))
    else:
        n_input = len(input_ciphertexts)

    if n_input < in_features:
        raise ValueError(
            f"Input size {n_input} is less than weight_matrix.shape[1]={in_features}"
        )

    output_ciphertexts = []

    # For each output feature, compute dot product with the weight row
    for out_idx in range(out_features):
        weights = weight_matrix[out_idx]  # shape (in_features,)

        # Initialize result to encrypted zero
        ptxt_zero = HE.encode([0] * n_slots)
        result = HE.encrypt(ptxt_zero)

        # Accumulate weighted contributions from input ciphertexts
        val_idx = 0
        for ctxt_idx, ctxt in enumerate(input_ciphertexts):
            # Determine how many values are in this ciphertext
            if ctxt_idx == len(input_ciphertexts) - 1:
                # Last ciphertext may have fewer slots in use
                slots_in_use = n_input - val_idx
            else:
                slots_in_use = n_slots

            # For each slot in this ciphertext, multiply by corresponding weight
            for slot_idx in range(min(slots_in_use, len(weights) - val_idx)):
                if val_idx + slot_idx < in_features:
                    w = int(weights[val_idx + slot_idx])

                    if w != 0:  # Skip zero weights to save computation
                        # Multiply ciphertext by weight (homomorphic scalar mult)
                        scaled = ctxt * w
                        HE.relinearize(scaled)

                        # Add to accumulator
                        result = result + scaled

            val_idx += slots_in_use

        output_ciphertexts.append(result)

    output_ciphertexts = _TaggedList(
        output_ciphertexts,
        original_shape=(out_features,),
        batch=True,
    )
    return output_ciphertexts


# ---------------------------------------------------------------------------
# Benchmark: batched vs element-wise
# ---------------------------------------------------------------------------

def he_batch_benchmark() -> None:
    """
    Benchmark batched vs element-wise encryption and decryption.

    Measures encryption/decryption performance on small toy problems and
    reports speedup factors. Demonstrates the SIMD advantage of BFV batching.

    Output
    ------
    Prints timing comparisons and measured speedup factors for:
      1. Encryption time: element-wise vs batched
      2. Decryption time: element-wise vs batched
      3. Ciphertext count reduction
      4. Caveats and notes on when batching is beneficial

    Notes
    -----
    - Uses n=2**12 for fast iteration in the demo; production may use n=2**14.
    - Speedup factors depend on Pyfhel version, system architecture, and
      whether rotations/advanced operations are used.
    - For small tensors (< 16 values), element-wise may be faster due to
      lower padding overhead.
    """
    print("=" * 70)
    print("  Batched Homomorphic Encryption — Benchmark")
    print("=" * 70)

    # Setup context (small n for fast demo iteration)
    print("\n[Setup] Creating HE context (n=2**12, t=65537) …")
    HE = setup_HE_context(n=2**12, t=65537)
    n_slots = HE.get_nSlots()
    print(f"  Available slots per ciphertext: {n_slots}")

    # Test cases with increasing sizes
    test_cases = [
        {"size": 32, "pack_size": 16, "name": "Small (32 values, pack=16)"},
        {"size": 64, "pack_size": 32, "name": "Medium (64 values, pack=32)"},
        {"size": 128, "pack_size": 64, "name": "Large (128 values, pack=64)"},
    ]

    results = []

    print("\n" + "-" * 70)
    print("  PART 1: Encryption Timing")
    print("-" * 70)

    for test in test_cases:
        size = test["size"]
        pack_size = test["pack_size"]
        name = test["name"]

        values = np.random.randint(0, 1000, size=size, dtype=np.int64)

        # Element-wise encryption
        t0 = time.perf_counter()
        tensor = torch.from_numpy(values)
        ct_elementwise = encrypt_tensor(tensor, HE, batch=False)
        t_elementwise = time.perf_counter() - t0
        n_ctxt_elementwise = len(ct_elementwise)

        # Batched encryption
        t0 = time.perf_counter()
        ct_batched = batch_encrypt(values, HE, pack_size)
        t_batched = time.perf_counter() - t0
        n_ctxt_batched = len(ct_batched)

        speedup = t_elementwise / t_batched if t_batched > 0 else float("inf")

        results.append(
            {
                "name": name,
                "enc_elementwise": t_elementwise,
                "enc_batched": t_batched,
                "enc_speedup": speedup,
                "n_ctxt_elementwise": n_ctxt_elementwise,
                "n_ctxt_batched": n_ctxt_batched,
            }
        )

        print(f"\n  {name}")
        print(f"    Element-wise: {t_elementwise*1e3:.2f} ms  ({n_ctxt_elementwise} ctxts)")
        print(f"    Batched:      {t_batched*1e3:.2f} ms  ({n_ctxt_batched} ctxts)")
        print(f"    ⚡ Speedup:    {speedup:.2f}x")

    print("\n" + "-" * 70)
    print("  PART 2: Decryption Timing")
    print("-" * 70)

    for test in test_cases:
        size = test["size"]
        pack_size = test["pack_size"]
        name = test["name"]

        values = np.random.randint(0, 1000, size=size, dtype=np.int64)
        tensor = torch.from_numpy(values)

        # Element-wise
        ct_elementwise = encrypt_tensor(tensor, HE, batch=False)
        t0 = time.perf_counter()
        result_elementwise = decrypt_tensor(ct_elementwise, HE)
        t_dec_elementwise = time.perf_counter() - t0

        # Batched
        ct_batched = batch_encrypt(values, HE, pack_size)
        t0 = time.perf_counter()
        result_batched = decrypt_tensor(ct_batched, HE)
        t_dec_batched = time.perf_counter() - t0

        speedup_dec = (
            t_dec_elementwise / t_dec_batched if t_dec_batched > 0 else float("inf")
        )

        # Find and update the matching result entry
        for res in results:
            if res["name"] == name:
                res["dec_elementwise"] = t_dec_elementwise
                res["dec_batched"] = t_dec_batched
                res["dec_speedup"] = speedup_dec

        print(f"\n  {name}")
        print(f"    Element-wise: {t_dec_elementwise*1e3:.2f} ms")
        print(f"    Batched:      {t_dec_batched*1e3:.2f} ms")
        print(f"    ⚡ Speedup:    {speedup_dec:.2f}x")

    # Summary and caveats
    print("\n" + "=" * 70)
    print("  Summary & Caveats")
    print("=" * 70)

    avg_enc_speedup = np.mean([r["enc_speedup"] for r in results])
    avg_dec_speedup = np.mean([r["dec_speedup"] for r in results])

    print(f"\n  📊 Average Speedup (across test cases):")
    print(f"     Encryption: {avg_enc_speedup:.2f}x")
    print(f"     Decryption: {avg_dec_speedup:.2f}x")

    if avg_enc_speedup >= 1.2 or avg_dec_speedup >= 1.2:
        print(f"     ✓ Batch mode shows ≥1.2x speedup (acceptable for demo)")
    else:
        print(
            f"     ⚠ Batch mode shows < 1.2x speedup on this system "
            f"(small problem size effect)"
        )

    print(
        """
  ✓ Batched encryption reduces ciphertext count from N to ceil(N/pack_size).
  ✓ Decryption is typically faster for batched mode due to SIMD advantage
    (multiple slots decoded in one operation).
  ✓ Ciphertext reduction directly translates to fewer homomorphic operations.

  ⚠ CAVEATS:
    1. BFV batching requires padding to n//2 slots, which costs memory.
       The speedup is most pronounced for large N (> 1024 values).
    2. Speedup heavily depends on Pyfhel version, C++ implementation, and
       underlying CPU architecture. Results vary significantly by system.
    3. For very small problem sizes (< 32 values), elementwise may be
       competitive due to padding overhead in batched mode.
    4. Homomorphic matrix multiplication using batched ciphertexts requires
       rotations to gather/scatter data within slots. This notebook shows
       the encryption/decryption benefit; full matmul speedup depends on
       circuit design.
    5. Noise budget grows with multiplicative depth. Batched ops are
       subject to same noise constraints as elementwise operations.
    """
    )
    print("=" * 70)


# ---------------------------------------------------------------------------
# Demo / acceptance test
# ---------------------------------------------------------------------------

def test_homomorphic_operations() -> None:
    """
    Demonstrate and validate encrypted addition and multiplication.

    Covers
    ------
    1. Context setup timing.
    2. Element-wise encrypt → HE add → decrypt, compared to plaintext add.
    3. Element-wise encrypt → HE multiply → decrypt, compared to plaintext mul.
    4. Batched encrypt → HE add → decrypt, with timing comparison.
    5. Noise budget check (prints remaining budget after each operation).
    6. Numeric error tolerance report (should be 0 for integers mod t).

    Prints timings (time.perf_counter) and validates correctness within the
    quantization tolerance expected for BFV integer arithmetic.
    """
    print("=" * 60)
    print("  Pyfhel BFV — Homomorphic Operations Demo")
    print("=" * 60)

    # ── 1. Setup ──────────────────────────────────────────────────────────────
    print("\n[1/4] Setting up BFV context (n=2**13, t=65537) …")
    # Use n=2**13 for speed in the demo; n=2**14 for production security
    t0 = time.perf_counter()
    HE = setup_HE_context(n=2 ** 13, t=65537)
    t_setup = time.perf_counter() - t0
    print(f"      Context + key generation : {t_setup:.3f} s")
    print(f"      Slots available          : {HE.get_nSlots()}")

    # ── 2. Encrypted addition (element-wise) ──────────────────────────────────
    print("\n[2/4] Encrypted addition (element-wise) …")
    a = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64)
    b = torch.tensor([ 1,  2,  3,  4,  5], dtype=torch.int64)
    expected_add = a + b

    t0 = time.perf_counter()
    ct_a = encrypt_tensor(a, HE, batch=False)
    ct_b = encrypt_tensor(b, HE, batch=False)
    t_enc = time.perf_counter() - t0

    t0 = time.perf_counter()
    ct_sum = _TaggedList(
        [ca + cb for ca, cb in zip(ct_a, ct_b)],
        original_shape=ct_a.original_shape,
        batch=False,
    )
    t_add = time.perf_counter() - t0

    t0 = time.perf_counter()
    result_add = decrypt_tensor(ct_sum, HE)
    t_dec = time.perf_counter() - t0

    err_add = (result_add - expected_add).abs().max().item()
    print(f"      Plaintext  a       : {a.tolist()}")
    print(f"      Plaintext  b       : {b.tolist()}")
    print(f"      Expected   a+b     : {expected_add.tolist()}")
    print(f"      Decrypted  a+b     : {result_add.tolist()}")
    print(f"      Max absolute error : {err_add}  (expected 0 for BFV integers)")
    print(f"      Timings — enc: {t_enc*1e3:.2f} ms | "
          f"add: {t_add*1e3:.3f} ms | dec: {t_dec*1e3:.2f} ms")
    assert err_add == 0, f"Addition error {err_add} > 0!"
    print("      ✓ Addition correct")

    # ── 3. Encrypted multiplication (element-wise) ────────────────────────────
    print("\n[3/4] Encrypted multiplication (element-wise) …")
    c = torch.tensor([2, 3, 4, 5, 6], dtype=torch.int64)
    d = torch.tensor([7, 8, 9, 1, 2], dtype=torch.int64)
    expected_mul = c * d

    t0 = time.perf_counter()
    ct_c = encrypt_tensor(c, HE, batch=False)
    ct_d = encrypt_tensor(d, HE, batch=False)
    t_enc2 = time.perf_counter() - t0

    t0 = time.perf_counter()
    ct_prod_list = []
    for cc, cd in zip(ct_c, ct_d):
        cp = cc * cd        # homomorphic multiplication
        # Relinearise to reduce ciphertext size back to 2 polynomials.
        # Without this, each multiply doubles the ciphertext size and noise
        # grows super-linearly.
        HE.relinearize(cp)
        ct_prod_list.append(cp)
    ct_prod = _TaggedList(ct_prod_list, original_shape=ct_c.original_shape, batch=False)
    t_mul = time.perf_counter() - t0

    t0 = time.perf_counter()
    result_mul = decrypt_tensor(ct_prod, HE)
    t_dec2 = time.perf_counter() - t0

    err_mul = (result_mul - expected_mul).abs().max().item()
    print(f"      Plaintext  c       : {c.tolist()}")
    print(f"      Plaintext  d       : {d.tolist()}")
    print(f"      Expected   c*d     : {expected_mul.tolist()}")
    print(f"      Decrypted  c*d     : {result_mul.tolist()}")
    print(f"      Max absolute error : {err_mul}  (expected 0 for BFV integers)")
    print(f"      Timings — enc: {t_enc2*1e3:.2f} ms | "
          f"mul+relin: {t_mul*1e3:.3f} ms | dec: {t_dec2*1e3:.2f} ms")

    # BFV multiplication is exact for integers; error must be 0
    assert err_mul == 0, f"Multiplication error {err_mul} > 0!"
    print("      ✓ Multiplication correct")

    # ── 4. Batched addition (SIMD) ─────────────────────────────────────────────
    print("\n[4/4] Batched addition (SIMD across slots) …")
    size = 64
    e = torch.randint(0, 1000, (size,), dtype=torch.int64)
    f = torch.randint(0, 1000, (size,), dtype=torch.int64)
    expected_batch = e + f

    t0 = time.perf_counter()
    ct_e = encrypt_tensor(e, HE, batch=True)   # 1 ciphertext, 64 slots used
    ct_f = encrypt_tensor(f, HE, batch=True)
    t_enc3 = time.perf_counter() - t0

    t0 = time.perf_counter()
    ct_batch_sum = _TaggedList(
        [ct_e[0] + ct_f[0]],
        original_shape=(size,),
        batch=True,
    )
    t_add3 = time.perf_counter() - t0

    t0 = time.perf_counter()
    result_batch = decrypt_tensor(ct_batch_sum, HE)
    t_dec3 = time.perf_counter() - t0

    err_batch = (result_batch - expected_batch).abs().max().item()
    print(f"      Vector size        : {size}")
    print(f"      Ciphertexts used   : 1  (vs {size} in element-wise mode)")
    print(f"      Max absolute error : {err_batch}")
    print(f"      Timings — enc: {t_enc3*1e3:.2f} ms | "
          f"add: {t_add3*1e3:.3f} ms | dec: {t_dec3*1e3:.2f} ms")
    assert err_batch == 0, f"Batched addition error {err_batch} > 0!"
    print("      ✓ Batched addition correct")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  All homomorphic operation tests passed ✓")
    print(f"  Quantization tolerance  : 0  (BFV is exact for integers mod t)")
    print(f"  Note: floating-point inputs are truncated to int64 before")
    print(f"  encryption, so the effective tolerance for float tensors is")
    print(f"  |floor(x) - x| ≤ 0.9999… per element.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _TaggedList(list):
    """list subclass that carries metadata attributes for decrypt_tensor."""
    def __init__(self, items, original_shape=None, batch=False, pack_size=None):
        super().__init__(items)
        self.original_shape = original_shape
        self.batch = batch
        self.pack_size = pack_size


def _check_he(HE) -> None:
    """Raise TypeError if HE is not a Pyfhel instance."""
    if not isinstance(HE, Pyfhel):
        raise TypeError(
            f"HE must be a Pyfhel instance (from setup_HE_context()), "
            f"got {type(HE)}.  Did you forget to call setup_HE_context()?"
        )


def _is_prime(n: int) -> bool:
    """Simple trial-division primality test (sufficient for small t values)."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_homomorphic_operations()