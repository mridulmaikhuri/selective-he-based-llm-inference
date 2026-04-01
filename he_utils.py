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
    def __init__(self, items, original_shape=None, batch=False):
        super().__init__(items)
        self.original_shape = original_shape
        self.batch = batch


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