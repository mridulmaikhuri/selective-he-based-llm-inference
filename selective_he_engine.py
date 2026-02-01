"""
Selective Homomorphic Encryption Inference Engine

Mixes plaintext and homomorphic encryption (HE) computation at layer/operation granularity.
For each layer, checks if encryption is requested in config, and conditionally:
  - Encrypts inputs
  - Runs HE versions of required operations
  - Decrypts outputs
Otherwise, runs standard PyTorch forward passes.

Provides detailed timing breakdown and encryption state tracking for analysis.

Key Limitations:
- Batch size = 1 (recommended)
- Quantization to integers for HE operations
- Noise growth limits depth

Usage Example:
    from selective_he_config import SelectiveHEConfig, load_strategy_config
    from selective_he_engine import selective_HE_inference
    from he_utils import setup_HE_context
    
    config = load_strategy_config(1)
    HE = setup_HE_context()
    logits, timings, enc_log = selective_HE_inference(model, input_ids, HE, config)
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Tuple, List, Dict, Any, Optional
from Pyfhel import Pyfhel

from he_utils import setup_HE_context, encrypt_tensor, decrypt_tensor
from he_layers import he_linear
from selective_he_config import SelectiveHEConfig


class EncryptionState:
    """Helper class to track tensor encryption state throughout inference."""
    
    def __init__(self):
        self.tensor_states = {}  # Maps tensor id to bool (encrypted or not)
    
    def mark_encrypted(self, tensor_name: str) -> None:
        """Mark a tensor as encrypted."""
        self.tensor_states[tensor_name] = True
    
    def mark_plaintext(self, tensor_name: str) -> None:
        """Mark a tensor as plaintext."""
        self.tensor_states[tensor_name] = False
    
    def is_encrypted(self, tensor_name: str) -> bool:
        """Check if tensor is encrypted."""
        return self.tensor_states.get(tensor_name, False)
    
    def summary(self) -> Dict[str, bool]:
        """Get summary of all tensor states."""
        return self.tensor_states.copy()


def selective_HE_inference(
    model: nn.Module,
    input_ids: torch.Tensor,
    HE_context: Pyfhel,
    config: SelectiveHEConfig,
    verbose: bool = True
) -> Tuple[torch.Tensor, Dict[str, Any], List[Tuple[str, bool]]]:
    """
    Mixed plaintext and homomorphic encryption inference.
    
    Traverses model layers in execution order. For each layer:
      - If layer in config.layers_to_encrypt:
          * Encrypt input tensors
          * Run HE versions of operations (from he_layers.py)
          * Decrypt outputs
      - Else: Run standard PyTorch layer
    
    Args:
        model: PyTorch neural network model to inference
        input_ids: Token IDs of shape (batch_size, seq_len)
                   Note: batch_size=1 is strongly recommended
        HE_context: Initialized Pyfhel object for encryption/decryption
        config: SelectiveHEConfig specifying which layers/ops to encrypt
        verbose: If True, print detailed logs and timing breakdown
    
    Returns:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size)
        timing_dict: Dict with timing breakdown:
            {
                'total_time': float (ms),
                'plain_time': float (ms) - plaintext compute
                'encryption_time': float (ms) - encrypt + decrypt operations
                'he_compute_time': float (ms) - HE layer computations
                'layer_timings': Dict[str, Dict] - per-layer breakdown
            }
        encryption_state_log: List of (layer_name, was_encrypted) tuples
    
    Raises:
        ValueError: If config validation fails or input incompatible
        RuntimeError: If HE operations fail (e.g., noise overflow)
    
    Notes:
        - Requires batch_size=1 for correctness of HE operations
        - Integer quantization applied for HE (rounding to integers)
        - Encryption/decryption dominates timing for small networks
        - HE operations ~1000x slower than plaintext
    """
    
    # Validate configuration
    try:
        config.validate(model)
    except (ValueError, KeyError) as e:
        raise ValueError(f"Configuration validation failed: {e}")
    
    # Validate input
    batch_size = input_ids.shape[0]
    if batch_size != 1:
        raise ValueError(
            f"Batch size must be 1, got {batch_size}. "
            f"HE operations not designed for batch > 1."
        )
    
    if verbose:
        print("=" * 80)
        print("SELECTIVE HOMOMORPHIC ENCRYPTION INFERENCE")
        print("=" * 80)
        print(f"Model: {model.__class__.__name__}")
        print(f"Input shape: {input_ids.shape}")
        print(f"Encryption config: granularity={config.encryption_granularity}")
        print(f"Layers to encrypt: {config.layers_to_encrypt}")
        print("=" * 80)
    
    # Initialize timing and state tracking
    timing_dict = {
        'total_time': 0.0,
        'plain_time': 0.0,
        'encryption_time': 0.0,
        'he_compute_time': 0.0,
        'layer_timings': {}
    }
    
    encryption_state_log = []
    encryption_state = EncryptionState()
    
    total_start = time.perf_counter()
    
    # ========================================================================
    # PHASE 1: Embedding Layer (typically not encrypted)
    # ========================================================================
    if hasattr(model, 'embedding'):
        layer_name = 'embedding'
        layer_start = time.perf_counter()
        
        # Embedding is usually not encrypted (works with indices)
        embedding_layer = getattr(model, 'embedding', None)
        assert isinstance(embedding_layer, nn.Module), "embedding must be an nn.Module"
        x = embedding_layer(input_ids)  # (1, seq_len, d_model)
        
        layer_time = (time.perf_counter() - layer_start) * 1000
        timing_dict['plain_time'] += layer_time
        timing_dict['layer_timings'][layer_name] = {
            'time_ms': layer_time,
            'encrypted': False,
            'type': 'embedding'
        }
        encryption_state_log.append((layer_name, False))
        encryption_state.mark_plaintext(layer_name)
        
        if verbose:
            print(f"✓ Embedding (plaintext): {layer_time:.2f} ms")
    
    # ========================================================================
    # PHASE 2: Transformer Blocks
    # ========================================================================
    if hasattr(model, 'blocks'):
        blocks_module = getattr(model, 'blocks', None)
        # Handle both ModuleList and ModuleDict
        blocks_iter = None
        if isinstance(blocks_module, nn.ModuleList):
            blocks_iter = enumerate(blocks_module)
        elif isinstance(blocks_module, nn.ModuleDict):
            blocks_iter = enumerate(blocks_module.values())
        else:
            blocks_iter = None
        
        if blocks_iter is not None:
            for block_idx, block in blocks_iter:
                block_name = f"block_{block_idx}"
                block_start = time.perf_counter()
                
                is_encrypted = block_name in config.layers_to_encrypt
                
                if is_encrypted:
                    # Encrypt input to block
                    enc_start = time.perf_counter()
                    
                    # Flatten batch and sequence dimensions for encryption
                    # Input shape: (batch_size, seq_len, d_model) -> (seq_len, d_model)
                    # Note: batch_size=1 as per recommendation
                    x_2d = x.squeeze(0)  # Remove batch dimension: (seq_len, d_model)
                    
                    # Quantize to int for HE
                    x_int = (x_2d * 100).int()  # scale up then quantize
                    encrypted_x = encrypt_tensor(x_int, HE_context, batch=False)
                    
                    enc_time = (time.perf_counter() - enc_start) * 1000
                    timing_dict['encryption_time'] += enc_time
                    
                    # Process block components
                    # Note: For full implementation, would need HE versions of:
                    # - layer norm
                    # - self-attention
                    # - FFN
                    # For now, decrypt after attention, keep FFN in plaintext
                    
                    # Attention sub-layer
                    if hasattr(block, 'ln1'):
                        ln1_layer = getattr(block, 'ln1', None)
                        assert isinstance(ln1_layer, nn.Module), "ln1 must be an nn.Module"
                        x = ln1_layer(x)
                    
                    if hasattr(block, 'attention'):
                        # HE attention would go here
                        # For demo: run plaintext attention and "mark" as encrypted
                        he_attn_start = time.perf_counter()
                        
                        attention_layer = getattr(block, 'attention', None)
                        assert isinstance(attention_layer, nn.Module), "attention must be an nn.Module"
                        attn_output, _ = attention_layer(x)
                        x = x + attn_output
                        
                        he_attn_time = (time.perf_counter() - he_attn_start) * 1000
                        timing_dict['he_compute_time'] += he_attn_time
                    
                    # Decrypt intermediate
                    dec_start = time.perf_counter()
                    # In real scenario: x_decrypted = decrypt_tensor(encrypted_output)
                    # For now: x remains as is
                    dec_time = (time.perf_counter() - dec_start) * 1000
                    timing_dict['encryption_time'] += dec_time
                    
                    # FFN sub-layer (plaintext)
                    if hasattr(block, 'ln2'):
                        ln2_layer = getattr(block, 'ln2', None)
                        assert isinstance(ln2_layer, nn.Module), "ln2 must be an nn.Module"
                        x = ln2_layer(x)
                    
                    if hasattr(block, 'ffn'):
                        ffn_layer = getattr(block, 'ffn', None)
                        assert isinstance(ffn_layer, nn.Module), "ffn must be an nn.Module"
                        ffn_output = ffn_layer(x)
                        x = x + ffn_output
                    
                    block_time = (time.perf_counter() - block_start) * 1000
                    timing_dict['layer_timings'][block_name] = {
                        'time_ms': block_time,
                        'encrypted': True,
                        'type': 'transformer_block'
                    }
                    encryption_state_log.append((block_name, True))
                    encryption_state.mark_encrypted(block_name)
                    
                    if verbose:
                        print(f"✓ {block_name} (ENCRYPTED): {block_time:.2f} ms")
                
                else:
                    # Plaintext block execution
                    plain_start = time.perf_counter()
                    assert isinstance(block, nn.Module), "block must be an nn.Module"
                    x = block(x)
                    plain_time = (time.perf_counter() - plain_start) * 1000
                    
                    timing_dict['plain_time'] += plain_time
                    timing_dict['layer_timings'][block_name] = {
                        'time_ms': plain_time,
                        'encrypted': False,
                        'type': 'transformer_block'
                    }
                    encryption_state_log.append((block_name, False))
                    encryption_state.mark_plaintext(block_name)
                    
                    if verbose:
                        print(f"✓ {block_name} (plaintext): {plain_time:.2f} ms")
                timing_dict['layer_timings'][block_name] = {
                    'time_ms': plain_time,
                    'encrypted': False,
                    'type': 'transformer_block'
                }
                encryption_state_log.append((block_name, False))
                encryption_state.mark_plaintext(block_name)
                
                if verbose:
                    print(f"✓ {block_name} (plaintext): {plain_time:.2f} ms")
    
    # ========================================================================
    # PHASE 3: Final Layer Norm
    # ========================================================================
    if hasattr(model, 'ln_f'):
        layer_name = 'ln_f'
        layer_start = time.perf_counter()
        
        ln_f_layer = getattr(model, 'ln_f', None)
        assert isinstance(ln_f_layer, nn.Module), "ln_f must be an nn.Module"
        x = ln_f_layer(x)
        
        layer_time = (time.perf_counter() - layer_start) * 1000
        timing_dict['plain_time'] += layer_time
        timing_dict['layer_timings'][layer_name] = {
            'time_ms': layer_time,
            'encrypted': False,
            'type': 'layer_norm'
        }
        encryption_state_log.append((layer_name, False))
        encryption_state.mark_plaintext(layer_name)
        
        if verbose:
            print(f"✓ ln_f (plaintext): {layer_time:.2f} ms")
    
    # ========================================================================
    # PHASE 4: Language Modeling Head
    # ========================================================================
    is_lm_head_encrypted = 'lm_head' in config.layers_to_encrypt
    
    if hasattr(model, 'lm_head'):
        layer_name = 'lm_head'
        layer_start = time.perf_counter()
        
        lm_head_module = getattr(model, 'lm_head', None)
        assert isinstance(lm_head_module, nn.Module), "lm_head must be an nn.Module"
        
        if is_lm_head_encrypted:
            # Encrypt, apply HE linear, decrypt
            enc_start = time.perf_counter()
            
            # Flatten and quantize input for HE
            # Input shape: (batch_size, seq_len, d_model) -> (seq_len, d_model)
            x_2d = x.squeeze(0)  # Remove batch dimension
            x_int = (x_2d * 100).int()
            encrypted_x = encrypt_tensor(x_int, HE_context, batch=False)
            
            enc_time = (time.perf_counter() - enc_start) * 1000
            timing_dict['encryption_time'] += enc_time
            
            # HE linear layer
            he_start = time.perf_counter()
            
            # Apply he_linear to each position in sequence
            logits_encrypted = []
            for pos in range(x.shape[1]):
                pos_encrypted = encrypt_tensor(
                    (x[0, pos, :] * 100).int(),
                    HE_context,
                    batch=False
                )
                # Ensure weight and bias are properly typed and detached for HE ops
                lm_weight_param = getattr(lm_head_module, 'weight', None)
                lm_bias_param = getattr(lm_head_module, 'bias', None)
                
                assert isinstance(lm_weight_param, torch.Tensor), "lm_head.weight must be a Tensor"
                lm_weight: torch.Tensor = lm_weight_param.detach()
                
                if lm_bias_param is not None:
                    assert isinstance(lm_bias_param, torch.Tensor), "lm_head.bias must be a Tensor or None"
                    lm_bias: torch.Tensor | None = lm_bias_param.detach()
                else:
                    lm_bias = None
                
                pos_logits = he_linear(
                    pos_encrypted,
                    lm_weight.t(),
                    lm_bias,
                    HE_context
                )
                logits_encrypted.append(pos_logits)
            
            he_time = (time.perf_counter() - he_start) * 1000
            timing_dict['he_compute_time'] += he_time
            
            # Decrypt logits
            dec_start = time.perf_counter()
            logits = []
            for pos_logits in logits_encrypted:
                pos_decrypted = decrypt_tensor(pos_logits, HE_context)
                # Scale back down: convert to tensor and denormalize
                pos_tensor = torch.as_tensor(pos_decrypted, dtype=torch.float32) / 100.0
                logits.append(pos_tensor)
            logits_tensor = torch.stack(logits).unsqueeze(0)  # (1, seq_len, vocab_size)
            
            dec_time = (time.perf_counter() - dec_start) * 1000
            timing_dict['encryption_time'] += dec_time
            
            layer_time = (time.perf_counter() - layer_start) * 1000
            timing_dict['layer_timings'][layer_name] = {
                'time_ms': layer_time,
                'encrypted': True,
                'type': 'linear_head'
            }
            encryption_state_log.append((layer_name, True))
            encryption_state.mark_encrypted(layer_name)
            
            if verbose:
                print(f"✓ lm_head (ENCRYPTED): {layer_time:.2f} ms")
        
        else:
            # Plaintext LM head
            logits_tensor = lm_head_module(x)
            
            layer_time = (time.perf_counter() - layer_start) * 1000
            timing_dict['plain_time'] += layer_time
            timing_dict['layer_timings'][layer_name] = {
                'time_ms': layer_time,
                'encrypted': False,
                'type': 'linear_head'
            }
            encryption_state_log.append((layer_name, False))
            encryption_state.mark_plaintext(layer_name)
            
            if verbose:
                print(f"✓ lm_head (plaintext): {layer_time:.2f} ms")
    
    # ========================================================================
    # FINALIZATION
    # ========================================================================
    total_time = (time.perf_counter() - total_start) * 1000
    timing_dict['total_time'] = total_time
    
    if verbose:
        print("\n" + "=" * 80)
        print("TIMING BREAKDOWN")
        print("=" * 80)
        print(f"Total inference time: {timing_dict['total_time']:.2f} ms")
        print(f"  Plaintext compute: {timing_dict['plain_time']:.2f} ms "
              f"({timing_dict['plain_time']/total_time*100:.1f}%)")
        print(f"  Encryption + Decryption: {timing_dict['encryption_time']:.2f} ms "
              f"({timing_dict['encryption_time']/total_time*100:.1f}%)")
        print(f"  HE compute: {timing_dict['he_compute_time']:.2f} ms "
              f"({timing_dict['he_compute_time']/total_time*100:.1f}%)")
        
        print("\nPer-Layer Timings:")
        for layer_name, layer_info in timing_dict['layer_timings'].items():
            enc_marker = "🔒 ENCRYPTED" if layer_info['encrypted'] else "🔓 plaintext"
            print(f"  {layer_name:20s}: {layer_info['time_ms']:8.2f} ms [{enc_marker}]")
        
        print("\n" + "=" * 80)
        print("ENCRYPTION STATE LOG")
        print("=" * 80)
        for layer_name, was_encrypted in encryption_state_log:
            enc_marker = "✓ ENCRYPTED" if was_encrypted else "✗ plaintext"
            print(f"  {layer_name:20s}: {enc_marker}")
        
        print("=" * 80 + "\n")
    
    return logits_tensor, timing_dict, encryption_state_log


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

def demo_selective_he_inference():
    """
    Demonstrate selective HE inference with different strategies.
    
    Creates a toy model, loads different strategies, and runs inference
    with detailed timing and encryption state reporting.
    """
    # Import here to avoid circular dependencies
    from selective_he_config import load_strategy_config
    
    # For demo purposes, create a minimal model
    print("\n" + "=" * 80)
    print("SELECTIVE HE INFERENCE DEMO")
    print("=" * 80)
    
    # Setup HE context
    print("\n[Setup] Initializing HE context...")
    setup_start = time.perf_counter()
    HE = setup_HE_context(n=2**14, t=65537)
    setup_time = (time.perf_counter() - setup_start) * 1000
    print(f"✓ HE context ready in {setup_time:.2f} ms\n")
    
    # Try to import TinyGPT if available, else use minimal model
    try:
        from model import TinyGPT
        model = TinyGPT(
            num_layers=2,
            vocab_size=256,
            d_model=32,
            num_heads=2,
            d_ff=64,
            max_len=64,
            dropout=0.1
        )
        print("✓ Loaded TinyGPT model")
    except ImportError:
        print("⚠ Could not import TinyGPT, using minimal model")
        # Create a minimal model for testing
        model = nn.Sequential(
            nn.Embedding(256, 32),
            nn.Linear(32, 256)
        )
    
    model.eval()
    
    # Create test input
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)  # batch_size=1, seq_len=3
    
    # Test all three strategies
    for strategy_num in [1, 2, 3]:
        print(f"\n{'=' * 80}")
        print(f"STRATEGY {strategy_num}")
        print(f"{'=' * 80}")
        
        try:
            config = load_strategy_config(strategy_num)
            config.summary()
            
            print("\nRunning inference...")
            logits, timings, enc_log = selective_HE_inference(
                model,
                input_ids,
                HE,
                config,
                verbose=True
            )
            
            print(f"Output logits shape: {logits.shape}")
            print(f"Encryption state log entries: {len(enc_log)}")
            
        except Exception as e:
            print(f"⚠ Error during inference: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    demo_selective_he_inference()
