"""
Selective Homomorphic Encryption Configuration Module

Allows fine-grained control over which parts of a neural network model
to apply homomorphic encryption to, supporting both layer-level and
operation-level granularity.
"""

import json
from typing import List, Dict, Literal, Optional
from pathlib import Path
import torch.nn as nn


class SelectiveHEConfig:
    """
    Configuration class for selective homomorphic encryption (HE) in neural networks.
    
    This class allows specifying which layers and operations within a model should be
    encrypted, with support for two granularity levels: layer-wise or operation-wise.
    """
    
    def __init__(
        self,
        layers_to_encrypt: List[str],
        operations_to_encrypt: Optional[Dict[str, List[str]]] = None,
        encryption_granularity: Literal["layer", "operation"] = "layer"
    ):
        """
        Initialize SelectiveHEConfig.
        
        Args:
            layers_to_encrypt: List of layer names to encrypt (e.g., ["attention_output", "ffn_output"])
            operations_to_encrypt: Dict mapping layer names to lists of operation names to encrypt
                                   (e.g., {"attention_output": ["matmul", "softmax"]})
            encryption_granularity: Level of encryption granularity - "layer" or "operation"
        """
        self.layers_to_encrypt = layers_to_encrypt
        self.operations_to_encrypt = operations_to_encrypt or {}
        self.encryption_granularity = encryption_granularity
    
    def validate(self, model: nn.Module) -> None:
        """
        Validate that the configuration's layer and operation names exist in the model.
        
        Args:
            model: PyTorch model to validate against
            
        Raises:
            ValueError: If any specified layers or operations don't exist in the model
            KeyError: If an operation is specified for a non-existent layer
        """
        # Get all module names from the model
        model_layer_names = {name for name, _ in model.named_modules()}
        
        # Validate layers_to_encrypt
        for layer in self.layers_to_encrypt:
            if layer not in model_layer_names:
                raise ValueError(
                    f"Layer '{layer}' not found in model. "
                    f"Available layers: {sorted(model_layer_names)}"
                )
        
        # Validate operations_to_encrypt
        for layer, ops in self.operations_to_encrypt.items():
            if layer not in self.layers_to_encrypt:
                raise ValueError(
                    f"Layer '{layer}' in operations_to_encrypt not found in layers_to_encrypt"
                )
            
            if layer not in model_layer_names:
                raise ValueError(
                    f"Layer '{layer}' in operations_to_encrypt not found in model"
                )
            
            # Validate that operations are valid (basic check)
            valid_ops = {"matmul", "softmax", "add", "mul", "linear", "conv2d", "relu", 
                        "gelu", "batch_norm", "layer_norm", "attention", "ffn"}
            for op in ops:
                if op not in valid_ops:
                    # Warning rather than error - new ops might be added
                    print(f"Warning: Operation '{op}' may not be recognized")
        
        # Validate granularity
        if self.encryption_granularity not in ["layer", "operation"]:
            raise ValueError(
                f"encryption_granularity must be 'layer' or 'operation', "
                f"got '{self.encryption_granularity}'"
            )
        
        print(f"✓ Configuration validated successfully against model")
    
    def to_json(self, path: str) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            path: File path to save the configuration
        """
        config_dict = {
            "layers_to_encrypt": self.layers_to_encrypt,
            "operations_to_encrypt": self.operations_to_encrypt,
            "encryption_granularity": self.encryption_granularity
        }
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✓ Configuration saved to {path}")
    
    @classmethod
    def from_json(cls, path: str) -> 'SelectiveHEConfig':
        """
        Load configuration from a JSON file.
        
        Args:
            path: File path to load the configuration from
            
        Returns:
            SelectiveHEConfig instance
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            layers_to_encrypt=config_dict.get("layers_to_encrypt", []),
            operations_to_encrypt=config_dict.get("operations_to_encrypt", {}),
            encryption_granularity=config_dict.get("encryption_granularity", "layer")
        )
    
    def summary(self) -> str:
        """
        Generate a human-readable summary of the encryption configuration.
        
        Returns:
            String containing formatted summary of the configuration
        """
        lines = []
        lines.append("=" * 70)
        lines.append("SELECTIVE HOMOMORPHIC ENCRYPTION CONFIGURATION SUMMARY")
        lines.append("=" * 70)
        
        lines.append(f"\nEncryption Granularity: {self.encryption_granularity.upper()}")
        
        lines.append(f"\nLayers to Encrypt ({len(self.layers_to_encrypt)}):")
        if self.layers_to_encrypt:
            for layer in self.layers_to_encrypt:
                lines.append(f"  • {layer}")
        else:
            lines.append("  (none)")
        
        lines.append(f"\nOperation-Level Encryption ({len(self.operations_to_encrypt)}):")
        if self.operations_to_encrypt:
            for layer, ops in self.operations_to_encrypt.items():
                ops_str = ", ".join(ops)
                lines.append(f"  • {layer}: [{ops_str}]")
        else:
            lines.append("  (none)")
        
        # Calculate encryption stats
        total_layers = len(self.layers_to_encrypt)
        total_ops = sum(len(ops) for ops in self.operations_to_encrypt.values())
        
        lines.append("\nEncryption Statistics:")
        lines.append(f"  - Total layers marked for encryption: {total_layers}")
        lines.append(f"  - Total operations marked for encryption: {total_ops}")
        
        if self.encryption_granularity == "operation" and not self.operations_to_encrypt:
            lines.append("\n  ⚠ WARNING: Granularity is 'operation' but no operations specified!")
        
        lines.append("\n" + "=" * 70)
        
        summary_text = "\n".join(lines)
        print(summary_text)
        return summary_text
    
    def __repr__(self) -> str:
        return (
            f"SelectiveHEConfig(layers_to_encrypt={self.layers_to_encrypt}, "
            f"operations_to_encrypt={self.operations_to_encrypt}, "
            f"encryption_granularity='{self.encryption_granularity}')"
        )


# ============================================================================
# EXAMPLE CONFIGURATIONS FOR STRATEGIES 1-3
# ============================================================================

# STRATEGY 1: Encrypt only attention outputs (minimal overhead)
STRATEGY_1_CONFIG = {
    "layers_to_encrypt": ["attention_output"],
    "operations_to_encrypt": {},
    "encryption_granularity": "layer"
}

# STRATEGY 2: Encrypt attention and FFN outputs (balanced)
STRATEGY_2_CONFIG = {
    "layers_to_encrypt": ["attention_output", "ffn_output"],
    "operations_to_encrypt": {},
    "encryption_granularity": "layer"
}

# STRATEGY 3: Encrypt all sensitive operations (maximum security, high overhead)
STRATEGY_3_CONFIG = {
    "layers_to_encrypt": ["attention_output", "ffn_output", "lm_head"],
    "operations_to_encrypt": {
        "attention_output": ["matmul", "softmax"],
        "ffn_output": ["matmul", "gelu"],
        "lm_head": ["matmul"]
    },
    "encryption_granularity": "operation"
}


# ============================================================================
# UTILITY FUNCTIONS FOR WORKING WITH EXAMPLE CONFIGS
# ============================================================================

def create_example_configs(output_dir: str = "he_configs") -> None:
    """
    Create example configuration JSON files for all three strategies.
    
    Args:
        output_dir: Directory to save the example configurations
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    configs = {
        "strategy_1_attention_only.json": STRATEGY_1_CONFIG,
        "strategy_2_attention_ffn.json": STRATEGY_2_CONFIG,
        "strategy_3_full_encryption.json": STRATEGY_3_CONFIG,
    }
    
    for filename, config in configs.items():
        filepath = output_path / filename
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Created {filepath}")


def load_strategy_config(strategy: int) -> SelectiveHEConfig:
    """
    Load a predefined strategy configuration.
    
    Args:
        strategy: Strategy number (1, 2, or 3)
        
    Returns:
        SelectiveHEConfig instance
        
    Raises:
        ValueError: If strategy is not 1, 2, or 3
    """
    strategies = {
        1: STRATEGY_1_CONFIG,
        2: STRATEGY_2_CONFIG,
        3: STRATEGY_3_CONFIG,
    }
    
    if strategy not in strategies:
        raise ValueError(f"Strategy must be 1, 2, or 3, got {strategy}")
    
    config_dict = strategies[strategy]
    return SelectiveHEConfig(
        layers_to_encrypt=config_dict["layers_to_encrypt"],
        operations_to_encrypt=config_dict["operations_to_encrypt"],
        encryption_granularity=config_dict["encryption_granularity"]
    )


if __name__ == "__main__":
    # Example usage
    print("\n--- Creating Example Configurations ---")
    create_example_configs()
    
    print("\n--- Loading and Summarizing Strategy 1 ---")
    config1 = load_strategy_config(1)
    config1.summary()
    
    print("\n--- Loading and Summarizing Strategy 2 ---")
    config2 = load_strategy_config(2)
    config2.summary()
    
    print("\n--- Loading and Summarizing Strategy 3 ---")
    config3 = load_strategy_config(3)
    config3.summary()
