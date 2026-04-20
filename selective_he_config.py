"""
selective_he_config.py
======================
Configuration class for selective homomorphic encryption (HE) of neural network layers.
Supports per-layer and per-operation granularity, JSON serialization, and validation.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Dict, List, Literal

import torch.nn as nn


# ---------------------------------------------------------------------------
# Core configuration class
# ---------------------------------------------------------------------------

class SelectiveHEConfig:
    """
    Expresses which parts of a model should be encrypted with homomorphic
    encryption (HE), at either layer or operation granularity.

    Parameters
    ----------
    layers_to_encrypt : List[str]
        Logical layer names to encrypt (e.g. ``["attention_output", "lm_head"]``).
    operations_to_encrypt : Dict[str, List[str]]
        Mapping from layer name → list of operations within that layer that
        should be encrypted (e.g. ``{"attention_output": ["matmul"]}``).
        Only meaningful when ``encryption_granularity == "operation"``.
    encryption_granularity : {"layer", "operation"}
        * ``"layer"``     – encrypt every operation inside the listed layers.
        * ``"operation"`` – encrypt only the specific operations listed in
          ``operations_to_encrypt`` for each layer.
    """

    # Recognised logical layer tokens (extend as needed)
    KNOWN_LAYERS: frozenset[str] = frozenset(
        {
            "attention_output",
            "ffn_output",
            "lm_head",
            "embedding",
            "layer_norm",
            "query_proj",
            "key_proj",
            "value_proj",
            "output_proj",
            "mlp_fc1",
            "mlp_fc2",
        }
    )

    # Recognised operation tokens per layer type
    KNOWN_OPERATIONS: Dict[str, frozenset[str]] = {
        "attention_output": frozenset({"matmul", "softmax", "scale", "add"}),
        "ffn_output":       frozenset({"matmul", "gelu", "relu", "add"}),
        "lm_head":          frozenset({"matmul", "add"}),
        "embedding":        frozenset({"lookup", "add"}),
        "layer_norm":       frozenset({"norm", "scale", "add"}),
        "query_proj":       frozenset({"matmul", "add"}),
        "key_proj":         frozenset({"matmul", "add"}),
        "value_proj":       frozenset({"matmul", "add"}),
        "output_proj":      frozenset({"matmul", "add"}),
        "mlp_fc1":          frozenset({"matmul", "add", "gelu", "relu"}),
        "mlp_fc2":          frozenset({"matmul", "add"}),
    }

    def __init__(
        self,
        layers_to_encrypt: List[str],
        operations_to_encrypt: Dict[str, List[str]],
        encryption_granularity: Literal["layer", "operation"],
    ) -> None:
        if encryption_granularity not in ("layer", "operation"):
            raise ValueError(
                f"encryption_granularity must be 'layer' or 'operation', "
                f"got {encryption_granularity!r}"
            )
        self.layers_to_encrypt: List[str] = list(layers_to_encrypt)
        self.operations_to_encrypt: Dict[str, List[str]] = {
            k: list(v) for k, v in operations_to_encrypt.items()
        }
        self.encryption_granularity: Literal["layer", "operation"] = encryption_granularity

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, model: nn.Module) -> None:
        """
        Validate the configuration against a concrete ``nn.Module``.

        Checks
        ------
        1. All ``layers_to_encrypt`` names are recognised logical tokens.
        2. All keys in ``operations_to_encrypt`` appear in ``layers_to_encrypt``.
        3. All operation names are recognised for their respective layer.
        4. When granularity is ``"operation"``, every layer in
           ``layers_to_encrypt`` has at least one operation defined.

        Raises
        ------
        ValueError
            On any configuration inconsistency.
        """
        errors: List[str] = []

        # 1 – unknown layer names
        unknown_layers = [
            ln for ln in self.layers_to_encrypt if ln not in self.KNOWN_LAYERS
        ]
        if unknown_layers:
            errors.append(
                f"Unknown layer name(s): {unknown_layers}. "
                f"Known layers: {sorted(self.KNOWN_LAYERS)}"
            )

        # 2 – operations defined for layers not in layers_to_encrypt
        orphan_ops = [
            ln for ln in self.operations_to_encrypt
            if ln not in self.layers_to_encrypt
        ]
        if orphan_ops:
            errors.append(
                f"operations_to_encrypt references layer(s) not in "
                f"layers_to_encrypt: {orphan_ops}"
            )

        # 3 – unknown operation names
        for layer_name, ops in self.operations_to_encrypt.items():
            allowed = self.KNOWN_OPERATIONS.get(layer_name, frozenset())
            bad_ops = [op for op in ops if op not in allowed]
            if bad_ops:
                errors.append(
                    f"Layer '{layer_name}' has unknown operation(s): {bad_ops}. "
                    f"Allowed: {sorted(allowed)}"
                )

        # 4 – operation granularity requires explicit ops for every layer
        if self.encryption_granularity == "operation":
            missing_ops = [
                ln for ln in self.layers_to_encrypt
                if ln not in self.operations_to_encrypt
                or not self.operations_to_encrypt[ln]
            ]
            if missing_ops:
                errors.append(
                    f"encryption_granularity='operation' but no operations "
                    f"defined for layer(s): {missing_ops}"
                )

        if errors:
            bullet_list = "\n  • ".join(errors)
            raise ValueError(
                f"SelectiveHEConfig validation failed:\n  • {bullet_list}"
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "layers_to_encrypt": self.layers_to_encrypt,
            "operations_to_encrypt": self.operations_to_encrypt,
            "encryption_granularity": self.encryption_granularity,
        }

    def to_json(self, path: str | Path) -> None:
        """Serialise configuration to a JSON file at *path*."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, data: dict) -> "SelectiveHEConfig":
        return cls(
            layers_to_encrypt=data["layers_to_encrypt"],
            operations_to_encrypt=data.get("operations_to_encrypt", {}),
            encryption_granularity=data["encryption_granularity"],
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "SelectiveHEConfig":
        """Deserialise a :class:`SelectiveHEConfig` from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable encryption plan."""
        sep = "─" * 60
        lines: List[str] = [
            sep,
            "  Selective Homomorphic Encryption Configuration",
            sep,
            f"  Granularity : {self.encryption_granularity.upper()}",
            f"  Layers      : {len(self.layers_to_encrypt)} selected",
            "",
        ]

        for layer in self.layers_to_encrypt:
            if self.encryption_granularity == "layer":
                lines.append(f"  ▸ {layer}")
                lines.append(f"      → encrypt ALL operations")
            else:
                ops = self.operations_to_encrypt.get(layer, [])
                lines.append(f"  ▸ {layer}")
                if ops:
                    for op in ops:
                        lines.append(f"      → encrypt: {op}")
                else:
                    lines.append(f"      → (no operations specified)")

        lines += ["", sep]
        result = "\n".join(lines)
        print(result)
        return result

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SelectiveHEConfig("
            f"layers={self.layers_to_encrypt}, "
            f"granularity={self.encryption_granularity!r})"
        )


# ---------------------------------------------------------------------------
# Example configurations  (Strategies 1 – 3)
# ---------------------------------------------------------------------------

STRATEGY_1_JSON = """\
{
  "layers_to_encrypt": [
    "lm_head"
  ],
  "operations_to_encrypt": {},
  "encryption_granularity": "layer"
}
"""
"""
Strategy 1 – Minimal / output-only encryption.
Encrypt only the final projection (lm_head) at layer granularity.
Lowest compute overhead; protects only the model's final predictions.
"""

STRATEGY_2_JSON = """\
{
  "layers_to_encrypt": [
    "attention_output",
    "ffn_output",
    "lm_head"
  ],
  "operations_to_encrypt": {
    "attention_output": ["matmul"],
    "ffn_output": ["matmul"],
    "lm_head": ["matmul"]
  },
  "encryption_granularity": "operation"
}
"""
"""
Strategy 2 – Balanced / matmul-only encryption.
Encrypts the most sensitive linear operations (matrix multiplications) in the
three critical layer types.  Skips non-linear ops (softmax, gelu) which are
expensive to evaluate under HE.
"""

STRATEGY_3_JSON = """\
{
  "layers_to_encrypt": [
    "query_proj",
    "key_proj",
    "value_proj",
    "attention_output",
    "mlp_fc1",
    "mlp_fc2",
    "ffn_output",
    "lm_head"
  ],
  "operations_to_encrypt": {},
  "encryption_granularity": "layer"
}
"""
"""
Strategy 3 – Maximum / full-model encryption.
Encrypts all projections, attention outputs, and feed-forward layers at layer
granularity.  Highest privacy guarantee; significant compute cost.
"""


# ---------------------------------------------------------------------------
# Quick smoke-test / demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    import tempfile, os

    configs = [
        ("Strategy 1 – Output-only", STRATEGY_1_JSON),
        ("Strategy 2 – Balanced matmul", STRATEGY_2_JSON),
        ("Strategy 3 – Full model", STRATEGY_3_JSON),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for name, blob in configs:
            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"{'='*60}")

            # Write JSON → load → summarise
            json_path = os.path.join(tmpdir, "cfg.json")
            Path(json_path).write_text(blob, encoding="utf-8")
            cfg = SelectiveHEConfig.from_json(json_path)
            cfg.summary()

            # Round-trip serialisation check
            out_path = os.path.join(tmpdir, "out.json")
            cfg.to_json(out_path)
            reloaded = SelectiveHEConfig.from_json(out_path)
            assert reloaded.to_dict() == cfg.to_dict(), "Round-trip mismatch!"
            print("  ✓ JSON round-trip OK")


if __name__ == "__main__":
    _demo()