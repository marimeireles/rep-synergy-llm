"""
Gaussian noise perturbation of attention head weights (Fig 4b).

Injects noise into query-projection rows and output-projection columns
for targeted attention heads. Supports GPT-NeoX, Gemma 3, and Qwen 3.

Unlike ablation (zeroing), noise perturbation allows partial function
while disrupting precise computation.
"""

import logging
from typing import Dict, List, Set, Tuple

import torch

from src.model_registry import ModelSpec

logger = logging.getLogger(__name__)


class GaussianNoisePerturbation:
    """
    Injects Gaussian noise into attention head weight matrices.

    For each targeted head h in layer l:
    - Q-projection: Add N(0, sigma * param_std) to the rows for head h
    - O-projection: Add N(0, sigma * param_std) to the columns for head h

    The noise magnitude is scaled relative to each parameter tensor's own
    standard deviation, so sigma=0.1 means noise_std = 10% of param_std.
    """

    def __init__(self, model, model_spec: ModelSpec, sigma: float = 0.1):
        self.model = model
        self.spec = model_spec
        self.sigma = sigma
        self._original_weights: Dict[str, torch.Tensor] = {}

    def _get_weight_locations(self, layer_idx: int):
        """
        Return (q_proj_param, o_proj_param) weight tensors for the given layer.

        Architecture-specific:
        - GPT-NeoX: fused QKV in query_key_value, O-proj in dense
        - Gemma 3 / Qwen 3: separate q_proj and o_proj
        """
        if self.spec.model_type == "gpt_neox":
            attn = self.model.gpt_neox.layers[layer_idx].attention
            return attn.query_key_value.weight, attn.dense.weight
        elif self.spec.model_type in ("gemma3", "qwen3"):
            attn = self.model.model.layers[layer_idx].self_attn
            return attn.q_proj.weight, attn.o_proj.weight
        else:
            raise ValueError(f"Unknown model type: {self.spec.model_type}")

    def _get_q_row_slice(self, head_in_layer: int) -> slice:
        """
        Get the row slice in the Q-projection weight for the given head.

        For GPT-NeoX (fused QKV): Q rows are [0, num_heads*head_dim),
            head h -> rows [h*head_dim, (h+1)*head_dim)
        For Gemma/Qwen (separate q_proj): same indexing.
        """
        h = head_in_layer
        d = self.spec.head_dim
        return slice(h * d, (h + 1) * d)

    def _get_o_col_slice(self, head_in_layer: int) -> slice:
        """
        Get the column slice in the O-projection weight for the given head.

        O-proj shape: (hidden_size, num_heads * head_dim)
        Head h -> columns [h*head_dim, (h+1)*head_dim)
        """
        h = head_in_layer
        d = self.spec.head_dim
        return slice(h * d, (h + 1) * d)

    def _backup_key(self, layer_idx: int, proj: str) -> str:
        return f"layer{layer_idx}_{proj}"

    def perturb_heads(self, head_indices: List[int], seed: int = 42):
        """
        Inject Gaussian noise into Q-projection and O-projection weights
        for the specified attention heads.

        Args:
            head_indices: list of FLAT head indices
                (converted to (layer, head_in_layer) via divmod)
            seed: random seed for reproducible noise
        """
        self.restore_weights()  # clear any previous perturbation

        # Group heads by layer
        layer_heads: Dict[int, Set[int]] = {}
        for flat_idx in head_indices:
            layer = flat_idx // self.spec.num_heads
            head = flat_idx % self.spec.num_heads
            if layer not in layer_heads:
                layer_heads[layer] = set()
            layer_heads[layer].add(head)

        rng = torch.Generator(device='cpu')
        rng.manual_seed(seed)

        num_perturbed = 0

        for layer_idx, heads in layer_heads.items():
            q_weight, o_weight = self._get_weight_locations(layer_idx)

            # Backup original weights (only once per layer)
            q_key = self._backup_key(layer_idx, "q")
            o_key = self._backup_key(layer_idx, "o")
            if q_key not in self._original_weights:
                self._original_weights[q_key] = q_weight.data.clone()
            if o_key not in self._original_weights:
                self._original_weights[o_key] = o_weight.data.clone()

            # Compute noise scale from parameter std
            q_std = q_weight.data.float().std().item()
            o_std = o_weight.data.float().std().item()

            for head_in_layer in heads:
                # Perturb Q-projection rows for this head
                q_slice = self._get_q_row_slice(head_in_layer)
                q_rows = q_weight.data[q_slice]
                noise_q = torch.randn(q_rows.shape, generator=rng, dtype=torch.float32)
                noise_q = noise_q.to(q_weight.device, dtype=q_weight.dtype)
                q_weight.data[q_slice] += noise_q * (self.sigma * q_std)

                # Perturb O-projection columns for this head
                o_slice = self._get_o_col_slice(head_in_layer)
                o_cols = o_weight.data[:, o_slice]
                noise_o = torch.randn(o_cols.shape, generator=rng, dtype=torch.float32)
                noise_o = noise_o.to(o_weight.device, dtype=o_weight.dtype)
                o_weight.data[:, o_slice] += noise_o * (self.sigma * o_std)

                num_perturbed += 1

        logger.info(
            f"Perturbed {num_perturbed} heads across {len(layer_heads)} layers "
            f"(sigma={self.sigma})"
        )

    def restore_weights(self):
        """Restore all perturbed weights to their original values."""
        if not self._original_weights:
            return

        for key, original in self._original_weights.items():
            # Parse key to find the layer and projection
            parts = key.split("_")
            layer_idx = int(parts[0].replace("layer", ""))
            proj = parts[1]

            q_weight, o_weight = self._get_weight_locations(layer_idx)

            if proj == "q":
                q_weight.data.copy_(original)
            elif proj == "o":
                o_weight.data.copy_(original)

        num_restored = len(self._original_weights)
        self._original_weights.clear()
        logger.info(f"Restored {num_restored} weight tensors to original values.")
