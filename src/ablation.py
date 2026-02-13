"""
Head ablation and behaviour divergence measurement.

Ablates attention heads by zeroing out their outputs and measures
KL divergence between original and ablated output distributions.

Supports GPT-NeoX (Pythia), Gemma 3, and Qwen 3 architectures via ModelSpec.
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.model_registry import ModelSpec, detect_model_spec

logger = logging.getLogger(__name__)


class HeadAblationEngine:
    """
    Ablates attention heads and measures behavior divergence via KL divergence.

    The ablated model is conditioned on the NON-ABLATED model's token sequence.
    We compare output distributions, not generated text.
    """

    def __init__(self, model, tokenizer, device, model_spec: ModelSpec = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._hooks = []

        if model_spec is None:
            model_spec = detect_model_spec(model)
        self.spec = model_spec

        self.num_layers = self.spec.num_layers
        self.num_heads = self.spec.num_heads
        self.head_dim = self.spec.head_dim

    def _register_ablation_hooks(self, heads_to_ablate):
        """
        Register pre-forward hooks on the output projection layer
        that zero out specified attention heads' contributions before projection.

        Args:
            heads_to_ablate: set of (layer_idx, head_within_layer_idx) tuples
        """
        self._remove_hooks()

        # Group by layer for efficiency
        layer_heads = {}
        for (layer, head) in heads_to_ablate:
            if layer not in layer_heads:
                layer_heads[layer] = set()
            layer_heads[layer].add(head)

        for layer_idx in range(self.num_layers):
            if layer_idx not in layer_heads:
                continue

            heads_in_layer = layer_heads[layer_idx]
            hook_module = self.spec.get_hook_module(self.model, layer_idx)

            def make_hook(head_indices, n_heads, h_dim):
                def hook_fn(module, args):
                    # args[0] shape: (batch, seq_len, num_heads * head_dim)
                    # Modify input to zero out specific heads before projection
                    x = args[0].clone()
                    batch, seq_len, _ = x.shape
                    x = x.view(batch, seq_len, n_heads, h_dim)
                    for h_idx in head_indices:
                        x[:, :, h_idx, :] = 0.0
                    x = x.view(batch, seq_len, -1)
                    return (x,) + args[1:]
                return hook_fn

            handle = hook_module.register_forward_pre_hook(
                make_hook(heads_in_layer, self.num_heads, self.head_dim)
            )
            self._hooks.append(handle)

    def _remove_hooks(self):
        """Remove all ablation hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    @torch.no_grad()
    def compute_ablated_logits(self, prompt_tokens, original_token_sequence,
                               heads_to_ablate):
        """
        Feed the original token sequence through the ablated model in a single
        forward pass (no KV cache needed since we have the full sequence).

        Args:
            prompt_tokens: token ids for the prompt (list or tensor)
            original_token_sequence: list of generated token ids from original model
            heads_to_ablate: set of (layer, head) tuples

        Returns:
            ablated_logits: np.ndarray of shape (num_steps, vocab_size)
        """
        self._register_ablation_hooks(heads_to_ablate)

        # Build full sequence: prompt + generated tokens (except the last one,
        # since we want logits for predicting each generated token)
        if isinstance(prompt_tokens, list):
            prompt_ids = prompt_tokens
        else:
            prompt_ids = prompt_tokens.squeeze().tolist()

        # Full input: prompt + all generated tokens except the last
        full_ids = prompt_ids + list(original_token_sequence[:-1])
        input_ids = torch.tensor([full_ids], device=self.device)

        outputs = self.model(input_ids=input_ids, use_cache=False)

        # Extract logits at positions corresponding to generated tokens
        prompt_len = len(prompt_ids)
        num_steps = len(original_token_sequence)
        start_pos = prompt_len - 1
        end_pos = start_pos + num_steps
        ablated_logits = outputs.logits[0, start_pos:end_pos, :].cpu().float().numpy()

        self._remove_hooks()

        return ablated_logits  # (num_steps, vocab_size)

    def compute_behaviour_divergence(self, prompt_tokens, original_token_sequence,
                                     original_logits, heads_to_ablate):
        """
        Compute KL divergence between original and ablated output distributions.

        KL(P_original || P_ablated) averaged over all generation steps.

        Args:
            prompt_tokens: token ids for the prompt
            original_token_sequence: generated tokens from original model
            original_logits: np.ndarray (num_steps, vocab_size) from original model
            heads_to_ablate: set of (layer, head) tuples

        Returns:
            float — mean KL divergence across generation steps
        """
        if len(heads_to_ablate) == 0:
            return 0.0

        ablated_logits = self.compute_ablated_logits(
            prompt_tokens, original_token_sequence, heads_to_ablate
        )

        # Compute KL(P_original || P_ablated) at each step
        # Convert logits to log probabilities
        orig_log_probs = torch.tensor(original_logits, dtype=torch.float32)
        orig_log_probs = F.log_softmax(orig_log_probs, dim=-1)
        orig_probs = torch.exp(orig_log_probs)

        abl_log_probs = torch.tensor(ablated_logits, dtype=torch.float32)
        abl_log_probs = F.log_softmax(abl_log_probs, dim=-1)

        # KL(P || Q) = sum P * (log P - log Q)
        kl_per_step = F.kl_div(abl_log_probs, orig_probs, reduction='none', log_target=False)
        kl_per_step = kl_per_step.sum(dim=-1)  # sum over vocab

        mean_kl = kl_per_step.mean().item()
        return mean_kl

    def run_iterative_ablation(self, prompts, prompt_token_ids,
                               original_tokens_per_prompt,
                               original_logits_per_prompt,
                               head_order, order_name='syn_red',
                               num_layers=None, num_heads_per_layer=None,
                               step_size=1):
        """
        Iteratively remove heads in the given order.

        Args:
            prompts: list of prompt strings
            prompt_token_ids: list of lists — tokenized prompts
            original_tokens_per_prompt: list of lists — generated tokens per prompt
            original_logits_per_prompt: np.ndarray (num_prompts, num_steps, vocab_size)
            head_order: list of flat head indices in ablation order
            order_name: label for this ordering
            num_layers: number of layers
            num_heads_per_layer: heads per layer
            step_size: measure KL every N heads removed (default 1).
                       Use >1 for large models (e.g., Qwen 3 with 1,152 heads).

        Returns:
            list of dicts with keys: num_heads_removed, mean_kl_div, order_type
        """
        if num_layers is None:
            num_layers = self.num_layers
        if num_heads_per_layer is None:
            num_heads_per_layer = self.num_heads

        num_prompts = len(prompts)
        total_heads = len(head_order)

        results = []
        ablated_set = set()

        # Record baseline (0 heads removed)
        results.append({
            'num_heads_removed': 0,
            'mean_kl_div': 0.0,
            'order_type': order_name,
        })

        for k, head_idx in enumerate(tqdm(head_order, desc=f"Ablation ({order_name})")):
            layer = head_idx // num_heads_per_layer
            head_in_layer = head_idx % num_heads_per_layer
            ablated_set.add((layer, head_in_layer))

            # Only measure at step_size intervals or at the last head
            if (k + 1) % step_size != 0 and (k + 1) != total_heads:
                continue

            # Compute divergence for each prompt
            kl_values = []
            for p_idx in range(num_prompts):
                kl = self.compute_behaviour_divergence(
                    prompt_token_ids[p_idx],
                    original_tokens_per_prompt[p_idx],
                    original_logits_per_prompt[p_idx],
                    ablated_set,
                )
                kl_values.append(kl)

            mean_kl = float(np.mean(kl_values))
            results.append({
                'num_heads_removed': k + 1,
                'mean_kl_div': mean_kl,
                'order_type': order_name,
            })

            if (k + 1) % max(16, step_size) == 0:
                logger.info(
                    f"  [{order_name}] Removed {k+1}/{total_heads} heads, "
                    f"mean KL = {mean_kl:.4f}"
                )

        return results
