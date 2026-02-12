"""
Hook-based per-head attention activation extraction during autoregressive generation.

For each attention head h at generation step t, we capture:
    a(h, t) = ||softmax(Q_h K_h^T / sqrt(d_k)) V_h||_2

This is the L2 norm of the attention-weighted value vector for that head.
"""

import logging
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class HeadActivationExtractor:
    """
    Extracts per-attention-head activation norms during autoregressive generation.

    Works with Pythia (GPT-NeoX architecture). Hooks capture the per-head
    attention output BEFORE the output projection combines them.
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Pythia-1B: 16 layers, 16 heads per layer
        config = model.config
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_total_heads = self.num_layers * self.num_heads

        # Storage for current generation step's activations
        self._current_step_norms = {}
        self._hooks = []

    def _register_hooks(self):
        """Register forward hooks on all attention layers."""
        self._remove_hooks()

        for layer_idx in range(self.num_layers):
            attn_module = self.model.gpt_neox.layers[layer_idx].attention

            def make_hook(l_idx):
                def hook_fn(module, input, output):
                    # GPTNeoXAttention.forward returns (attn_output, present)
                    # attn_output shape: (batch, seq_len, hidden_size)
                    # But we need per-head outputs. We hook into the attention
                    # computation by intercepting the output and reshaping.
                    #
                    # The attn_output at this point has already been through
                    # the dense (output projection). We need to go deeper.
                    #
                    # Instead, we'll use a hook on the internal attention
                    # computation. For GPTNeoX, the attention weighted values
                    # before the output projection are available if we hook
                    # at the right point.
                    #
                    # Actually, let's use a different approach: hook into the
                    # dense layer and capture its INPUT, which is the per-head
                    # concatenated attention output before projection.
                    pass
                return hook_fn

            # Better approach: hook the dense (output projection) layer's input
            dense_module = attn_module.dense

            def make_dense_hook(l_idx):
                def hook_fn(module, input, output):
                    # input[0] shape: (batch, seq_len, hidden_size)
                    # This is the concatenated per-head outputs before projection
                    attn_concat = input[0]  # (batch, seq_len, hidden_size)
                    batch_size, seq_len, hidden = attn_concat.shape

                    # Reshape to (batch, seq_len, num_heads, head_dim)
                    attn_per_head = attn_concat.view(
                        batch_size, seq_len, self.num_heads, self.head_dim
                    )

                    # Take last position (the newly generated token when using KV cache)
                    # Shape: (batch, num_heads, head_dim)
                    last_pos = attn_per_head[:, -1, :, :]

                    # L2 norm per head: (batch, num_heads)
                    norms = torch.norm(last_pos, dim=-1)  # L2 norm over head_dim

                    # Store norms (take batch dim 0 since batch=1)
                    self._current_step_norms[l_idx] = norms[0].detach().cpu().numpy()

                return hook_fn

            handle = dense_module.register_forward_hook(make_dense_hook(layer_idx))
            self._hooks.append(handle)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    @torch.no_grad()
    def generate_and_extract(
        self, prompt: str, num_tokens: int = 100
    ) -> Tuple[np.ndarray, List[int], np.ndarray]:
        """
        Autoregressively generate num_tokens from prompt.
        At each generation step, record the L2 norm of each head's output.

        Returns:
            activations: np.ndarray of shape (num_total_heads, num_tokens)
            generated_tokens: list of generated token ids
            logits_history: np.ndarray of shape (num_tokens, vocab_size)
        """
        self._register_hooks()

        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_tokens = []
        all_step_norms = []  # list of (num_total_heads,) arrays
        all_logits = []

        # Manual autoregressive generation with KV cache
        past_key_values = None

        for step in range(num_tokens):
            self._current_step_norms = {}

            if step == 0:
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=None,
                    use_cache=True,
                )
            else:
                # Feed only the last generated token
                next_input = torch.tensor(
                    [[generated_tokens[-1]]], device=self.device
                )
                outputs = self.model(
                    input_ids=next_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values

            # Get logits for the last position
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)
            all_logits.append(logits[0].detach().cpu().numpy())

            # Greedy decoding
            next_token = torch.argmax(logits, dim=-1).item()
            generated_tokens.append(next_token)

            # Collect per-head norms from hooks
            step_norms = np.zeros(self.num_total_heads)
            for l_idx in range(self.num_layers):
                if l_idx in self._current_step_norms:
                    head_norms = self._current_step_norms[l_idx]  # (num_heads,)
                    start = l_idx * self.num_heads
                    step_norms[start : start + self.num_heads] = head_norms
            all_step_norms.append(step_norms)

        self._remove_hooks()

        # Shape: (num_total_heads, num_tokens)
        activations = np.stack(all_step_norms, axis=1)
        logits_history = np.stack(all_logits, axis=0)  # (num_tokens, vocab_size)

        return activations, generated_tokens, logits_history

    @torch.no_grad()
    def extract_all_prompts(
        self, prompts: List[str], num_tokens: int = 100
    ) -> Tuple[np.ndarray, list, np.ndarray]:
        """
        Run generate_and_extract for all prompts.

        Returns:
            all_activations: np.ndarray of shape (num_prompts, num_total_heads, num_tokens)
            all_tokens: list of lists of generated token ids
            all_logits: np.ndarray of shape (num_prompts, num_tokens, vocab_size)
        """
        all_activations = []
        all_tokens = []
        all_logits = []

        for i, prompt in enumerate(tqdm(prompts, desc="Extracting activations")):
            logger.info(f"Prompt {i+1}/{len(prompts)}: {prompt[:60]}...")
            activations, tokens, logits = self.generate_and_extract(prompt, num_tokens)
            all_activations.append(activations)
            all_tokens.append(tokens)
            all_logits.append(logits)

        all_activations = np.stack(all_activations, axis=0)
        all_logits = np.stack(all_logits, axis=0)

        logger.info(f"Activations shape: {all_activations.shape}")
        logger.info(f"Logits shape: {all_logits.shape}")

        return all_activations, all_tokens, all_logits
