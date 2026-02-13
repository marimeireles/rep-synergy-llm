"""
Architecture detection and hook point abstraction for multi-model support.

Supports:
- GPT-NeoX (Pythia): model.gpt_neox.layers[i].attention.dense
- Gemma 3:           model.model.layers[i].self_attn.o_proj
- Qwen 3:            model.model.layers[i].self_attn.o_proj
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    """Architecture specification for a model."""
    model_type: str        # "gpt_neox", "gemma3", "qwen3"
    num_layers: int
    num_heads: int         # attention heads per layer
    head_dim: int
    num_total_heads: int   # num_layers * num_heads
    vocab_size: int
    hidden_size: int

    def get_hook_module(self, model, layer_idx):
        """
        Return the module to hook for capturing per-head attention outputs.

        We hook the output projection (dense/o_proj) and capture its INPUT,
        which is the concatenated per-head attention outputs before projection.
        """
        if self.model_type == "gpt_neox":
            return model.gpt_neox.layers[layer_idx].attention.dense
        elif self.model_type in ("gemma3", "qwen3"):
            return model.model.layers[layer_idx].self_attn.o_proj
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


def detect_model_spec(model) -> ModelSpec:
    """
    Auto-detect architecture from a loaded HuggingFace model.

    Returns a ModelSpec with correct layer count, head count, head_dim,
    and hook module accessor.
    """
    config = model.config

    # Handle text_config nesting (Gemma 3 wraps text config)
    text_config = getattr(config, 'text_config', config)

    model_type_str = getattr(text_config, 'model_type', '')
    num_layers = text_config.num_hidden_layers
    hidden_size = text_config.hidden_size
    vocab_size = text_config.vocab_size

    if model_type_str == 'gpt_neox':
        num_heads = text_config.num_attention_heads
        head_dim = hidden_size // num_heads
        model_type = "gpt_neox"

    elif model_type_str in ('gemma3', 'gemma3_text', 'gemma2'):
        num_heads = text_config.num_attention_heads
        # Gemma 3 uses explicit head_dim that may differ from hidden_size // num_heads
        # (e.g., 4B: 8 heads, hidden=2560, but head_dim=256, not 320)
        head_dim = getattr(text_config, 'head_dim', hidden_size // num_heads)
        model_type = "gemma3"

    elif model_type_str in ('qwen3', 'qwen2'):
        num_heads = text_config.num_attention_heads
        head_dim = getattr(text_config, 'head_dim', hidden_size // num_heads)
        model_type = "qwen3"

    else:
        # Fallback: try to detect from module structure
        if hasattr(model, 'gpt_neox'):
            num_heads = text_config.num_attention_heads
            head_dim = hidden_size // num_heads
            model_type = "gpt_neox"
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            num_heads = text_config.num_attention_heads
            head_dim = getattr(text_config, 'head_dim', hidden_size // num_heads)
            # Check if it has o_proj (LLaMA-style)
            first_attn = model.model.layers[0].self_attn
            if hasattr(first_attn, 'o_proj'):
                # Could be Gemma, Qwen, LLaMA — use generic name
                model_type = "qwen3"  # same hook pattern
            else:
                raise ValueError(
                    f"Cannot detect model type. model_type='{model_type_str}', "
                    f"no o_proj found in self_attn."
                )
        else:
            raise ValueError(
                f"Cannot detect model type from config.model_type='{model_type_str}' "
                f"or module structure."
            )

    num_total_heads = num_layers * num_heads

    spec = ModelSpec(
        model_type=model_type,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        num_total_heads=num_total_heads,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
    )

    logger.info(
        f"Detected {model_type}: {num_layers} layers, {num_heads} heads/layer, "
        f"head_dim={head_dim}, total_heads={num_total_heads}, "
        f"hidden={hidden_size}, vocab={vocab_size}"
    )

    return spec
