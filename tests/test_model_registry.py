"""
Tests for model_registry architecture detection.

Tests use mock config objects to avoid loading actual models.
"""

import os
import sys
from unittest.mock import MagicMock
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_registry import ModelSpec, detect_model_spec


def _make_mock_model(model_type, num_layers, num_heads, hidden_size,
                     head_dim=None, has_text_config=False, has_gpt_neox=False,
                     vocab_size=50304):
    """Create a mock model with the given config attributes.

    Uses SimpleNamespace for config to avoid MagicMock auto-creating attributes.
    """
    # Build the text-level config
    config_attrs = {
        'model_type': model_type,
        'num_hidden_layers': num_layers,
        'num_attention_heads': num_heads,
        'hidden_size': hidden_size,
        'vocab_size': vocab_size,
    }
    if head_dim is not None:
        config_attrs['head_dim'] = head_dim

    text_config = SimpleNamespace(**config_attrs)

    if has_text_config:
        # Wrap in outer config with text_config nesting
        outer_config = SimpleNamespace(text_config=text_config)
        config = outer_config
    else:
        config = text_config

    model = MagicMock()
    model.config = config

    # Set up module structure
    if has_gpt_neox:
        model.gpt_neox = MagicMock()
        layers = [MagicMock() for _ in range(num_layers)]
        model.gpt_neox.layers = layers
        for layer in layers:
            layer.attention.dense = MagicMock()
        # Remove model.model so hasattr(model, 'model') from MagicMock doesn't confuse
        del model.model
    else:
        # Remove gpt_neox so hasattr returns False
        del model.gpt_neox
        model.model = MagicMock()
        layers = [MagicMock() for _ in range(num_layers)]
        model.model.layers = layers
        for layer in layers:
            layer.self_attn.o_proj = MagicMock()

    return model


class TestGPTNeoXDetection:
    def test_pythia_1b(self):
        model = _make_mock_model(
            model_type='gpt_neox',
            num_layers=16,
            num_heads=8,
            hidden_size=2048,
            has_gpt_neox=True,
        )
        spec = detect_model_spec(model)
        assert spec.model_type == "gpt_neox"
        assert spec.num_layers == 16
        assert spec.num_heads == 8
        assert spec.head_dim == 256  # 2048 // 8
        assert spec.num_total_heads == 128
        assert spec.hidden_size == 2048

    def test_hook_module(self):
        model = _make_mock_model(
            model_type='gpt_neox',
            num_layers=16,
            num_heads=8,
            hidden_size=2048,
            has_gpt_neox=True,
        )
        spec = detect_model_spec(model)
        module = spec.get_hook_module(model, 0)
        assert module is model.gpt_neox.layers[0].attention.dense


class TestGemma3Detection:
    def test_gemma3_4b_head_dim(self):
        """Gemma 3 4B: 8 heads, hidden=2560, but head_dim=256 (NOT 320)."""
        model = _make_mock_model(
            model_type='gemma3',
            num_layers=34,
            num_heads=8,
            hidden_size=2560,
            head_dim=256,
        )
        spec = detect_model_spec(model)
        assert spec.model_type == "gemma3"
        assert spec.num_heads == 8
        assert spec.head_dim == 256  # must be 256, NOT 2560//8=320
        assert spec.num_total_heads == 34 * 8

    def test_gemma3_text_config_nesting(self):
        """Gemma 3 may wrap config inside text_config."""
        model = _make_mock_model(
            model_type='gemma3_text',
            num_layers=34,
            num_heads=8,
            hidden_size=2560,
            head_dim=256,
            has_text_config=True,
        )
        spec = detect_model_spec(model)
        assert spec.model_type == "gemma3"
        assert spec.head_dim == 256
        assert spec.num_layers == 34

    def test_hook_module(self):
        model = _make_mock_model(
            model_type='gemma3',
            num_layers=34,
            num_heads=8,
            hidden_size=2560,
            head_dim=256,
        )
        spec = detect_model_spec(model)
        module = spec.get_hook_module(model, 0)
        assert module is model.model.layers[0].self_attn.o_proj


class TestQwen3Detection:
    def test_qwen3_8b(self):
        model = _make_mock_model(
            model_type='qwen3',
            num_layers=36,
            num_heads=32,
            hidden_size=4096,
            head_dim=128,
        )
        spec = detect_model_spec(model)
        assert spec.model_type == "qwen3"
        assert spec.num_layers == 36
        assert spec.num_heads == 32
        assert spec.head_dim == 128
        assert spec.num_total_heads == 36 * 32

    def test_hook_module(self):
        model = _make_mock_model(
            model_type='qwen3',
            num_layers=36,
            num_heads=32,
            hidden_size=4096,
            head_dim=128,
        )
        spec = detect_model_spec(model)
        module = spec.get_hook_module(model, 5)
        assert module is model.model.layers[5].self_attn.o_proj


class TestModelSpec:
    def test_unknown_model_type_raises(self):
        spec = ModelSpec(
            model_type="unknown_arch",
            num_layers=1, num_heads=1, head_dim=64,
            num_total_heads=1, vocab_size=100, hidden_size=64,
        )
        model = MagicMock()
        with pytest.raises(ValueError, match="Unknown model type"):
            spec.get_hook_module(model, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
