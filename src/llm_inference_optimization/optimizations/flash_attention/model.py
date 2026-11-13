from types import MethodType

from llm_inference_optimization.optimizations.kv_cache.model import (
    load_qwen_model_with_kv_cache,
)

from .attention import flash_attention_forward


def load_qwen_model_with_flash_attention(checkpoint_dir="checkpoint"):
    model, tokenizer = load_qwen_model_with_kv_cache(checkpoint_dir)

    for decoder in model.transformer.decoders:
        decoder.attention.forward = MethodType(flash_attention_forward, decoder.attention)

    return model, tokenizer
