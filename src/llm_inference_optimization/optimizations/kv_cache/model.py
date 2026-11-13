import torch
import torch.nn as nn

from llm_inference_optimization.inference.load_model import (
    load_checkpoint_config,
    load_model_and_tokenizer,
)
from llm_inference_optimization.model.layers.normalization import RMSNormalization

from .cache import KVCache
from .decoder import DecoderBlockWithKVCache


class TransformerBackboneWithKVCache(nn.Module):
    def __init__(self, hidden_size, vocab_size, decoders):
        super().__init__()

        self.decoders = decoders
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.norm = RMSNormalization(hidden_size=hidden_size)

    def forward(self, input_ids, position_ids, cache, mask=None):
        x = self.embedding(input_ids)

        for layer_index, decoder in enumerate(self.decoders):
            x = decoder(x, position_ids, cache, layer_index, mask)

        return self.norm(x)


class CausalLanguageModelWithKVCache(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, input_ids, position_ids, cache, mask=None):
        hidden_states = self.transformer(input_ids, position_ids, cache, mask)
        return hidden_states @ self.transformer.embedding.weight.T


def load_qwen_model_with_kv_cache(checkpoint_dir="checkpoint"):
    checkpoint_dir, config = load_checkpoint_config(checkpoint_dir)

    with torch.device("meta"):
        decoders = nn.ModuleList([
            DecoderBlockWithKVCache(
                hidden_size=config["hidden_size"],
                h_q=config["num_attention_heads"],
                h_kv=config["num_key_value_heads"],
                teta_base=config["rope_theta"],
                d_h=config["head_dim"],
                Dff=config["intermediate_size"],
            )
            for _ in range(config["num_hidden_layers"])
        ])
        model = CausalLanguageModelWithKVCache(
            TransformerBackboneWithKVCache(
                hidden_size=config["hidden_size"],
                vocab_size=config["vocab_size"],
                decoders=decoders,
            )
        )

    return load_model_and_tokenizer(model, checkpoint_dir)
