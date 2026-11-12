import torch.nn as nn

from llm_inference_optimization.model.layers.feed_forward import GatedFeedForward
from llm_inference_optimization.model.layers.normalization import RMSNormalization

from .attention import GroupedQueryAttentionWithKVCache


class DecoderBlockWithKVCache(nn.Module):
    def __init__(self, hidden_size, h_q, h_kv, teta_base, d_h, Dff):
        super().__init__()

        self.first_norm = RMSNormalization(hidden_size=hidden_size)
        self.second_norm = RMSNormalization(hidden_size=hidden_size)
        self.attention = GroupedQueryAttentionWithKVCache(
            hidden_size=hidden_size,
            h_q=h_q,
            h_kv=h_kv,
            teta_base=teta_base,
            d_h=d_h,
        )
        self.feed_forward = GatedFeedForward(hidden_size=hidden_size, Dff=Dff)

    def forward(self, x, position_ids, cache, layer_index, mask=None):
        residual = x
        x = self.first_norm(x)
        x = self.attention(x, position_ids, cache, layer_index, mask)
        x = x + residual

        residual = x
        x = self.second_norm(x)
        x = self.feed_forward(x)

        return x + residual
