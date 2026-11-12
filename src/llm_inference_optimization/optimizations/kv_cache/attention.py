from math import sqrt

import torch
import torch.nn as nn

from llm_inference_optimization.model.layers.normalization import RMSNormalization
from llm_inference_optimization.model.layers.rotary_embedding import RotaryPositionEncoding


class GroupedQueryAttentionWithKVCache(nn.Module):
    def __init__(self, hidden_size, h_q, h_kv, teta_base, d_h):
        super().__init__()

        self.hidden_size = hidden_size
        self.h_q = h_q
        self.h_kv = h_kv
        self.d_h = d_h

        self.W_q = nn.Linear(hidden_size, d_h * h_q, bias=False)
        self.W_k = nn.Linear(hidden_size, d_h * h_kv, bias=False)
        self.W_v = nn.Linear(hidden_size, d_h * h_kv, bias=False)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=False)

        self.norm_query = RMSNormalization(hidden_size=d_h)
        self.norm_key = RMSNormalization(hidden_size=d_h)
        self.rot_embedding = RotaryPositionEncoding(
            teta_base=teta_base,
            hidden_size=hidden_size,
            d_h=d_h,
        )

    def forward(self, x, position_ids, cache, layer_index, mask=None):
        query = self.W_q(x).reshape(x.shape[0], x.shape[1], self.h_q, self.d_h)
        key = self.W_k(x).reshape(x.shape[0], x.shape[1], self.h_kv, self.d_h)
        value = self.W_v(x).reshape(x.shape[0], x.shape[1], self.h_kv, self.d_h)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        query = self.norm_query(query)
        key = self.norm_key(key)

        query = self.rot_embedding(query, position_ids)
        key = self.rot_embedding(key, position_ids)

        key, value = cache.update(layer_index, key, value)

        group_size = self.h_q // self.h_kv
        query = query.reshape(
            query.shape[0],
            self.h_kv,
            group_size,
            query.shape[2],
            query.shape[3],
        )
        key = key.unsqueeze(2).transpose(-2, -1)
        value = value.unsqueeze(2)

        scores = (query @ key) * (1 / sqrt(self.d_h))

        if mask is not None:
            scores = scores + mask.unsqueeze(2)

        attention_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        output = attention_weights @ value
        output = output.reshape(
            output.shape[0],
            self.h_q,
            output.shape[3],
            output.shape[4],
        )

        output = output.transpose(1, 2).reshape(
            x.shape[0],
            x.shape[1],
            self.hidden_size,
        )

        return self.W_o(output)
