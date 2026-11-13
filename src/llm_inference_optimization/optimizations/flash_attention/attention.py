import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def flash_attention_available():
    return (
        torch.cuda.is_available()
        and torch.backends.cuda.is_flash_attention_available()
        and torch.backends.cuda.flash_sdp_enabled()
    )


def flash_attention_forward(attention, x, position_ids, cache, layer_index, _mask=None):
    query = attention.W_q(x).reshape(x.shape[0], x.shape[1], attention.h_q, attention.d_h)
    key = attention.W_k(x).reshape(x.shape[0], x.shape[1], attention.h_kv, attention.d_h)
    value = attention.W_v(x).reshape(x.shape[0], x.shape[1], attention.h_kv, attention.d_h)

    query = attention.norm_query(query.transpose(1, 2))
    key = attention.norm_key(key.transpose(1, 2))
    value = value.transpose(1, 2)
    query = attention.rot_embedding(query, position_ids)
    key = attention.rot_embedding(key, position_ids)

    first_forward = cache.keys[layer_index] is None
    key, value = cache.update(layer_index, key, value)

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=first_forward,
            enable_gqa=True,
        )

    output = output.transpose(1, 2).reshape(x.shape[0], x.shape[1], attention.hidden_size)
    return attention.W_o(output)
