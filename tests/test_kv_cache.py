import torch

from llm_inference_optimization.optimizations.kv_cache.cache import KVCache


def test_kv_cache_appends_tokens_for_one_layer():
    cache = KVCache(num_layers=2)
    key = torch.ones(1, 2, 3, 4)
    value = torch.zeros(1, 2, 3, 4)

    cached_key, cached_value = cache.update(0, key, value)
    next_key, next_value = cache.update(0, key * 2, value + 3)

    assert cached_key.shape == (1, 2, 3, 4)
    assert cached_value.shape == (1, 2, 3, 4)
    assert next_key.shape == (1, 2, 6, 4)
    assert next_value.shape == (1, 2, 6, 4)
    assert torch.equal(next_key[:, :, :3], key)
    assert torch.equal(next_key[:, :, 3:], key * 2)
    assert cache.keys[1] is None
