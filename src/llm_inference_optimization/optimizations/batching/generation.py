import time

import torch

from llm_inference_optimization.inference.utils import create_causal_mask
from llm_inference_optimization.optimizations.kv_cache.cache import KVCache


def generate_batched_with_kv_cache(model, tokenizer, prompts, max_new_tokens, measure=False):
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    tokens = tokenizer(prompts, padding=True, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)
    key_mask = tokens["attention_mask"].to(device).bool()
    generated_ids = input_ids.clone()
    cache = KVCache(len(model.transformer.decoders))
    latencies = []

    position_ids = key_mask.long().cumsum(dim=1).sub(1).clamp_min(0)
    sequence_length = input_ids.shape[1]
    mask = create_causal_mask(sequence_length, device)
    mask = mask.masked_fill(~key_mask[:, None, None, :], float("-inf"))

    with torch.inference_mode():
        for token_index in range(max_new_tokens):
            if measure:
                torch.cuda.synchronize(device)
                start = time.perf_counter()

            logits = model(input_ids, position_ids, cache, mask)

            if measure:
                torch.cuda.synchronize(device)
                latencies.append((time.perf_counter() - start) * 1000)

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat((generated_ids, next_token), dim=1)
            key_mask = torch.cat((key_mask, torch.ones_like(next_token, dtype=torch.bool)), dim=1)

            input_ids = next_token
            position_ids = key_mask.long().sum(dim=1, keepdim=True) - 1
            mask = torch.zeros(
                generated_ids.shape[0], 1, 1, key_mask.shape[1], device=device
            )
            mask = mask.masked_fill(~key_mask[:, None, None, :], float("-inf"))

    texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return texts, latencies
