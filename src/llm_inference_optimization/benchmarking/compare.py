import gc
import time

import torch

from llm_inference_optimization.inference.load_model import load_qwen_model
from llm_inference_optimization.inference.utils import create_causal_mask, forward_baseline
from llm_inference_optimization.optimizations.kv_cache.cache import KVCache
from llm_inference_optimization.optimizations.kv_cache.model import (
    load_qwen_model_with_kv_cache,
)


MIB = 1024 ** 2


@torch.inference_mode()
def generate_baseline(model, input_ids, max_new_tokens, measure=False, save_logits=False):
    input_ids = input_ids.clone()
    device = input_ids.device
    latencies = []
    all_logits = []

    for _ in range(max_new_tokens):
        if measure:
            torch.cuda.synchronize(device)
            start = time.perf_counter()

        logits = forward_baseline(model, input_ids)

        if measure:
            torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - start) * 1000)

        if save_logits:
            all_logits.append(logits[:, -1, :].float().cpu())

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat((input_ids, next_token), dim=1)

    return input_ids, all_logits, latencies


@torch.inference_mode()
def generate_with_kv_cache(model, input_ids, max_new_tokens, measure=False, save_logits=False):
    input_ids = input_ids.clone()
    generated_ids = input_ids.clone()
    device = input_ids.device
    cache = KVCache(len(model.transformer.decoders))
    latencies = []
    all_logits = []

    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    mask = create_causal_mask(input_ids.shape[1], device)

    for token_index in range(max_new_tokens):
        if measure:
            torch.cuda.synchronize(device)
            start = time.perf_counter()

        logits = model(input_ids, position_ids, cache, mask)

        if measure:
            torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - start) * 1000)

        if save_logits:
            all_logits.append(logits[:, -1, :].float().cpu())

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat((generated_ids, next_token), dim=1)

        if token_index == 0:
            mask = None

        position_ids = torch.full(
            (1, 1),
            generated_ids.shape[1] - 1,
            device=device,
        )
        input_ids = next_token

    return generated_ids, all_logits, latencies


def benchmark(model, input_ids, max_new_tokens, warmup, repetitions, use_kv_cache):
    generate = generate_with_kv_cache if use_kv_cache else generate_baseline

    for _ in range(warmup):
        generate(model, input_ids, max_new_tokens)

    device = input_ids.device
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    all_latencies = []

    for _ in range(repetitions):
        _, _, latencies = generate(model, input_ids, max_new_tokens, measure=True)
        all_latencies.append(latencies)

    ttft_ms = sum(latencies[0] for latencies in all_latencies) / repetitions
    mean_token_latency_ms = sum(sum(latencies) for latencies in all_latencies) / (
        repetitions * max_new_tokens
    )
    throughput = 1000 / mean_token_latency_ms

    return {
        "ttft_ms": ttft_ms,
        "mean_token_latency_ms": mean_token_latency_ms,
        "throughput_tokens_s": throughput,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / MIB,
    }


def print_results(baseline, kv_cache):
    latency_speedup = baseline["mean_token_latency_ms"] / kv_cache["mean_token_latency_ms"]
    throughput_speedup = kv_cache["throughput_tokens_s"] / baseline["throughput_tokens_s"]
    vram_change = (kv_cache["peak_vram_mib"] / baseline["peak_vram_mib"] - 1) * 100

    print(f"\n{'Metric':<28}{'Baseline':<18}{'KV Cache':<18}{'Gain'}")
    print("-" * 82)
    print(
        f"{'TTFT (ms)':<28}"
        f"{baseline['ttft_ms']:<18.2f}"
        f"{kv_cache['ttft_ms']:<18.2f}"
        f"{baseline['ttft_ms'] / kv_cache['ttft_ms']:<.2f}x"
    )
    print(
        f"{'Mean token latency (ms)':<28}"
        f"{baseline['mean_token_latency_ms']:<18.2f}"
        f"{kv_cache['mean_token_latency_ms']:<18.2f}"
        f"{latency_speedup:<.2f}x"
    )
    print(
        f"{'Throughput (tokens/s)':<28}"
        f"{baseline['throughput_tokens_s']:<18.2f}"
        f"{kv_cache['throughput_tokens_s']:<18.2f}"
        f"{throughput_speedup:<.2f}x"
    )
    print(
        f"{'Peak VRAM (MiB)':<28}"
        f"{baseline['peak_vram_mib']:<18.2f}"
        f"{kv_cache['peak_vram_mib']:<18.2f}"
        f"{vram_change:+.1f}%"
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This comparison requires CUDA.")

    prompt = "Paris est la capitale de"
    correctness_tokens = 8
    max_new_tokens = 64
    warmup = 2
    repetitions = 3

    baseline_model, tokenizer = load_qwen_model()
    device = next(baseline_model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    baseline_output, baseline_logits, _ = generate_baseline(
        baseline_model,
        input_ids,
        correctness_tokens,
        save_logits=True,
    )
    baseline_dtype = next(baseline_model.parameters()).dtype
    baseline_parameters = sum(parameter.numel() for parameter in baseline_model.parameters())

    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()

    kv_model, _ = load_qwen_model_with_kv_cache()
    kv_output, kv_logits, _ = generate_with_kv_cache(
        kv_model,
        input_ids,
        correctness_tokens,
        save_logits=True,
    )
    kv_dtype = next(kv_model.parameters()).dtype
    kv_parameters = sum(parameter.numel() for parameter in kv_model.parameters())

    same_tokens = torch.equal(baseline_output, kv_output)
    logits_close = all(
        torch.allclose(baseline_logit, kv_logit, rtol=0.05, atol=0.5)
        for baseline_logit, kv_logit in zip(baseline_logits, kv_logits)
    )
    max_logit_difference = max(
        (baseline_logit - kv_logit).abs().max().item()
        for baseline_logit, kv_logit in zip(baseline_logits, kv_logits)
    )

    print(f"Same generated tokens: {same_tokens}")
    print(f"Logits close: {logits_close} (max difference: {max_logit_difference:.4f})")
    print(f"Same dtype: {baseline_dtype == kv_dtype} ({baseline_dtype})")
    print(f"Same number of parameters: {baseline_parameters == kv_parameters}")

    if not same_tokens or not logits_close or baseline_dtype != kv_dtype:
        raise RuntimeError("KV Cache validation failed.")

    kv_results = benchmark(
        kv_model,
        input_ids,
        max_new_tokens,
        warmup,
        repetitions,
        use_kv_cache=True,
    )

    del kv_model
    gc.collect()
    torch.cuda.empty_cache()

    baseline_model, _ = load_qwen_model()
    baseline_results = benchmark(
        baseline_model,
        input_ids,
        max_new_tokens,
        warmup,
        repetitions,
        use_kv_cache=False,
    )

    print_results(baseline_results, kv_results)


if __name__ == "__main__":
    main()
