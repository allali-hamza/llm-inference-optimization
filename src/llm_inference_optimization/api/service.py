import gc
import time

import torch

from llm_inference_optimization.inference.load_model import load_qwen_model
from llm_inference_optimization.inference.utils import create_causal_mask, forward_baseline
from llm_inference_optimization.optimizations.flash_attention.model import (
    load_qwen_model_with_flash_attention,
)
from llm_inference_optimization.optimizations.kv_cache.model import load_qwen_model_with_kv_cache
from llm_inference_optimization.optimizations.kv_cache.cache import KVCache
from llm_inference_optimization.optimizations.quantization.model import (
    load_quantized_qwen_model,
)



def load_variant(variant):
    if variant == "baseline":
        return load_qwen_model()

    if variant == "kv_cache":
        return load_qwen_model_with_kv_cache()

    if variant == "kv_cache_flash_attention":
        return load_qwen_model_with_flash_attention()

    if variant == "kv_cache_int8":
        return load_quantized_qwen_model(8)

    if variant == "kv_cache_int4":
        return load_quantized_qwen_model(4)

    raise ValueError("Unknown variant")


@torch.inference_mode()
def generate_and_measure(model, tokenizer, prompt, max_new_tokens, variant):
    device = next(model.parameters()).device
    messages = [{"role": "user", "content": prompt}]
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    input_ids = tokens["input_ids"].to(device)
    prompt_length = input_ids.shape[1]
    model.eval()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    latencies = []

    if variant == "baseline":
        for _ in range(max_new_tokens):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            logits = forward_baseline(model, input_ids)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - start) * 1000)

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break
    else:
        cache = KVCache(len(model.transformer.decoders))
        generated_ids = input_ids.clone()
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        mask = create_causal_mask(input_ids.shape[1], device)

        for token_index in range(max_new_tokens):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            logits = model(input_ids, position_ids, cache, mask)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - start) * 1000)

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat((generated_ids, next_token), dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            input_ids = next_token
            position_ids = torch.full(
                (1, 1),
                generated_ids.shape[1] - 1,
                device=device,
            )
            mask = None

        input_ids = generated_ids

    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    metrics = {
        "ttft_ms": latencies[0] if latencies else 0.0,
        "mean_decode_step_latency_ms": mean_latency,
        "decode_throughput_tokens_s": 1000 / mean_latency if mean_latency else 0.0,
        "peak_allocated_vram_mib": (
            torch.cuda.max_memory_allocated(device) / 1024 ** 2
            if device.type == "cuda"
            else 0.0
        ),
    }

    return tokenizer.decode(input_ids[0, prompt_length:], skip_special_tokens=True), metrics


def generate_response(request):
    model, tokenizer = load_variant(request.variant)
    generate_and_measure(model, tokenizer, request.prompt, 1, request.variant)

    text, metrics = generate_and_measure(
        model,
        tokenizer,
        request.prompt,
        request.max_new_tokens,
        request.variant,
    )

    response = {
        "variant": request.variant,
        "text": text,
        "metrics": metrics,
        "baseline_metrics": None,
        "difference_vs_baseline": None,
    }

    if not request.compare_to_baseline or request.variant == "baseline":
        return response

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    baseline_model, baseline_tokenizer = load_variant("baseline")
    generate_and_measure(
        baseline_model,
        baseline_tokenizer,
        request.prompt,
        1,
        "baseline",
    )
    _, baseline_metrics = generate_and_measure(
        baseline_model,
        baseline_tokenizer,
        request.prompt,
        request.max_new_tokens,
        "baseline",
    )

    response["baseline_metrics"] = baseline_metrics
    response["difference_vs_baseline"] = {
        "decode_latency_speedup": (
            baseline_metrics["mean_decode_step_latency_ms"]
            / metrics["mean_decode_step_latency_ms"]
        ),
        "throughput_speedup": (
            metrics["decode_throughput_tokens_s"]
            / baseline_metrics["decode_throughput_tokens_s"]
        ),
        "vram_difference_mib": (
            metrics["peak_allocated_vram_mib"]
            - baseline_metrics["peak_allocated_vram_mib"]
        ),
    }

    del baseline_model, baseline_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response


