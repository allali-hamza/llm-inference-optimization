import gc
from pathlib import Path

import torch

from llm_inference_optimization.benchmarking.compare import benchmark
from llm_inference_optimization.benchmarking.utils import save_csv
from llm_inference_optimization.inference.load_model import load_qwen_model
from llm_inference_optimization.optimizations.flash_attention import (
    flash_attention_available,
    load_qwen_model_with_flash_attention,
)
from llm_inference_optimization.optimizations.kv_cache.model import (
    load_qwen_model_with_kv_cache,
)
from llm_inference_optimization.optimizations.quantization import load_quantized_qwen_model


def run_variant(name, loader, use_kv_cache, prompts, warmup, repetitions):
    model, tokenizer = loader()
    device = next(model.parameters()).device
    results = []

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        metrics = benchmark(model, input_ids, 64, warmup, repetitions, use_kv_cache)
        results.append({"variant": name, "prompt_tokens": input_ids.shape[1], **metrics})

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    prompts = [" ".join(["Paris"] * 128), " ".join(["Paris"] * 512)]
    warmup = 5
    repetitions = 10
    results = []

    results += run_variant("baseline_bf16", load_qwen_model, False, prompts, warmup, repetitions)
    results += run_variant("kv_cache_bf16", load_qwen_model_with_kv_cache, True, prompts, warmup, repetitions)

    if flash_attention_available():
        results += run_variant(
            "kv_cache_flash_attention",
            load_qwen_model_with_flash_attention,
            True,
            prompts,
            warmup,
            repetitions,
        )
    else:
        print("FlashAttention backend unavailable: no FlashAttention benchmark executed.")

    results += run_variant("kv_cache_int8", lambda: load_quantized_qwen_model(8), True, prompts, warmup, repetitions)
    results += run_variant("kv_cache_int4", lambda: load_quantized_qwen_model(4), True, prompts, warmup, repetitions)

    output_dir = Path("benchmarks/results/advanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv("advanced.csv", results[0].keys(), results, output_dir)

    for result in results:
        print(result)


if __name__ == "__main__":
    main()
