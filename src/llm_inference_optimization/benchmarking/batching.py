from pathlib import Path

import torch

from llm_inference_optimization.optimizations.batching.generation import (
    generate_batched_with_kv_cache,
)
from llm_inference_optimization.benchmarking.utils import save_csv
from llm_inference_optimization.optimizations.kv_cache.model import (
    load_qwen_model_with_kv_cache,
)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    model, tokenizer = load_qwen_model_with_kv_cache()
    device = next(model.parameters()).device
    prompts = [
        "Paris est la capitale de " * 20,
        "Le modèle de langage calcule " * 16,
        "PyTorch permet de mesurer " * 18,
        "Les réseaux de neurones utilisent " * 14,
    ]
    max_new_tokens = 64
    warmup = 5
    repetitions = 10
    results = []

    for batch_size in (1, 2, 4):
        batch_prompts = prompts[:batch_size]

        for _ in range(warmup):
            generate_batched_with_kv_cache(model, tokenizer, batch_prompts, max_new_tokens)

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        all_latencies = []

        for _ in range(repetitions):
            _, latencies = generate_batched_with_kv_cache(
                model, tokenizer, batch_prompts, max_new_tokens, measure=True
            )
            all_latencies.append(latencies)

        mean_latency = sum(sum(values) for values in all_latencies) / (repetitions * max_new_tokens)
        global_throughput = batch_size * 1000 / mean_latency
        results.append({
            "batch_size": batch_size,
            "ttft_ms": sum(values[0] for values in all_latencies) / repetitions,
            "latency_per_request_ms": mean_latency,
            "global_throughput_tokens_s": global_throughput,
            "throughput_per_request_tokens_s": global_throughput / batch_size,
            "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1024 ** 2,
        })

    output_dir = Path("benchmarks/results/batching")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv("batching.csv", results[0].keys(), results, output_dir)

    for result in results:
        print(result)


if __name__ == "__main__":
    main()
