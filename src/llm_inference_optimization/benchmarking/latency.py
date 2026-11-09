import numpy as np
import torch

from llm_inference_optimization.inference.load_model import load_qwen_model
from llm_inference_optimization.benchmarking.utils import (
    prepare_inputs,
    save_csv,
    save_metadata,
)

@torch.inference_mode()
def benchmark_forward(model, input_ids, position_ids,mask, warmup=5, repetitions=30):
    # Warmup
    for _ in range(warmup):
        model(input_ids, mask, position_ids)

    torch.cuda.synchronize()

    latencies = []

    # Benchmark
    for _ in range(repetitions):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        model(input_ids, mask, position_ids)

        end_event.record()
        end_event.synchronize()

        latency_ms = start_event.elapsed_time(end_event)
        latencies.append(latency_ms)

    return latencies


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    model, _ = load_qwen_model()
    model.eval()

    device = next(model.parameters()).device
    vocab_size = model.transformer.embedding.num_embeddings

    sequence_lengths = [16, 32, 64, 128, 256]

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Precision: {next(model.parameters()).dtype}")
    print("Batch size: 1")
    print("Warmup: 5")
    print("Repetitions: 30")

    print(
        f"\n{'Seq Len':<12}"
        f"{'Mean (ms)':<15}"
        f"{'Median (ms)':<15}"
        f"{'P95 (ms)':<15}"
    )
    print("-" * 57)

    results = []

    for seq_len in sequence_lengths:
        input_ids, position_ids, mask = prepare_inputs(
            seq_len=seq_len,
            device=device,
            vocab_size=vocab_size,
        )

        latencies = benchmark_forward(
            model=model,
            input_ids=input_ids,
            position_ids=position_ids,
            mask=mask,
        )

        mean_latency = np.mean(latencies)
        median_latency = np.median(latencies)
        p95_latency = np.percentile(latencies, 95)

        results.append({
            "sequence_length": seq_len,
            "mean_latency_ms": mean_latency,
            "median_latency_ms": median_latency,
            "p95_latency_ms": p95_latency,
        })

        print(
            f"{seq_len:<12}"
            f"{mean_latency:<15.3f}"
            f"{median_latency:<15.3f}"
            f"{p95_latency:<15.3f}"
        )

    save_csv(
        "latency.csv",
        ["sequence_length", "mean_latency_ms", "median_latency_ms", "p95_latency_ms"],
        results,
    )
    save_metadata(device, next(model.parameters()).dtype)


if __name__ == "__main__":
    main()
