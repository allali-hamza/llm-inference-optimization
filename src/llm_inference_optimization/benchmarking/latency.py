import numpy as np
import torch

from llm_inference_optimization.inference.load_model import load_qwen_model


def prepare_inputs(seq_len, device, vocab_size):
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(1, seq_len),
        device=device,
    )

    position_ids = torch.arange(
        seq_len,
        device=device,
    ).unsqueeze(0)

    mask = torch.full(
        (seq_len, seq_len),
        float("-inf"),
        device=device,
    )

    mask = torch.triu(mask, diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0)

    return input_ids, position_ids, mask

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
    print("Repetitions: 20")

    print(
        f"\n{'Seq Len':<12}"
        f"{'Mean (ms)':<15}"
        f"{'Median (ms)':<15}"
        f"{'P95 (ms)':<15}"
    )
    print("-" * 57)

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

        print(
            f"{seq_len:<12}"
            f"{mean_latency:<15.3f}"
            f"{median_latency:<15.3f}"
            f"{p95_latency:<15.3f}"
        )


if __name__ == "__main__":
    main()