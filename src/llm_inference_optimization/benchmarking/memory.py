import torch

from llm_inference_optimization.inference.load_model import load_qwen_model
from llm_inference_optimization.benchmarking.utils import (
    generate_next_token,
    prepare_inputs,
    save_csv,
    save_metadata,
)


MIB = 1024 ** 2


def get_memory_metrics(device, baseline_allocated, baseline_reserved):
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    return {
        "baseline_mib": baseline_allocated / MIB,
        "peak_mib": peak_allocated / MIB,
        "additional_mib": (peak_allocated - baseline_allocated) / MIB,
        "reserved_baseline_mib": baseline_reserved / MIB,
        "reserved_peak_mib": peak_reserved / MIB,
    }


@torch.inference_mode()
def benchmark_forward_memory(
    model,
    input_ids,
    position_ids,
    mask,
    warmup=3,
):
    device = input_ids.device

    # Warmup
    for _ in range(warmup):
        model(input_ids, mask, position_ids)

    torch.cuda.synchronize(device)

    # Baseline memory
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)

    # Reset peak statistics
    torch.cuda.reset_peak_memory_stats(device)

    # Forward pass
    logits = model(input_ids, mask, position_ids)

    torch.cuda.synchronize(device)

    # Memory measurements
    metrics = get_memory_metrics(
        device,
        baseline_allocated,
        baseline_reserved,
    )

    del logits

    return metrics


@torch.inference_mode()
def generate_tokens(model, input_ids, max_new_tokens):
    for _ in range(max_new_tokens):
        input_ids = generate_next_token(model, input_ids)

    return input_ids


@torch.inference_mode()
def benchmark_generation_memory(
    model,
    input_ids,
    max_new_tokens,
    warmup=2,
):
    device = input_ids.device

    # Warmup: complete generations
    for _ in range(warmup):
        warmup_output = generate_tokens(
            model,
            input_ids,
            max_new_tokens,
        )
        del warmup_output

    torch.cuda.synchronize(device)

    # Baseline memory
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)

    # Reset peak statistics
    torch.cuda.reset_peak_memory_stats(device)

    # Generate tokens
    output_ids = generate_tokens(
        model,
        input_ids,
        max_new_tokens,
    )

    torch.cuda.synchronize(device)

    # Memory measurements
    metrics = get_memory_metrics(
        device,
        baseline_allocated,
        baseline_reserved,
    )

    del output_ids

    return metrics


def print_results(title, results, first_column):
    print(f"\n{title}")

    print(
        f"{first_column:<12}"
        f"{'Baseline (MiB)':<18}"
        f"{'Peak (MiB)':<16}"
        f"{'Additional (MiB)':<20}"
        f"{'Reserved Peak (MiB)':<22}"
    )

    print("-" * 88)

    for value, metrics in results:
        print(
            f"{value:<12}"
            f"{metrics['baseline_mib']:<18.2f}"
            f"{metrics['peak_mib']:<16.2f}"
            f"{metrics['additional_mib']:<20.2f}"
            f"{metrics['reserved_peak_mib']:<22.2f}"
        )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Memory benchmark requires CUDA.")

    # Load model once
    model, _ = load_qwen_model()
    model.eval()

    device = next(model.parameters()).device
    vocab_size = model.transformer.embedding.num_embeddings

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Precision: {next(model.parameters()).dtype}")
    print("Batch size: 1")
    print("KV Cache: Disabled")

    # ==========================================
    # Benchmark 1: Forward memory
    # ==========================================

    sequence_lengths = [16, 32, 64, 128, 256]
    forward_results = []

    for seq_len in sequence_lengths:
        input_ids, position_ids, mask = prepare_inputs(
            seq_len=seq_len,
            device=device,
            vocab_size=vocab_size,
        )

        metrics = benchmark_forward_memory(
            model=model,
            input_ids=input_ids,
            position_ids=position_ids,
            mask=mask,
            warmup=3,
        )

        forward_results.append((seq_len, metrics))

        del input_ids, position_ids, mask

    print_results(
        title="FORWARD MEMORY BENCHMARK",
        results=forward_results,
        first_column="Seq Len",
    )

    # ==========================================
    # Benchmark 2: Generation memory
    # ==========================================

    prompt_len = 32
    max_new_tokens_list = [16, 32, 64]
    generation_results = []

    input_ids, _, _ = prepare_inputs(
        seq_len=prompt_len,
        device=device,
        vocab_size=vocab_size,
    )

    for max_new_tokens in max_new_tokens_list:
        metrics = benchmark_generation_memory(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            warmup=2,
        )

        generation_results.append((max_new_tokens, metrics))

    print_results(
        title="GENERATION MEMORY BENCHMARK",
        results=generation_results,
        first_column="New Tokens",
    )

    save_csv(
        "forward_memory.csv",
        ["sequence_length", "baseline_mib", "peak_mib", "additional_mib", "reserved_baseline_mib", "reserved_peak_mib"],
        [
            {"sequence_length": seq_len, **metrics}
            for seq_len, metrics in forward_results
        ],
    )
    save_csv(
        "generation_memory.csv",
        ["max_new_tokens", "baseline_mib", "peak_mib", "additional_mib", "reserved_baseline_mib", "reserved_peak_mib"],
        [
            {"max_new_tokens": max_new_tokens, **metrics}
            for max_new_tokens, metrics in generation_results
        ],
    )
    save_metadata(device, next(model.parameters()).dtype)

    del input_ids

    print("\nMemory benchmarking completed.")


if __name__ == "__main__":
    main()
