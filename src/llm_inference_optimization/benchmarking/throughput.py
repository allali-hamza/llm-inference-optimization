import time

import torch

from llm_inference_optimization.benchmarking.utils import prepare_inputs
from llm_inference_optimization.inference.load_model import load_qwen_model


@torch.inference_mode()
def generate_and_measure(model, input_ids, max_new_tokens):
    device = input_ids.device
    model.eval()

    latencies = []

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    total_start = time.perf_counter()

    for _ in range(max_new_tokens):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start = time.perf_counter()

        sequence_length = input_ids.shape[1]

        position_ids = torch.arange(
            sequence_length,
            device=device,
        ).unsqueeze(0)

        mask = torch.full(
            (sequence_length, sequence_length),
            float("-inf"),
            device=device,
        )

        mask = torch.triu(
            mask,
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

        logits = model(input_ids, mask, position_ids)

        next_token = logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

        input_ids = torch.cat(
            (input_ids, next_token),
            dim=1,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

    total_end = time.perf_counter()
    total_time = total_end - total_start

    return latencies, total_time


def compute_metrics(latencies, total_time):

    num_tokens = len(latencies)

    ttft = latencies[0]

    if num_tokens > 1:
        tpot = sum(latencies[1:]) / (num_tokens - 1)
    else:
        tpot = 0.0

    throughput = num_tokens / total_time

    return {
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "total_time_s": total_time,
        "throughput_tokens_s": throughput,
    }



def main():
    model, _ = load_qwen_model()
    model.eval()

    device = next(model.parameters()).device
    vocab_size = model.transformer.embedding.num_embeddings

    prompt_len = 32
    max_new_tokens_list = [16, 32, 64]
    warmup = 2

    input_ids, _, _ = prepare_inputs(
        seq_len=prompt_len,
        device=device,
        vocab_size=vocab_size,
    )

    print(f"Device: {device}")
    print(f"Precision: {next(model.parameters()).dtype}")
    print(f"Prompt length: {prompt_len}")
    print(f"Batch size: {input_ids.shape[0]}")
    print(f"Warmup: {warmup}")

    print(
        f"\n{'New Tokens':<14}"
        f"{'TTFT (ms)':<14}"
        f"{'TPOT (ms)':<14}"
        f"{'Total (s)':<14}"
        f"{'Throughput (tok/s)':<22}"
    )
    print("-" * 78)

    for max_new_tokens in max_new_tokens_list:

        for _ in range(warmup):
            generate_and_measure(
                model=model,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
            )

        latencies, total_time = generate_and_measure(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )

        metrics = compute_metrics(latencies, total_time)

        print(
            f"{max_new_tokens:<14}"
            f"{metrics['ttft_ms']:<14.3f}"
            f"{metrics['tpot_ms']:<14.3f}"
            f"{metrics['total_time_s']:<14.3f}"
            f"{metrics['throughput_tokens_s']:<22.2f}"
        )


if __name__ == "__main__":
    main()