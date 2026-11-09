import csv
import json
from pathlib import Path

import torch


RESULTS_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "results" / "baseline"


def save_csv(filename, fieldnames, rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with (RESULTS_DIR / filename).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_metadata(device, dtype):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": "Qwen/Qwen3-1.7B",
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": 1,
        "kv_cache": False,
    }

    with (RESULTS_DIR / "metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2)


def prepare_attention_inputs(seq_len, device):
    position_ids = torch.arange(
        seq_len,
        device=device,
    ).unsqueeze(0)

    mask = torch.full(
        (seq_len, seq_len),
        float("-inf"),
        device=device,
    )

    mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

    return position_ids, mask


def prepare_inputs(seq_len, device, vocab_size):
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(1, seq_len),
        device=device,
    )

    position_ids, mask = prepare_attention_inputs(seq_len, device)

    return input_ids, position_ids, mask


def generate_next_token(model, input_ids):
    position_ids, mask = prepare_attention_inputs(
        input_ids.shape[1],
        input_ids.device,
    )

    logits = model(input_ids, mask, position_ids)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return torch.cat((input_ids, next_token), dim=1)
