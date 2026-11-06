import torch
import time
from llm_inference_optimization.inference.generation import generate
from llm_inference_optimization.inference.load_model import load_qwen_model


model, tokenizer = load_qwen_model()

def prepare_inputs(seq_len, device):
    input_ids = torch.randint(
        low=0,
        high=151936,
        size=(1, seq_len),
        device=device
    )

    position_ids = torch.arange(
        seq_len,
        device=device
    ).unsqueeze(0)

    mask = torch.full(
        (seq_len, seq_len),
        float("-inf"),
        device=device
    )

    mask = torch.triu(mask, diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0)

    return input_ids, position_ids, mask

