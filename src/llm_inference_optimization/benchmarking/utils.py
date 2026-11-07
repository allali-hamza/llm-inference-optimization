import torch

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
