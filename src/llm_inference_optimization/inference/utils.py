import torch


def create_causal_mask(sequence_length, device):
    mask = torch.full(
        (sequence_length, sequence_length),
        float("-inf"),
        device=device,
    )
    return torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)


def forward_baseline(model, input_ids):
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    mask = create_causal_mask(input_ids.shape[1], input_ids.device)
    return model(input_ids, mask, position_ids)
