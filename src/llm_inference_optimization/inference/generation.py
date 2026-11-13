import torch

from .utils import forward_baseline


def generate(model, tokenizer, prompt, max_new_tokens):
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    model.eval()

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = forward_baseline(model, input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)
