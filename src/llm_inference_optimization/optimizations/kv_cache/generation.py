import torch

from llm_inference_optimization.inference.utils import create_causal_mask

from .cache import KVCache


def generate_with_kv_cache(model, tokenizer, prompt, max_new_tokens):
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    cache = KVCache(len(model.transformer.decoders))

    model.eval()

    with torch.inference_mode():
        prompt_length = input_ids.shape[1]
        position_ids = torch.arange(prompt_length, device=device).unsqueeze(0)
        mask = create_causal_mask(prompt_length, device)

        logits = model(input_ids, position_ids, cache, mask)

        for _ in range(max_new_tokens):
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            position_ids = torch.full(
                (1, 1),
                input_ids.shape[1] - 1,
                device=device,
            )
            logits = model(next_token, position_ids, cache)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)
