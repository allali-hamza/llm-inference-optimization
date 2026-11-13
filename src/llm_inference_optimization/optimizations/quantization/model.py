import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_inference_optimization.optimizations.kv_cache.model import (
    load_qwen_model_with_kv_cache,
)


class QuantizedLinear(nn.Module):
    def __init__(self, linear, bits):
        super().__init__()
        weight = linear.weight.detach()
        max_value = 127 if bits == 8 else 7
        scale = weight.abs().amax(dim=1, keepdim=True) / max_value
        scale = scale.clamp_min(torch.finfo(weight.dtype).eps)
        values = torch.round(weight / scale).clamp(-max_value, max_value).to(torch.int8)

        self.bits = bits
        self.in_features = linear.in_features
        self.register_buffer("scale", scale)

        if bits == 8:
            self.register_buffer("weight", values)
        else:
            values = (values + 8).to(torch.uint8)
            if values.shape[1] % 2:
                values = F.pad(values, (0, 1))
            self.register_buffer("weight", values[:, 0::2] | (values[:, 1::2] << 4))

    def forward(self, x):
        if self.bits == 8:
            weight = self.weight.to(x.dtype)
        else:
            low = self.weight & 15
            high = self.weight >> 4
            weight = torch.stack((low, high), dim=-1).reshape(self.weight.shape[0], -1)
            weight = weight[:, :self.in_features].to(x.dtype) - 8

        return F.linear(x, weight * self.scale.to(x.dtype))


def quantize_model(model, bits):
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, QuantizedLinear(child, bits))

    return model


def load_quantized_qwen_model(bits, checkpoint_dir="checkpoint"):
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")

    model, tokenizer = load_qwen_model_with_kv_cache(checkpoint_dir)
    return quantize_model(model, bits), tokenizer
