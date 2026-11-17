import pytest
import torch
import torch.nn as nn

from llm_inference_optimization.optimizations.quantization.model import (
    QuantizedLinear,
    load_quantized_qwen_model,
    quantize_model,
)


@pytest.mark.parametrize("bits", [4, 8])
def test_quantized_linear_runs_on_cpu(bits):
    torch.manual_seed(0)
    linear = nn.Linear(5, 3, bias=False)
    quantized = QuantizedLinear(linear, bits)
    x = torch.randn(2, 5)

    output = quantized(x)

    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()


def test_int8_output_stays_close_to_linear_output():
    torch.manual_seed(0)
    linear = nn.Linear(5, 3, bias=False)
    quantized = QuantizedLinear(linear, bits=8)
    x = torch.randn(2, 5)

    assert torch.allclose(quantized(x), linear(x), rtol=0.05, atol=0.01)


def test_quantize_model_replaces_linear_layers():
    model = nn.Sequential(nn.Linear(3, 4, bias=False), nn.ReLU(), nn.Linear(4, 2, bias=False))

    quantize_model(model, bits=8)

    assert isinstance(model[0], QuantizedLinear)
    assert isinstance(model[2], QuantizedLinear)


def test_quantized_model_rejects_unknown_bit_width():
    with pytest.raises(ValueError, match="bits must be 4 or 8"):
        load_quantized_qwen_model(2)
