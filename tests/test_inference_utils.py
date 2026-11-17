import torch

from llm_inference_optimization.inference.utils import (
    create_causal_mask,
    forward_baseline,
)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.mask = None
        self.position_ids = None

    def forward(self, input_ids, mask, position_ids):
        self.mask = mask
        self.position_ids = position_ids
        return input_ids.float().unsqueeze(-1)


def test_create_causal_mask():
    mask = create_causal_mask(3, torch.device("cpu"))

    assert mask.shape == (1, 1, 3, 3)
    assert torch.equal(mask[0, 0].isneginf(), torch.tensor([
        [False, True, True],
        [False, False, True],
        [False, False, False],
    ]))


def test_forward_baseline_creates_inputs_for_model():
    model = FakeModel()
    input_ids = torch.tensor([[4, 7, 2]])

    logits = forward_baseline(model, input_ids)

    assert logits.shape == (1, 3, 1)
    assert torch.equal(model.position_ids, torch.tensor([[0, 1, 2]]))
    assert model.mask.shape == (1, 1, 3, 3)
