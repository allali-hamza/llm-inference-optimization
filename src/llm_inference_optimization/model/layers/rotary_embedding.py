import torch
import torch.nn as nn


class RotaryPositionEncoding(nn.Module):
    def __init__(self, teta_base, hidden_size, d_h):
        super().__init__()
        self.hidden_size = hidden_size
        self.teta_base = teta_base
        self.d_h = d_h

    def forward(self, x, position_ids):
        device = x.device
        num_pairs = self.d_h // 2

        tetas = torch.arange(num_pairs, device=device, dtype=torch.float32)
        tetas = 1 / (self.teta_base ** ((2 * tetas) / self.d_h))

        x_left = x[:, :, :, : self.d_h // 2]
        x_right = x[:, :, :, self.d_h // 2 :]
        angles = tetas * position_ids.to(device=device, dtype=torch.float32).unsqueeze(-1)
        cos = torch.cos(angles).to(dtype=x.dtype).unsqueeze(1)
        sin = torch.sin(angles).to(dtype=x.dtype).unsqueeze(1)

        y_left = x_left * cos - x_right * sin
        y_right = x_left * sin + x_right * cos
        return torch.cat((y_left, y_right), dim=-1)
