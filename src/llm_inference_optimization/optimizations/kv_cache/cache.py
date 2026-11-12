import torch


class KVCache:
    def __init__(self, num_layers):
        self.keys = [None] * num_layers
        self.values = [None] * num_layers

    def update(self, layer_index, key, value):
        if self.keys[layer_index] is None:
            self.keys[layer_index] = key
            self.values[layer_index] = value
        else:
            self.keys[layer_index] = torch.cat((self.keys[layer_index], key), dim=2)
            self.values[layer_index] = torch.cat((self.values[layer_index], value), dim=2)

        return self.keys[layer_index], self.values[layer_index]
