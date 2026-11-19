"""Step 2: the weight edit.

Adds a constant vector v = alpha * d_L to the output of layer L's MLP (`mlp.down_proj`), i.e. a new bias
parameter on that Linear (896 numbers = 0.00018% of the model's ~494M parameters). Because the bias is
added at every position, it shifts the residual stream by v everywhere downstream - a permanent, input-
independent parameter change (no hooks at inference time). The delta is saved as a tiny file.
"""
import torch
from torch import nn


def apply_edit(model, layer, vec):
    old = model.model.layers[layer].mlp.down_proj
    new = nn.Linear(old.in_features, old.out_features, bias=True, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.copy_(old.weight)
        new.bias.copy_(vec.to(old.weight.dtype))
    model.model.layers[layer].mlp.down_proj = new
    return old  # keep to revert


def revert_edit(model, layer, old):
    model.model.layers[layer].mlp.down_proj = old


def random_like(vec, seed):
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(vec.shape, generator=g)
    return r / r.norm() * vec.norm()


def save_edit(path, layer, vec):
    torch.save({"layer": layer, "down_proj_bias": vec.clone()}, path)


def load_edit(model, path):
    e = torch.load(path)
    return apply_edit(model, e["layer"], e["down_proj_bias"])
