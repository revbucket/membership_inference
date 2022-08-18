""" Utilities for adversarial weight perturbations"""

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import matplotlib.pyplot as plt

# ==========================================
# =           Collect directions           =
# ==========================================

@torch.no_grad()
def normalize(v, ref_v, style='filter'):
    """ Normalize v according to a reference v
    ARGS:
        v: tensor
        ref_v:
        style: "layer" or "filter" -
            if layer: v has the frobenius norm of ref_v
            if filter: each 0-index slice of v has the frobenius norm
                       of corresponding 0-index slice of ref_v
    RETURNS:
        MODIFIED version of v
    """
    eps = 1e-10
    assert style in ['filter', 'layer']

    if style == 'layer':
        v.mul_(ref_v.norm() / (v.norm() + eps))

    else:
        for sub_v, sub_ref in zip(v.data, ref_v.data):
            sub_v.mul_(sub_ref.norm() / (sub_v.norm() + eps))
    return v


@torch.no_grad()
def get_random_direction(model, norm_style='filter'):
    dir_dict = OrderedDict()

    for name, param in model.named_parameters():
        if param.dim() <= 1:
            continue
        v = torch.randn_like(param)
        dir_dict[name] = normalize(v, param, style=norm_style)

    return dir_dict


def get_awp_direction(model, batch, norm_style='filter'):
    dir_dict = OrderedDict()

    optimizer = optim.SGD(model.parameters(), lr=1)
    optimizer.zero_grad()
    F.cross_entropy(model(batch[0]), batch[1]).backward()

    for name, param in model.named_parameters():
        if param.dim() <= 1:
            continue
        v = param.grad.data
        dir_dict[name] = normalize(v, param, style=norm_style)
    return dir_dict



# =====================================================
# =           Evaluate weight perturbations           =
# =====================================================
@torch.no_grad()
def step_weight(model, dir_dict, scale):
    for name, param in model.named_parameters():
        if name not in dir_dict:
            continue
        param.data.add_(dir_dict[name] * scale)
    return model



def evaluate_model(model, dir_dict, gamma, batch,
                   custom_loss=None, num_steps=100):
    clone_model = copy.deepcopy(model)

    if custom_loss is None:
        def custom_loss(m):
            with torch.no_grad():
                return F.cross_entropy(m(batch[0]), batch[1]).item()

    clone_model = step_weight(clone_model, dir_dict, -gamma)
    inc = gamma / float(num_steps)
    alphas = [-1]
    losses = [custom_loss(clone_model)]

    for i in range(2 * num_steps):
        clone_model = step_weight(clone_model, dir_dict, gamma / num_steps)
        alphas.append(alphas[-1] + 1.0 / num_steps)
        losses.append(custom_loss(clone_model))

    return alphas, losses


def plot_awps(awp_list):
    """ awp_list is a list of (alphas, losses, plot_kwargs) tuples """


    fig, ax = plt.subplots()

    if not isinstance(awp_list, list):
        ax.plot(awp_list[0], awp_list[1], **awp_list[2])
        return

    for xs, ys, kwargs in awp_list:
        ax.plot(xs, ys, **kwargs)







