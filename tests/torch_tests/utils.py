import torch as pt

def generate_blob(rg=False):
    xi = pt.rand(10, 2)
    xi_labels = pt.tensor([-1., 1.])[(pt.rand(10) > .5).long()]
    if rg:
        xi.requires_grad_(True)
        xi_labels.requires_grad_(True)
    return xi, xi_labels
