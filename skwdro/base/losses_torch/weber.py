from typing import Optional

import torch as pt
import torch.nn as nn


class SimpleWeber(nn.Module):
    def __init__(self, d: int) -> None:
        super(SimpleWeber, self).__init__()
        self.pos = nn.Parameter(pt.zeros(d))
        self.d = d

    def forward(self, xi: pt.Tensor, xi_labels: pt.Tensor) -> pt.Tensor:
        distances = pt.linalg.norm(
            xi - self.pos.unsqueeze(0), dim=-1, keepdims=True
        )
        val = xi_labels * distances * xi_labels.shape[1]
        assert isinstance(val, pt.Tensor)
        return val
