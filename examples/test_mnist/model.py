import torch as pt
import torch.nn as nn
from math import isqrt

class TestAlexnet(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        nn.init.kaiming_normal_(self.net.classifier[6].weight.data, nonlinearity='relu') # type: ignore
        self.classif = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1000, 10, bias=True),
        )

    def forward(self, x):
        *batch_dims, d = x.shape
        assert d % 3 == 0 and d > 3
        hw = d // 3
        h = w = isqrt(hw)
        assert h * w == hw
        image = x.view(*batch_dims, 3, h, w)
        return self.classif(self.net(image))

def make_alexnet():
    model = pt.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    return TestAlexnet(model)
