import torch as pt
import torch.nn as nn

class TestAlexnet(nn.Module):
    def __init__(self, net: nn.Module, upscale: int=255):
        super().__init__()
        self.net = net
        self.upscale = upscale
        nn.init.kaiming_normal_(self.net.classifier[6].weight.data, nonlinearity='relu') # type: ignore
        self.classif = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1000, 10, bias=True),
        )

    def forward(self, x):
        B, N = x.shape
        assert B > 0, "Please provide non-empty batches"
        end_wh = (self.upscale // 8) * 8
        image = nn.functional.interpolate(x.reshape(B, 3, 8, 8), size=(end_wh, end_wh), mode='nearest')
        return self.classif(self.net(image))

def make_alexnet():
    model = pt.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    return TestAlexnet(model)
