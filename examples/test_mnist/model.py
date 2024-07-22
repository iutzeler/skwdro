import torch as pt
import torch.nn as nn
import torchvision
from torchvision.transforms import v2 as transforms

class TestAlexnet(nn.Module):
    def __init__(self, net: nn.Module, device):
        super().__init__()
        self.device = device
        self.net = net
        for p in net.parameters():
            p.requires_grad_(False)

        last_layer = self.net.classifier[-1]#[6]
        nn.init.kaiming_normal_(last_layer.weight.data, nonlinearity='relu') # type: ignore
        _ = [*map(
            lambda p: p.requires_grad_(True),
            last_layer.parameters()
        )]
        last_layer.train()

        self.classif = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1000, 10, bias=True),
        )
        self.preprocesspipe = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToImage(), transforms.ToDtype(pt.float32, scale=True),
            transforms.Lambda( lambda x: x.repeat(3, 1, 1) ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @pt.no_grad()
    def preprocess(self, x):
        x = self.preprocesspipe(x).unsqueeze(0)
        x = self.net.features(
                        x.to(self.device)
                    )
        x = self.net.avgpool(
                    x
                ).flatten(1)
        return self.net.classifier[:-1](x).squeeze(0)

    def forward(self, x):
        x = self.net.classifier[-1](x)
        return self.classif(
            x
        )

    def train(self, mode: bool = True):
        self.classif.train(mode)
        return self
    def eval(self):
        self.classif.eval()
        return self

def make_alexnet(device):
    model = pt.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=torchvision.models.AlexNet_Weights.DEFAULT)
    model.eval()
    return TestAlexnet(model, device)
