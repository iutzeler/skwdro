import torch as pt
import torch.nn as nn

class Loss:
    """ Base class for loss functions """

    def value(self,theta,xi):
        raise NotImplementedError("Please Implement this method")


class NewsVendorLoss_torch(Loss):

    def __init__(self, k=5, u=7, name="NewsVendor loss"):
        self.k = k
        self.u = u
        self.name = name

    def value(self,theta,xi):
        return self.k*theta-self.u*pt.minimum(theta,xi)

class WeberLoss_torch(Loss):

    def __init__(self, name="Weber loss"):
        self.name = name

    def value(self,y,x,w):
        return w*pt.linalg.norm(x-y)

class LogisticLoss(Loss, nn.Module):
    def __init__(self, d: int=0, fit_intercept: bool=False) -> None:
        super(LogisticLoss, self).__init__()
        assert d > 0, "Please provide a valid data dimension d>0"
        self.linear = nn.Linear(d, 1, bias=fit_intercept)
        self.classif = nn.Tanh()
        self.L = nn.BCEWithLogitsLoss()

    def forward(self, X):
        coefs = self.linear(X)
        return self.classif(coefs), coefs

    def value(self, X, y):
        _, coefs = self.__call__(X)
        return self.L(
                coefs,
                (y == 1).long(),
                reduction='none')
