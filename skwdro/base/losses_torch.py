import torch
import numpy as np
from sqwash import SuperquantileReducer

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
        return self.k*theta-self.u*torch.minimum(theta,xi)

class WeberLoss_torch(Loss):

    def __init__(self, name="Weber loss"):
        self.name = name

    def value(self,y,x,w):
        return w*torch.norm(x-y)
    
class PortfolioLoss_torch(Loss):

    def __init__(self, eta, alpha, name="Portfolio loss"):
        self.eta = eta
        self.alpha = alpha
        self.name = name

    def value(self, theta, X):
        #Conversion np.array to torch.tensor if necessary
        if isinstance(theta, (np.ndarray,np.generic)):
            theta = torch.from_numpy(theta)
        if isinstance(X, (np.ndarray,np.generic)):
            X = torch.from_numpy(X)

        N = X.size()[0]

        #We add a double cast in the dot product to solve torch type issues for torch.dot
        in_sample_products = torch.tensor([torch.dot(theta, X[i].double()) for i in range(N)]) 
        expected_value = -(1/N) * torch.sum(in_sample_products)
        reducer = SuperquantileReducer(superquantile_tail_fraction=self.alpha)
        reduce_loss = reducer(in_sample_products)

        return expected_value + self.eta*reduce_loss