import torch

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
    
