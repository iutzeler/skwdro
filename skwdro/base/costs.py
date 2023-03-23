import numpy as np

class Cost:
    """ Base class for transport functions """

    def __init__(self):
        pass

    def value(self,x,y):
        raise NotImplementedError("Please Implement this method")
    


class NormCost(Cost):
    """ p-norm to some power """

    def __init__(self, p = 1.0, power = 1.0, name="Norm"):
        self.p = p
        self.power = power

    def value(self,x,y):
        diff = np.array(x-y).flatten()
        return np.linalg.norm(diff,ord=self.p)**self.power
