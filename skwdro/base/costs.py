import numpy as np

ENGINES_NAMES = {
    "pt": "PyTorch tensors",
    "np": "Numpy arrays",
    "jx": "Jax arrays"
}

class Cost:
    """ Base class for transport functions """

    def __init__(self, name: str="", engine: str=""):
        self.name = name
        self.engine = engine

    def value(self, x, y):
        raise NotImplementedError("Please Implement this method")

    def __str__(self) -> str:
        return "Cost named " + self.name + " using as data: " + ENGINES_NAMES[self.engine]


class NormCost(Cost):
    """ p-norm to some power """

    def __init__(self, p=1.0, power=1.0):
        super().__init__(name="Norm", engine="np")
        self.p = p
        self.power = power

    def value(self,x,y):
        diff = np.array(x-y).flatten()
        return np.linalg.norm(diff,ord=self.p)**self.power
