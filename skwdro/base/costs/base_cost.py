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

    def value(self, x, y, *args):
        raise NotImplementedError("Please Implement this method")

    def __str__(self) -> str:
        return "Cost named " + self.name + " using as data: " + ENGINES_NAMES[self.engine]

    def sampler(self, xi, xi_labels, epsilon):
        return self._sampler_data(xi, epsilon), self._sampler_labels(xi_labels, epsilon)

    def _sampler_data(self, xi, epsilon):
        raise NotImplementedError()

    def _sampler_labels(self, xi_labels, epsilon):
        raise NotImplementedError()
