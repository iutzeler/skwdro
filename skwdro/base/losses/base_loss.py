class Loss:
    """ Base class for loss functions """

    def value(self,theta,xi):
        raise NotImplementedError("Please Implement this method")

    def grad_theta(self, theta, xi, xi_labels):
        raise NotImplementedError("Please Implement this method")

    def value_split(self, theta, xi, xi_labels):
        raise NotImplementedError("Please Implement this method")

    def grad_theta_split(self, theta, xi, xi_labels):
        raise NotImplementedError("Please Implement this method")
