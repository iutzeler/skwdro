from .base_loss import Loss

import numpy as np
from scipy.special import logsumexp, expit, log_expit

class LogisticLoss(Loss):

    def __init__(self, l2_reg=None, name="Logistic loss", fit_intercept = False):
        self.l2_reg = l2_reg
        self.name = name
        self.fit_intercept = fit_intercept

    def value(self, theta, xi):
        return SyntaxError

    def _parallel_value_split(self, theta, X, y):
        # New parallelized:
        # shapes in:
        # X(:=zeta): (n_samples, m, d)
        # y(:=zeta_labels): (n_samples, m, 1)
        # theta: (d,)
        # shapes out:
        # value: (n_samples, m)
        # NOTE: no mean on m !!!!
        linear = np.einsum("ijk,k->ij", X, theta)[:, :, None] # https://stackoverflow.com/questions/42983474/how-do-i-do-an-einsum-that-mimics-keepdims
        return -log_expit(y * linear)


    def value_split(self,theta,X,y,intercept=0.0):
        if len(X.shape) > 2:
            # Parallelized
            return self._parallel_value_split(theta, X, y)
        else:
            m = np.size(y)

            if self.l2_reg != None:
                raise NotImplementedError("l2 regression is not yet available")

            if m == 1:
                return logsumexp([0,-y*(np.dot(X,theta)+intercept)])
                #return np.log(1+np.exp(-y*(np.dot(X,theta)+intercept)))
            else:
                val = 0
                for i in range(m):
                    print(y,theta,X)
                    val += np.log(1+np.exp(-y[i]*(np.dot(X[i,:],theta)+intercept)))

                return val/m


    def _parallel_grad_theta_split(self, theta, X, y):
        # New parallelized:
        # shapes in:
        # X(:=zeta): (n_samples, m, d)
        # y(:=zeta_labels): (n_samples, m, 1)
        # theta: (d,)
        # shapes out:
        # grad: (n_samples, m, d)
        # NOTE: no mean on m !!!!
        linear = np.einsum("ijk,k->ij", X, theta)[:, :, None] # https://stackoverflow.com/questions/42983474/how-do-i-do-an-einsum-that-mimics-keepdims
        grads = -y*X * expit(-y*linear) # (n_samples, m, d)
        return grads

    def grad_theta_split(self,theta,X,y,intercept=0.0):
        if len(X.shape) > 2:
            # Parallelized
            return self._parallel_grad_theta_split(theta, X, y)
        else:
            m = np.size(y)

            if self.l2_reg != None:
                raise NotImplementedError("l2 regression is not yet available")

            if m == 1:
                return -y*X*expit(-y*(np.dot(X,theta)+intercept))
                #return -y*X/(1+np.exp(y*(np.dot(X,theta)+intercept)))
            else:
                grad = np.zeros(theta.shape)
                for i in range(m):
                    grad += -y[i]*X[i,:]/(1+np.exp(y[i]*(np.dot(X[i,:],theta)+intercept)))
                return grad/m

    def grad_interceptSplit(self,theta,X,y,intercept=0.0):
        m = np.size(y)

        if self.l2_reg != None:
            raise NotImplementedError("l2 regression is not yet available")

        if m == 1:
            return -y*expit(-y*(np.dot(X,theta)+intercept))
            #return y/(1+np.exp(y*(np.dot(X,theta)+intercept)))
        else:
            grad = np.zeros(theta.shape)
            for i in range(m):
                grad += -y[i]/(1+np.exp(y[i]*(np.dot(X[i,:],theta)+intercept)))

            return grad/m
