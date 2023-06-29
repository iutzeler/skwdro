from .base_loss import Loss

import numpy as np

class QuadraticLoss(Loss):

    def __init__(self, l2_reg=None, name="Logistic loss", fit_intercept = True):
        self.l2_reg = l2_reg
        self.name = name

    def value_split(self,theta,X,y,intercept=0.0):
        if len(X.shape) > 2:
            # Parallelized
            return self._parallel_value_split(theta, X, y)
        else:
            m = np.size(y)

            if self.l2_reg != None:
                raise NotImplementedError("l2 regression is not yet available")

            if m == 1:
                return 0.5*np.linalg.norm(np.dot(X,theta)+intercept-y)**2
            else:
                val = 0
                for i in range(m):
                    val += 0.5*np.linalg.norm(np.dot(X[i,:],theta)+intercept-y[i])**2

                return val/m


    def _parallel_value_split(self, theta, X, y):
        # New parallelized:
        # shapes in:
        # X(:=zeta): (n_samples, m, d)
        # y(:=zeta_labels): (n_samples, m, 1)
        # theta: (d,)
        # shapes out:
        # value: (n_samples, m)
        # NOTE: no mean on m !!!!
        linear = np.einsum("ijk,k->ij", X, theta)[:, :, None] - y # https://stackoverflow.com/questions/42983474/how-do-i-do-an-einsum-that-mimics-keepdims
        return 0.5*linear*linear

    def grad_theta_split(self,theta,X,y,intercept=0.0):
        if len(X.shape) > 2:
            # Parallelized
            return self._parallel_grad_theta_split(theta, X, y)
        else:
            m = np.size(y)

            if self.l2_reg != None:
                raise NotImplementedError("l2 regression is not yet available")

            if m == 1:
                return np.dot(X.T , (np.dot(X,theta)+intercept-y) )
            else:
                return np.dot(X.T , (np.dot(X,theta)+intercept-y) )

                # np.zeros(theta.shape)
                # for i in range(m):
                #     inner = np.dot(X[i,:],theta)+intercept-y[i]
                #     print(inner.shape)
                #     grad += np.dot(X[i,:].T , inner )

                # return grad/m


    def _parallel_grad_theta_split(self, theta, X, y):
        # New parallelized:
        # shapes in:
        # X(:=zeta): (n_samples, m, d)
        # y(:=zeta_labels): (n_samples, m, 1)
        # theta: (d,)
        # shapes out:
        # grad: (n_samples, m, d)
        # NOTE: no mean on m !!!!
        linear = np.einsum("ijk,k->ij", X, theta)[:, :, None] - y # https://stackoverflow.com/questions/42983474/how-do-i-do-an-einsum-that-mimics-keepdims
        grads   = X*linear
        return grads


    def grad_interceptSplit(self,theta,X,y,intercept=0.0):
        m = np.size(y)

        if self.l2_reg != None:
            raise NotImplementedError("l2 regression is not yet available")

        if m == 1:
            return  (np.dot(X,theta)+intercept-y)
        else:
            grad = np.zeros(theta.shape)
            for i in range(m):
                grad += (np.dot(X[i,:],theta)+intercept-y[i])

            return grad/m
