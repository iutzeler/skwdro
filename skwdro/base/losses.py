import numpy as np
from scipy.special import logsumexp, expit, log_expit

class Loss:
    """ Base class for loss functions """

    def value(self,theta,xi):
        raise NotImplementedError("Please Implement this method")

    def grad_theta(self,theta,xi):
        raise NotImplementedError("Please Implement this method")


class NewsVendorLoss(Loss):

    def __init__(self, k=5, u=7, name="NewsVendor loss"):
        self.k = k
        self.u = u
        self.name = name

    def value(self,theta,xi):
        return self.k*theta-self.u*np.minimum(theta,xi)

    def _parallel_grad_theta(self, theta, X):
        # New parallelized:
        # shapes in:
        # X(:=zeta): (n_samples, m, 1)
        # theta: (1,)
        # shapes out:
        # grad: (n_samples, m, 1)
        # NOTE: no mean on m !!!!
        grads = self.k*np.ones_like(X) - self.u*(X>theta).astype(int)
        return grads
      
    def grad_theta(self,theta,xi):
        if len(xi) >= 2:
            # Parallelized
            return self._parallel_grad_theta(theta, xi)
        else:
            if theta>=xi :
                return self.k
            else:
                return self.k-self.u

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


    def valueSplit(self,theta,X,y,intercept=0.0):
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

    def grad_thetaSplit(self,theta,X,y,intercept=0.0):
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


class QuadraticLoss(Loss):

    def __init__(self, l2_reg=None, name="Logistic loss", fit_intercept = True):
        self.l2_reg = l2_reg
        self.name = name

    def valueSplit(self,theta,X,y,intercept=0.0):
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
    
    def grad_thetaSplit(self,theta,X,y,intercept=0.0):
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


