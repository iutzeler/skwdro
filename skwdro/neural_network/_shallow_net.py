"""
Neural Network, 1 layer Relu, 1 linear
"""

import numpy as np
import torch as pt

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from skwdro.base.problems import EmpiricalDistributionWithLabels
from skwdro.neural_network._loss_shallownet import ShallowNetLoss as ShallowNetLossTorch
from skwdro.base.costs_torch import NormLabelCost


import skwdro.solvers.entropic_dual_torch as entTorch
from skwdro.solvers.oracle_torch import DualLoss, DualPreSampledLoss

from skwdro.solvers.optim_cond import OptCondTorch


class ShallowNet(BaseEstimator, RegressorMixin):  # ClassifMixin
    """ A Wasserstein Distributionally Robust shallow network.


    The cost function is XXX
    Uncertainty is XXX

    Parameters
    ----------
    rho : float, default=1e-2
        Robustness radius
    n_neurons : int, default=10
        Number of ReLU neurons
    fit_intercept : boolean, default=True
        If true, layers will include a bias
    cost: Loss, default=NormCost(p=2)
        Transport cost
    solver: str, default='entropic'
        Solver to be used: 'entropic' or 'dedicated'
    solver_reg: float, default=1.0
        regularization value for the entropic solver

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (:math:`w` in the cost function formula)
    intercept_ : float
        constant term in decision function.

    Examples
    --------
    >>> import numpy as np
    >>> TODO
    """

    def __init__(self,
                 rho=1e-2,
                 l2_reg=None,
                 fit_intercept=True,
                 cost="t-NLC-2-2",
                 solver="entropic_torch",
                 solver_reg=0.01,
                 n_zeta_samples: int = 10,
                 n_neurons: int = 10,
                 ly1=None,
                 ly2=None,
                 random_state: int = 0,
                 opt_cond=OptCondTorch(2)
                 ):

        if rho is not float:
            try:
                rho = float(rho)
            except TypeError:
                raise TypeError(
                    f"The uncertainty radius rho should be numeric, received {type(rho)}")

        if rho < 0:
            raise ValueError(
                f"The uncertainty radius rho should be non-negative, received {rho}")

        self.rho = rho
        self.l2_reg = l2_reg
        self.cost = cost
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.solver_reg = solver_reg
        self.opt_cond = opt_cond
        self.n_zeta_samples = n_zeta_samples
        self.n_neurons = n_neurons
        self.ly1 = ly1
        self.ly2 = ly2
        self.random_state = random_state

    def fit(self, X, y):
        """Fits the WDRO neural network.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int. Only -1 or +1 are currently supported

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, y_numeric=True)
        X = np.array(X)
        y = np.array(y)

        if len(y.shape) != 1:
            y.flatten()
            raise DataConversionWarning(
                f"y expects a shape (n_samples,) but receiced shape {y.shape}")

        # Store data
        self.X_ = X
        self.y_ = y

        m, d = np.shape(X)
        self.n_features_in_ = d

        # Setup problem parameters ################
        emp = EmpiricalDistributionWithLabels(
            m=m, samples_x=X, samples_y=y[:, None]
        )
        # #########################################

        if self.solver == "entropic_torch" or self.solver == "entropic_torch_post":
            _wdro_loss = DualLoss(
                ShallowNetLossTorch(n_neurons=self.n_neurons, d=d,
                                    fit_intercept=self.fit_intercept, ly1=self.ly1, ly2=self.ly2),
                NormLabelCost(2., 1., 1e8),
                n_samples=self.n_zeta_samples,
                epsilon_0=pt.tensor(self.solver_reg),
                rho_0=pt.tensor(self.rho)
            )

            self.coef_, self.intercept_, self.dual_var_, self.robust_loss_ = entTorch.solve_dual_wdro(
                _wdro_loss, emp, self.opt_cond)

            self.parameters_ = _wdro_loss.primal_loss.parameters_iter
        elif self.solver == "entropic_torch_pre":
            _wdro_loss = DualPreSampledLoss(
                ShallowNetLossTorch(n_neurons=self.n_neurons,
                                    d=d, fit_intercept=self.fit_intercept),
                NormLabelCost(2., 1., 1e8),
                n_samples=self.n_zeta_samples,
                epsilon_0=pt.tensor(self.solver_reg),
                rho_0=pt.tensor(self.rho)
            )

            self.coef_, self.intercept_, self.dual_var_, self.robust_loss_ = entTorch.solve_dual_wdro(
                _wdro_loss, emp, self.opt_cond)

            self.parameters_ = _wdro_loss.primal_loss.parameters_iter
        elif self.solver == "entropic":
            raise NotImplementedError
        elif self.solver == "dedicated":
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.is_fitted_ = True

        # Return the classifier
        return self

    def predict(self, X):
        """ Robust prediction.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The prediction
        """

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)
        X = pt.tensor(X, dtype=pt.float32, device="cpu")
        model = ShallowNetLossTorch(
            None, n_neurons=self.n_neurons, d=self.n_features_in_, fit_intercept=self.fit_intercept)
        model.load_state_dict(self.parameters_)  # load

        return model.pred(X).cpu().detach().numpy().flatten()

    def params(self):
        """ Return the network's parameters in a standard format
        Returns
        -------
        ly1 : ndarray, shape (n_neurons, data_dim+1)
            First layer with bias concatenated
        ly2 : ndarray, shape (1, n_neurons)
            Second layer
        """

        ly1nb = self.parameters_["linear1.weight"].cpu().detach().numpy()
        ly1b = self.parameters_["linear1.bias"].cpu().detach().numpy()
        ly1 = np.hstack((ly1nb, ly1b[:, None]))
        ly2 = self.parameters_["linear2.weight"].cpu().detach().numpy()
        # otherwise it's just a pointer to memory...
        return np.array(ly1), np.array(ly2)
