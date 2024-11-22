"""
Logistic Regression
"""
from typing import Optional
import numpy as np
import torch as pt
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.exceptions import DataConversionWarning
from scipy.special import expit


from skwdro.base.problems import EmpiricalDistributionWithLabels
from skwdro.base.losses_torch.logistic import BiDiffSoftMarginLoss
from skwdro.solvers.optim_cond import OptCondTorch
from skwdro.base.cost_decoder import cost_from_str

import skwdro.solvers.specific_solvers as spS
import skwdro.solvers.entropic_dual_torch as entTorch
from skwdro.wrap_problem import dualize_primal_loss


class LogisticRegression(BaseEstimator, ClassifierMixin):
    r""" A Wasserstein Distributionally Robust logistic regression classifier.


    The cost function is XXX

    Uncertainty is XXX

    Parameters
    ----------
    rho: float, default=1e-2
        Robustness radius
    l2_reg: float, default=None
        l2 regularization
    fit_intercept: boolean, default=True
        Determines if an intercept is fit or not
    cost: str, default="n-NC-1-2"
        Tiret-separated code to define the transport cost: "<engine>-<cost id>-<k-norm type>-<power>" for :math:`c(x, y):=\|x-y\|_k^p`
    solver: str, default='entropic_torch'
        Solver to be used: 'entropic', 'entropic_torch' (_pre or _post) or 'dedicated'
    solver_reg: float, default=1e-2
        regularization value for the entropic solver
    n_zeta_samples: int, default=10
        number of adversarial samples to draw
    opt_cond: Optional[OptCondTorch]
        optimality condition, see :py:class:`OptCondTorch`

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (:math:`w` in the cost function formula)
    intercept_ : float
        constant term in decision function.

    Examples
    --------
    >>> import numpy as np
    >>> from skwdro.linear_models import LogisticRegression
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=0)
    >>> y = np.sign(y-0.5)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    >>> estimator = LogisticRegression()
    >>> estimator.fit(X_train,y_train)
    LogisticRegression()
    >>> estimator.predict(X_test)
    array([-1., -1., -1.,  1., -1.,  1.,  1., -1., -1.,  1.,  1.,  1., -1.,
            1.,  1.,  1.,  1.,  1., -1., -1., -1.,  1.,  1., -1., -1.,  1.,
           -1.,  1.,  1.,  1.,  1.,  1., -1.])
    >>> estimator.score(X_test,y_test)
    0.9393939393939394
    """

    def __init__(self,
                 rho: float = 1e-2,
                 l2_reg: float = 0.,
                 fit_intercept: bool = True,
                 cost: str = "t-NLC-2-2",
                 solver="entropic_torch",
                 solver_reg: Optional[float] = None,
                 sampler_reg: Optional[float] = None,
                 n_zeta_samples: int = 10,
                 random_state: int = 0,
                 opt_cond: Optional[OptCondTorch] = OptCondTorch(2, 1e-4, 0.)
                 ):

        if rho < 0:
            raise ValueError(
                f"The uncertainty radius rho should be non-negative, received {rho}")

        self.rho = rho
        self.l2_reg = l2_reg
        self.cost = cost
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.solver_reg = solver_reg
        self.sampler_reg = sampler_reg  # sigma
        self.opt_cond = opt_cond
        self.n_zeta_samples = n_zeta_samples
        self.random_state = random_state

    def fit(self, X, y):
        """Fits the WDRO classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int. Only -1 or +1 are currently supported

        Returns
        -------
        self : LogisticRegression
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        X = np.array(X)
        y = np.array(y)

        # Type checking for rho
        if self.rho is not float:
            try:
                self.rho = float(self.rho)
            except BaseException:
                raise TypeError(
                    f"The uncertainty radius rho should be numeric, received {type(self.rho)}")

        if len(y.shape) != 1:
            y = y.ravel()
            raise DataConversionWarning(
                f"Given y is {y.shape}, while expected shape is (n_sample,)")

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Encode the labels
        if len(self.classes_) == 2:
            # Binary classification
            self.le_ = LabelEncoder()
            y = self.le_.fit_transform(y)
            if y is None:
                raise ValueError("Problem with labels, none out of label encoder")
            else:
                y = np.array(y, dtype=X.dtype)
            y[y == 0.] = -1.
        elif len(self.classes_) > 2:
            # Multiclass classification
            self.le_ = OneHotEncoder(sparse_output=False)
            y = self.le_.fit_transform(y[:, None])
            if y is None:
                raise ValueError("Problem with labels, none out of label encoder")
            else:
                y = np.array(y, dtype=X.dtype)
        else:
            raise ValueError(f"Found {len(self.classes_)} classes, while 2 are expected.")

        # Check type
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError(f"Input X has dtype  {X.dtype}")

        # Store data
        self.X_ = X
        self.y_ = y

        # Setup problem parameters ################
        m, d = np.shape(X)
        self.n_features_in_ = d
        samples_y = y[:, None] if len(self.classes_) == 2 else y
        emp = EmpiricalDistributionWithLabels(m=m, samples_x=X, samples_y=samples_y)

        # Define cleanly the hyperparameters of the problem.
        self.cost_ = cost_from_str(self.cost)
        # #########################################

        if self.solver == "entropic":
            raise (DeprecationWarning(
                "The entropic (numpy) solver is now deprecated"))
        elif self.solver == "dedicated":
            if len(self.classes_) > 2:
                raise NotImplementedError(
                    f"Multiclass classification is not implemented for dedicated solver. ({len(self.classes_)} classes were found : {self.classes_})")

            # The logistic regression has a dedicated MP problem-description (solved using cvxopt)
            # One may use it by specifying this option
            self.coef_, self.intercept_, self.dual_var_, self.result_ = spS.WDROLogisticSpecificSolver(
                rho=self.rho,
                kappa=1000,
                X=X,
                y=y,
                fit_intercept=self.fit_intercept
            )
        elif "torch" in self.solver:
            _post_sample = self.solver == "entropic_torch" or self.solver == "entropic_torch_post"
            module_out = 1 if len(self.classes_) == 2 else len(self.classes_)
            loss = BiDiffSoftMarginLoss(reduction='none') if len(self.classes_) == 2 else nn.CrossEntropyLoss(reduction='none')
            self.wdro_loss_ = dualize_primal_loss(
                loss,
                nn.Linear(self.n_features_in_, module_out, bias=self.fit_intercept),
                pt.tensor(self.rho),
                pt.Tensor(emp.samples_x),
                pt.Tensor(emp.samples_y),
                _post_sample,
                self.cost,
                self.n_zeta_samples,
                self.random_state,
                sigma=self.sampler_reg,
                epsilon=self.solver_reg,
                imp_samp=_post_sample,  # hard set
                adapt="prodigy",
                l2reg=self.l2_reg
            )

            # The problem is solved with the new "dual loss"
            self.coef_, self.intercept_, self.dual_var_, self.robust_loss_ = entTorch.solve_dual_wdro(
                self.wdro_loss_,
                emp,
                self.opt_cond  # type: ignore
            )

            self.coef_ = self.wdro_loss_.primal_loss.transform.weight.detach().cpu().squeeze().numpy()

            if self.fit_intercept:
                self.intercept_ = self.wdro_loss_.primal_loss.transform.bias.detach().cpu().squeeze().numpy()
            else:
                self.intercept_ = None

            # # TODO: deprecate ?
            # # Stock the robust loss result
            # Problems w/ dtypes (f32->f64 for some reason)
            # To be fixed later
            # =============================================
            # Stock the robust loss result
            # if self.solver == "entropic_torch_pre":
            #     #self.result_ = _wdro_loss.forward(xi=self.X_, xi_labels=self.y_, zeta=?, zeta_labels=?)
            #     #raise NotImplementedError("Result for pre_sample not available")
            #     pass
            # elif self.solver == "entropic_torch" or self.solver == "entropic_torch_post":
            #     self.result_ = _wdro_loss.forward(xi=pt.from_numpy(emp.samples_x), xi_labels=pt.from_numpy(emp.samples_y)).item()

        else:
            raise NotImplementedError()
        self.is_fitted_ = True

        # Return the classifier
        return self

    def predict_proba_2Class(self, X):
        """ Robust prediction probability for class +1.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : ndarray, shape (n_samples,)
            The probability of class +1 for each of the samples.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        if self.intercept_ is not None:
            p = expit(X.dot(self.coef_) + self.intercept_)
        else:
            p = expit(X.dot(self.coef_))
        # p = 1 / (1 + np.exp(-(X.dot(self.coef_)+self.intercept_)))
        return p

    def predict_proba(self, X):
        """ Robust prediction probability for classes -1 and +1.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : ndarray, shape (n_samples,2)
            The probability of each class for each of the samples.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)
        p = expit(X.dot(self.coef_) + self.intercept_)
        return np.vstack((1 - p, p)).T

    def predict(self, X):
        """ Robust prediction.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)
        X = np.array(X)

        proba = self.predict_proba_2Class(X)
        out = self.le_.inverse_transform((proba >= 0.5).astype('uint8'))
        return out

    def _more_tags(self):
        return {'poor_score': True,  # In order to pass with any rho...
                'binary_only': True,  # Only binary classification
                # 'non_deterministic': True # For stochastic methods
                }
