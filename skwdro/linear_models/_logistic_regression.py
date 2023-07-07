"""
Logistic Regression
"""
from typing import Optional
import numpy as np
import torch as pt
from sklearn.base import BaseEstimator, ClassifierMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import DataConversionWarning
from scipy.special import expit



from skwdro.base.problems import WDROProblem, EmpiricalDistributionWithLabels
from skwdro.base.losses import LogisticLoss
from skwdro.base.losses_torch import LogisticLoss as LogisticLossTorch
from skwdro.base.samplers.torch import LabeledCostSampler
from skwdro.solvers.optim_cond import OptCond
from skwdro.base.cost_decoder import cost_from_str

import skwdro.solvers.specific_solvers as spS
import skwdro.solvers.entropic_dual_solvers as entS
import skwdro.solvers.entropic_dual_torch as entTorch
from skwdro.solvers.oracle_torch import DualLoss, DualPreSampledLoss

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
    opt_cond: Optional[OptCond]
        optimality condition, see :py:class:`OptCond`

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
                 rho: float=1e-2,
                 l2_reg: float=0.,
                 fit_intercept: bool=True,
                 cost: str="t-NLC-2-2",
                 solver="entropic_torch",
                 solver_reg=0.01,
                 n_zeta_samples: int=10,
                 random_state: int=0,
                 opt_cond: Optional[OptCond]=OptCond(2)
                 ):

        if rho < 0:
            raise ValueError(f"The uncertainty radius rho should be non-negative, received {rho}")

        self.rho            = rho
        self.l2_reg         = l2_reg
        self.cost           = cost
        self.fit_intercept  = fit_intercept
        self.solver         = solver
        self.solver_reg     = solver_reg
        self.opt_cond       = opt_cond
        self.n_zeta_samples = n_zeta_samples
        self.random_state   = random_state



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

        #Type checking for rho
        if self.rho is not float:
            try:
                self.rho = float(self.rho)
            except:
                raise TypeError(f"The uncertainty radius rho should be numeric, received {type(self.rho)}")

        if len(y.shape) != 1:
            y = y.ravel()
            raise DataConversionWarning(f"Given y is {y.shape}, while expected shape is (n_sample,)")

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.le_ = LabelEncoder()
        y = self.le_.fit_transform(y)
        if y is None: raise ValueError("Problem with labels, none out of label encoder")
        else: y = np.array(y)
        y[y==0] = -1

        if len(self.classes_)>2:
            raise NotImplementedError(f"Multiclass classificaion is not implemented. ({len(self.classes_)} classes were found : {self.classes_})")

        if len(self.classes_)<2:
            raise ValueError(f"Found {len(self.classes_)} classes, while 2 are expected.")

        if not np.issubdtype(X.dtype, np.number):
            raise ValueError(f"Input X has dtype  {X.dtype}")

        # Store data
        self.X_ = X
        self.y_ = y

        # Setup problem parameters ################
        m, d = np.shape(X)
        self.n_features_in_ = d
        emp = EmpiricalDistributionWithLabels(m=m,samples_x=X,samples_y=y[:,None])

        # Define cleanly the hyperparameters of the problem.
        self.cost_ = cost_from_str(self.cost)
        self.problem_ = WDROProblem(
                cost=self.cost_,
                loss=LogisticLoss(l2_reg=self.l2_reg),
                p_hat=emp,
                n=d,
                d=d,
                d_labels=1,
                xi_bounds=[-1e8,1e8],
                theta_bounds=[-1e8,1e8],
                rho=self.rho
            )
        # #########################################

        if self.solver=="entropic":
            if self.opt_cond is None:
                self.opt_cond = OptCond(2)
            # In the entropic case, we use the numpy gradient descent solver
            self.coef_ , self.intercept_, self.dual_var_ = entS.WDROEntropicSolver(
                    self.problem_,
                    fit_intercept=self.fit_intercept,
                    opt_cond=self.opt_cond
            )
        elif self.solver=="dedicated":
            # The logistic regression has a dedicated MP problem-description (solved using cvxopt)
            # One may use it by specifying this option
            self.coef_ , self.intercept_, self.dual_var_, self.result_ = spS.WDROLogisticSpecificSolver(
                    rho=self.problem_.rho,
                    kappa=1000,
                    X=X,
                    y=y,
                    fit_intercept=self.fit_intercept
            )
        elif "torch" in self.solver:
            custom_sampler = LabeledCostSampler(
                    cost,
                    pt.Tensor(self.problem_.p_hat.samples_x),
                    pt.Tensor(self.problem_.p_hat.samples_y),
                    epsilon=pt.tensor(self.solver_reg),
                    seed=self.random_state
                )
            # The problem loss is changed to a more suitable "dual loss"
            if self.solver == "entropic_torch" or self.solver == "entropic_torch_post":
                # Default torch implementation resamples from pi_0 at each SGD step
                self.problem_.loss = DualLoss(
                        LogisticLossTorch(custom_sampler, d=self.problem_.d, fit_intercept=self.fit_intercept),
                        cost,
                        n_iter=1000,
                        n_samples=self.n_zeta_samples,
                        epsilon_0=pt.tensor(self.solver_reg),
                        rho_0=pt.tensor(self.rho)
                    )

            elif self.solver == "entropic_torch_pre":
                # One may specify this option to use ~ the WangGaoXie algorithm, i.e. sample once and do BFGS steps
                self.problem_.loss = DualPreSampledLoss(
                        LogisticLossTorch(custom_sampler, d=self.problem_.d, fit_intercept=self.fit_intercept),
                        cost,
                        n_samples=self.n_zeta_samples,
                        epsilon_0=pt.tensor(self.solver_reg),
                        rho_0=pt.tensor(self.rho)
                    )
            else:
                raise NotImplementedError()

            # The problem is solved with the new "dual loss"
            self.coef_, self.intercept_, self.dual_var_ = entTorch.solve_dual(
                    self.problem_,
                    seed=self.random_state,
                    sigma_=self.solver_reg,
                )
            
            # Stock the robust loss result 
            if self.solver == "entropic_torch_pre":
                #self.result_ = self.problem_.loss.forward(xi=self.X_, xi_labels=self.y_, zeta=?, zeta_labels=?)
                raise NotImplementedError("Result for pre_sample not available")
            elif self.solver == "entropic_torch_post":
                self.result_ = self.problem_.loss.forward(xi=pt.from_numpy(self.X_), xi_labels=pt.from_numpy(self.y_))

        else:
            raise NotImplementedError()
        self.is_fitted_ = True

        # Return the classifier
        return self

    def predict_proba_2Class(self,X):
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
            p = expit(X.dot(self.coef_)+self.intercept_)
        else:
            p = expit(X.dot(self.coef_))
        # p = 1 / (1 + np.exp(-(X.dot(self.coef_)+self.intercept_)))
        return p

    def predict_proba(self,X):
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
        p = expit(X.dot(self.coef_)+self.intercept_)
        return np.vstack((1-p,p)).T

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
        out =  self.le_.inverse_transform((proba>=0.5).astype('uint8'))
        return out


    def _more_tags(self):
        return {'poor_score': True, # In order to pass with any rho...
                'binary_only': True, # Only binary classification
               # 'non_deterministic': True # For stochastic methods
                }
