"""
Logistic Regression
"""
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
from skwdro.base.costs import Cost, NormCost
from skwdro.base.costs_torch import NormLabelCost
from skwdro.solvers.optim_cond import OptCond

import skwdro.solvers.specific_solvers as spS
import skwdro.solvers.entropic_dual_solvers as entS
import skwdro.solvers.entropic_dual_torch as entTorch
from skwdro.solvers.oracle_torch import DualLoss, DualPreSampledLoss


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """ A Wasserstein Distributionally Robust logistic regression classifier.


    The cost function is XXX

    Uncertainty is XXX

    Parameters
    ----------
    rho : float, default=1e-2
        Robustness radius
    l2_reg  : float, default=None
        l2 regularization
    fit_intercept : boolean, default=True
        Determines if an intercept is fit or not
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
                 rho=1e-2,
                 l2_reg: int=0,
                 fit_intercept: bool=True,
                 cost="quad",
                 solver="entropic_torch",
                 solver_reg=0.01,
                 n_zeta_samples: int=10,
                 opt_cond=None
                 ):

        if rho is not float:
            try:
                rho = float(rho)
            except:
                raise TypeError(f"The uncertainty radius rho should be numeric, received {type(rho)}")

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

        self.cost_ = NormCost(p=2)

        # Define cleanly the hyperparameters of the problem.
        self.problem_ = WDROProblem(
                cost=self.cost_,
                Xi_bounds=[-1e8,1e8],
                Theta_bounds=[-1e8,1e8],
                rho=self.rho,
                loss=LogisticLoss(l2_reg=self.l2_reg),
                n=d,
                d=d,
                dLabel=1,
                P=emp
            )

        self.opt_cond_ = OptCond(2)
        # #########################################

        if self.solver=="entropic":
            # In the entropic case, we use the numpy gradient descent solver
            self.coef_ , self.intercept_, self.dual_var_ = entS.WDROEntropicSolver(
                    self.problem_,
                    fit_intercept=self.fit_intercept,
                    opt_cond=self.opt_cond_
            )
        elif self.solver=="dedicated":
            # The logistic regression has a dedicated MP problem-description (solved using cvxopt)
            # One may use it by specifying this option
            self.coef_ , self.intercept_, self.dual_var_ = spS.WDROLogisticSpecificSolver(
                    rho=self.problem_.rho,
                    kappa=1000,
                    X=X,
                    y=y,
                    fit_intercept=self.fit_intercept
            )
        elif "torch" in self.solver:
            # The problem loss is changed to a more suitable "dual loss"
            if self.solver == "entropic_torch" or self.solver == "entropic_torch_post":
                # Default torch implementation resamples from pi_0 at each SGD step
                self.problem_.loss = DualLoss(
                        LogisticLossTorch(None, d=self.problem_.d, fit_intercept=self.fit_intercept),
                        NormLabelCost(2., 1., 1e8),
                        n_samples=10,
                        epsilon_0=pt.tensor(self.solver_reg),
                        rho_0=pt.tensor(self.rho)
                    )

            elif self.solver == "entropic_torch_pre":
                # One may specify this option to use ~ the WangGaoXie algorithm, i.e. sample once and do BFGS steps
                self.problem_.loss = DualPreSampledLoss(
                        LogisticLossTorch(None, d=self.problem_.d, fit_intercept=self.fit_intercept),
                        NormLabelCost(2., 1., 1e8),
                        n_samples=10,
                        epsilon_0=pt.tensor(self.rho),
                        rho_0=pt.tensor(self.rho)
                    )
            else:
                raise NotImplementedError()

            # The problem is solved with the new "dual loss"
            self.coef_, self.intercept_, self.dual_var_ = entTorch.solve_dual(
                    self.problem_,
                    sigma_=self.solver_reg,
                )
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
