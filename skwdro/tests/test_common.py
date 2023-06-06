import pytest

from sklearn.utils.estimator_checks import check_estimator

from skwdro.operations_research import Weber
from skwdro.linear_models import LogisticRegression

rob_Weber = Weber()
rob_LogReg = LogisticRegression(
        rho=1e-4,
        l2_reg=None,
        fit_intercept=True,
        solver="dedicated")

@pytest.mark.parametrize(
    "estimator",
    [rob_LogReg]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
