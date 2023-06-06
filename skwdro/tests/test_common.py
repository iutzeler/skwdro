import pytest

from sklearn.utils.estimator_checks import check_estimator

from skwdro.operations_research import Weber
from skwdro.linear_models import LogisticRegression

rob_Weber = Weber()
rob_LogReg = LogisticRegression()

@pytest.mark.parametrize(
    "estimator",
    [rob_Weber, rob_LogReg]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
