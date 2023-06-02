import pytest

from sklearn.utils.estimator_checks import check_estimator

from skwdro.operations_research import Weber

Weber_rob_est = Weber()

@pytest.mark.parametrize(
    "estimator",
    [Weber_rob_est]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
