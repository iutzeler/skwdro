import pytest

from sklearn.utils.estimator_checks import check_estimator

from skwdro.operations_research import Weber, Portfolio

Weber_rob_est = Weber()
Rob_Portfolio = Portfolio()

@pytest.mark.parametrize(
    "estimator",
    [Weber_rob_est, Rob_portfolio]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
