import pytest

from sklearn.utils.estimator_checks import check_estimator

from skwdro.operations_research import Portfolio

Rob_Portfolio = Portfolio()

@pytest.mark.parametrize(
    "estimator",
    [Rob_Portfolio]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
