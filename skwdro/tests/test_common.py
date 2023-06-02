import pytest

from sklearn.utils.estimator_checks import check_estimator


@pytest.mark.parametrize(
    "estimator",
    []
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
