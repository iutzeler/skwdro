import pytest

from sklearn.utils.estimator_checks import check_estimator

from skwdro import TemplateEstimator
from skwdro import TemplateClassifier
from skwdro import TemplateTransformer


@pytest.mark.parametrize(
    "estimator",
    [TemplateEstimator(), TemplateTransformer(), TemplateClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
