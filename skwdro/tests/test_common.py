import pytest

from sklearn.base import clone
from sklearn.utils.estimator_checks import check_estimator

from skwdro.operations_research import Weber, NewsVendor, Portfolio
from skwdro.linear_models import LogisticRegression, LinearRegression


dict_wdro_estimators = {}


dict_wdro_estimators["Weber"] = Weber()
dict_wdro_estimators["Portfolio"] = Portfolio()

dict_wdro_estimators["NewsVendor"] = NewsVendor(solver="dedicated")

dict_wdro_estimators["Logistic"] = LogisticRegression(
        rho=1e-4,
        l2_reg=None,
        fit_intercept=True,
        solver="dedicated")

dict_wdro_estimators["LinearReg"] = LinearRegression(
        rho=1e-4)

@pytest.mark.parametrize(
    "estimator_name",
    dict_wdro_estimators.keys()
)
def test_all_estimators(estimator_name):

    if estimator_name == "Weber":
        pytest.xfail("Weber is not checked due to the random behavior of the entropic solver") # Issue #21

    if estimator_name == "NewsVendor":
        pytest.xfail("NewsVendor is 1D so not for check in sklearn")

    est = clone(dict_wdro_estimators[estimator_name])
    return check_estimator(est)
