import pytest

from sklearn.base import clone
from sklearn.utils.estimator_checks import check_estimator

from skwdro.operations_research import Weber, NewsVendor
from skwdro.linear_models import LogisticRegression, LinearRegression


dict_wdro_estimators = {}


dict_wdro_estimators["Weber"] = Weber() 

dict_wdro_estimators["NewsVendor"] = NewsVendor(solver="dedicated")

dict_wdro_estimators["Logistic"] = LogisticRegression(solver="dedicated")
dict_wdro_estimators["Logistic_ent"] = LogisticRegression(solver="entropic")
dict_wdro_estimators["Logistic_torch"] = LogisticRegression(solver="entropic_torch")

dict_wdro_estimators["LinearReg"] = LinearRegression(solver="dedicated")

@pytest.mark.parametrize(
    "estimator_name",
    ["Weber", "NewsVendor", "Logistic", "Logistic_torch", "LinearReg"]
)
def test_all_estimators(estimator_name):

    if estimator_name == "Weber":
        pytest.xfail("Weber is not checked due to the random behavior of the entropic solver") # Issue #21

    if estimator_name == "NewsVendor":
        pytest.xfail("NewsVendor is 1D so not for check in sklearn")

    est = clone(dict_wdro_estimators[estimator_name])
    return check_estimator(est)
