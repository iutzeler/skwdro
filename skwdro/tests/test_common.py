import pytest

from sklearn.base import clone
from sklearn.utils.estimator_checks import check_estimator

from skwdro.operations_research import Weber, NewsVendor, Portfolio
from skwdro.linear_models import LogisticRegression, LinearRegression
from skwdro.neural_network import ShallowNet


dict_wdro_estimators = {}


dict_wdro_estimators["Weber (entropic presampled)"] = Weber(solver="entropic_torch_pre")

dict_wdro_estimators["Portfolio"] = Portfolio(cost="n-NC-1-1")

#dict_wdro_estimators["Portfolio_entropic"] = Portfolio(solver="entropic_torch")

dict_wdro_estimators["NewsVendor"] = NewsVendor(solver="dedicated")

dict_wdro_estimators["Logistic"] = LogisticRegression(solver="dedicated")

dict_wdro_estimators["Logistic (entropic presampled)"] = LogisticRegression(solver="entropic_torch_pre")

dict_wdro_estimators["LinearReg"] = LinearRegression(solver="dedicated")

dict_wdro_estimators["LinearReg (entropic presampled)"] = LinearRegression(solver="entropic_torch_pre")

dict_wdro_estimators["ShallowNet (entropic presampled)"] = ShallowNet(solver="entropic_torch_pre")

@pytest.mark.parametrize(
    "estimator_name",
    dict_wdro_estimators.keys()
)
def test_all_estimators(estimator_name):

    if estimator_name == "NewsVendor":
        pytest.xfail("NewsVendor is 1D so not for check in sklearn")

    if estimator_name == "Weber (entropic presampled)":
        pytest.xfail("TODO: Investigate Weber")

    if estimator_name == "ShallowNet (entropic presampled)":
        pytest.xfail("TODO: Investigate the behavior of Shallownet")

    if estimator_name.endswith("(entropic presampled)"):
        pytest.xfail("TODO: fix presample")

    est = clone(dict_wdro_estimators[estimator_name])
    return check_estimator(est)
