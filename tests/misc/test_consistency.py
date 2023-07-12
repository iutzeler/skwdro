import numpy as np
from joblib import Parallel, delayed

from skwdro.linear_models import LogisticRegression

from tests.torch_tests.utils import generate_blob

def fit_new_model(data, data_l):
    model = LogisticRegression(
            rho=1e-3,
            cost="t-NLC-2-2",
            random_state=666
            )
    model.fit(data, data_l)
    return model.coef_


def test_consistency():
    params = []
    data, data_l = generate_blob(False)
    params = Parallel(n_jobs=4)(delayed(fit_new_model)(data, data_l) for _ in range(4))
    assert isinstance(params, list)
    assert len(params) == 4
    assert isinstance(params[0], np.ndarray)
    assert np.var(np.stack(params, axis=0), axis=0).mean() < 1e-1
