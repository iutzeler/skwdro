import numpy as np
import multiprocessing as mp

from skwdro.linear_models import LogisticRegression


from tests.torch_tests.test_binary_separable_log_classif import generate_points

def launch_solver(fi: bool=False):
    estimator = LogisticRegression(
            rho=1e-2,
            l2_reg=.0,
            cost="t-NLC-2-2",
            fit_intercept=fi,
            solver="entropic_torch_pre"
            )
    X, y = generate_points()
    estimator.fit(X, y)
    # print(X,y)
    # print(estimator.coef_)
    # print(estimator.predict(X))
    assert estimator.score(X, y) > .5


def test_autostop():
    """
    On a separable problem w/ convex loss, the algo should converge in finite time.
    With two points, this time should be small.
    """
    p = mp.Process(target=launch_solver)
    p.start()
    p.join(90)
    if p.is_alive():
        p.kill()
        p.join()
        raise TimeoutError("Algorithm took more than 1'30'' to run on simple case, we consider that it failed to converge")
