from collections.abc import Iterable
from functools import wraps


class SolverResult(Iterable):
    def __init__(self, coef=None, intercept=None, dual_var=None, robust_loss=None, _iter_attrs=['coef', 'intercept', 'dual_var'], **kwargs):
        self.coef = coef
        self.intercept = intercept
        self.dual_var = dual_var
        self.robust_loss = robust_loss
        self._iter_attrs = _iter_attrs

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __iter__(self):
        """
            Iterator for legacy behaviour
        """
        return iter((getattr(self, attr) for attr in self._iter_attrs))


def wrap_solver_result(solver_func):
    """
        Decorator to wrap the return of a legacy solver
    """
    @wraps(solver_func)  # This is a convenience function for invoking update_wrapper() as a function decorator when defining a wrapper function. It enables the doc to go through.
    def wrapper(*args, **kwargs):
        legacy_res = solver_func(*args, **kwargs)
        if isinstance(legacy_res, SolverResult):
            return legacy_res
        elif isinstance(legacy_res, tuple):
            assert len(legacy_res) >= 1
            assert len(legacy_res) <= 4
            d = {}
            d['coef'] = legacy_res[0]
            if len(legacy_res) == 2:
                d['dual_var'] = legacy_res[1]
            if len(legacy_res) >= 3:
                d['intercept'] = legacy_res[1]
                d['dual_var'] = legacy_res[2]
            if len(legacy_res) == 4:
                d['robust_loss'] = legacy_res[3]
            return SolverResult(**d, _iter_attrs=d.keys())
        elif isinstance(legacy_res, dict):
            return SolverResult(**legacy_res, _iter_attrs=legacy_res.keys())
        raise RuntimeError("Invalid return from solver")
    return wrapper
