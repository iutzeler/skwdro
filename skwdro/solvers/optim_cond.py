import numpy as np

class OptCond:
    def __init__(self, order, tol_theta: float=1e-4, tol_lambda: float=1e-6, max_iter: int=int(1e5), mode: str="both"):
        assert order > 0
        self.p = order
        self.tol_t = tol_theta
        self.tol_l = tol_lambda
        self.maxiter = max_iter

        self.mode = True
        if mode in {"both", "theta_and_lambda", "t&l"}:
            self.mode = True
        elif mode in {"one", "theta_or_lambda", "tUl"}:
            self.mode = False
        else:
            raise ValueError("Please provide a valid string value for parameter 'mode' in OptCond: '" + str(mode) + "' is invalid.")

    def __call__(self, grads, it: int) -> bool:
        grad_theta, grad_lambda = grads
        grad_t_stop = float(np.linalg.norm(grad_theta, ord=self.p)) < self.tol_t and self.tol_t  > 0
        grad_l_stop = float(grad_lambda) < self.tol_l and self.tol_l  > 0
        it_stop = it >= self.maxiter and self.maxiter > 0
        if it_stop: # Warn the user that the optimality conditions may not be verified (loss of accuracy)
            Warning("Maximum number of iterations reached") 
        return self.grad_stop(grad_t_stop, grad_l_stop) or it_stop

    def grad_stop(self, grad_t_stop: bool, grad_l_stop: bool) -> bool:
        if self.mode:
            return grad_t_stop and grad_l_stop
        else:
            return grad_t_stop or grad_l_stop
