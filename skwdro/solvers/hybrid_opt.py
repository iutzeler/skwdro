import torch
from torch.optim import SGD, Adam

PRERULES = {}
POSTRULES = {}
BOUND = 10**2


def prerule(name):
    def decorator(func):
        PRERULES[name] = func
        return func
    return decorator


def postrule(name):
    def decorator(func):
        POSTRULES[name] = func
        return func
    return decorator


@prerule('mwu')
def prerule_mwu(p):
    assert (p > 0).all()
    p.log_()


@postrule('mwu')
def postrule_mwu(p):
    p.exp_()


@prerule('mwu_simplex')
def prerule_mwu_simplex(p):
    assert (p > 0).all()
    p.log_()


@postrule('mwu_simplex')
def postrule_mwu_simplex(p):
    p.exp_()
    p /= torch.sum(p)


@postrule('non_neg')
def postrule_non_neg(p):
    p.clip_(0, None)


@prerule('max')
@postrule('max')
def rule_max(p):
    p.neg_()


@prerule('bound')
def prerule_bound(p):
    p.grad.clip_(-BOUND, BOUND)


class HybridOpt(object):
    def __init__(self, params, **kwargs):
        super(HybridOpt, self).__init__(params, **kwargs)

    def _apply_rules(self, rules):
        for group in self.param_groups:
            intersection = group.keys() & rules.keys()
            assert len(intersection) <= 1
            if len(intersection) == 1:
                key = intersection.pop()
                for p in group['params']:
                    with torch.no_grad():
                        rules[key](p)

    def step(self, *args, **kwargs):
        self._apply_rules(PRERULES)

        super(HybridOpt, self).step(*args, **kwargs)

        self._apply_rules(POSTRULES)


class HybridSGD(HybridOpt, SGD):
    def __init__(self, *args, **kwargs):
        super(HybridSGD, self).__init__(*args, **kwargs)


class HybridAdam(HybridOpt, Adam):
    def __init__(self, *args, **kwargs):
        super(HybridAdam, self).__init__(*args, **kwargs)
