import torch as pt

import skwdro.base.samplers.torch as smplr
import skwdro.base.costs_torch as costs

from tests.torch_tests.utils import generate_blob

dummy_cost = costs.NormLabelCost(2, 2, 1e2)

SEED = 666

def assert_shapes(sampler):
    test_labs_values = isinstance(sampler, smplr.ClassificationNormalBernouilliSampler)
    samples_x, samples_y = sampler.sample(20)
    assert samples_x.size() == pt.Size((20, 10, 2))
    assert samples_y.size() == pt.Size((20, 10, 1))
    if test_labs_values:
        assert (samples_y.abs() == pt.ones_like(samples_y)).all()

def assert_grad(sampler, xi, xi_labels, s):
    samples_x, samples_y = sampler.sample(20)
    samples_x.backward(pt.ones_like(samples_x))
    assert xi.grad is not None
    if sampler.produces_labels:
        samples_y.backward(pt.ones_like(samples_y))
        assert xi_labels.grad is not None
    assert s.grad is not None

def test_size_samples_classif():
    xi, xi_labels = generate_blob()
    list(map(assert_shapes, [
            smplr.ClassificationNormalNormalSampler(xi, xi_labels.unsqueeze(-1), SEED, l_sigma=.1, sigma=.1),
            smplr.ClassificationNormalBernouilliSampler(xi, xi_labels.unsqueeze(-1), SEED, p=.1, sigma=.1),
            smplr.LabeledCostSampler(dummy_cost, xi, xi_labels.unsqueeze(-1), .1, SEED)
        ]))

def test_grads_sampler():
    xi, xi_labels = generate_blob(rg=True)
    s = pt.tensor([.1], requires_grad=True)
    for sampler in [
            smplr.ClassificationNormalNormalSampler(xi, xi_labels.unsqueeze(-1), SEED, l_sigma=s, sigma=s),
            smplr.LabeledCostSampler(dummy_cost, xi, xi_labels.unsqueeze(-1), s, SEED),
            smplr.NoLabelsCostSampler(dummy_cost, xi, s, SEED)
            ]:
        assert_grad(sampler, xi, xi_labels, s)

if __name__ == "__main__": test_size_samples_classif()
