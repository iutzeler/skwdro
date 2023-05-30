import torch as pt

import skwdro.base.samplers.torch as smplr

from utils import generate_blob

def test_size_samples_classif():
    xi, xi_labels = generate_blob()
    samples_y: pt.Tensor = pt.ones_like(xi_labels)
    for cases in (
            (smplr.ClassificationNormalNormalSampler, {"l_sigma": .1}),
            (smplr.ClassificationNormalBernouilliSampler, {"p": .1})
            ):
        TestedClass, kw_args = cases
        sampler = TestedClass(xi, xi_labels.unsqueeze(-1), sigma=.1, **kw_args)
        samples_x, samples_y = sampler.sample(20)
        assert samples_x.size() == pt.Size((20, 10, 2))
        assert samples_y.size() == pt.Size((20, 10, 1))
    assert (samples_y.abs() == pt.ones_like(samples_y)).all()

def test_grads_sampler():
    xi, xi_labels = generate_blob(rg=True)
    s = pt.tensor([.1], requires_grad=True)
    sampler = smplr.ClassificationNormalNormalSampler(xi, xi_labels.unsqueeze(-1), l_sigma=s, sigma=s)
    samples_x, samples_y = sampler.sample(20)
    samples_x.backward(pt.ones_like(samples_x))
    samples_y.backward(pt.ones_like(samples_y))
    assert xi.grad is not None
    assert xi_labels.grad is not None
