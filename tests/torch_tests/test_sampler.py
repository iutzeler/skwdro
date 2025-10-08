import pytest
import torch as pt

import skwdro.base.samplers.torch as smplr
import skwdro.base.costs_torch as costs
import skwdro.distributions as dst

from tests.torch_tests.utils import generate_blob

DUMMY_COST = costs.NormLabelCost(2, 2, 1e2)

SEED = 666

SEED = 666


class ToyNormalSampler(smplr.NoLabelsSampler, smplr.IsOptionalCovarianceSampler):
    def __init__(self, d: int, **kwargs):
        covar = self.init_covar(d, **kwargs)
        super().__init__(
            dst.MultivariateNormal( 
                loc=pt.zeros((d,)),
                **covar  # type: ignore 
            ),
            None
        )

    def reset_mean(self, xi, xi_labels):
        assert xi_labels is None
        self.data_s = dst.MultivariateNormal( 
            loc=pt.zeros_like(xi),
            **covar  # type: ignore 
        )


@pytest.mark.parametrize("d", [1, 3])
def test_covars_handler(d: int):
    ToyNormalSampler(d, sigma=pt.tensor(1.))
    ToyNormalSampler(d, cov=pt.eye(d))
    ToyNormalSampler(d, prec=pt.eye(d))
    with pytest.raises(ValueError) as e_info:
        ToyNormalSampler(d)

def assert_shapes(sampler):
    test_labs_values = isinstance(sampler, smplr.ClassificationNormalBernouilliSampler)
    samples_x, samples_y = sampler.sample(20)
    assert samples_x.size() == pt.Size((20, 10, 2))
    assert samples_y.size() == pt.Size((20, 10, 1))
    if test_labs_values:
        assert (samples_y.abs() == pt.ones_like(samples_y)).all()
    sampler.reset_mean(samples_x[0, ...], samples_y[0, ...])

def assert_grad(sampler, xi, xi_labels, s):
    samples_x, samples_y = sampler.sample(20)
    samples_x.backward(pt.ones_like(samples_x))
    assert xi.grad is not None
    if sampler.produces_labels:
        samples_y.backward(pt.ones_like(samples_y))
        assert xi_labels.grad is not None
    assert s.grad is not None

def assert_probs(sampler, xi, xi_labels):
    for _, (z, zl) in zip(range(3), sampler):
        assert z.size() == pt.Size((1, 10, 2))
        if sampler.produces_labels:
            assert zl.size() == pt.Size((1, 10, 1))
        assert sampler.log_prob(z, zl).size() == pt.Size((1, 10, 1))
        assert sampler.log_prob_recentered(
            xi,
            xi_labels if sampler.produces_labels else None,
            z, zl
        ).size() == pt.Size((1, 10, 1))
    sampler.reset_mean(xi, xi_labels.unsqueeze(-1))


def test_size_samples_classif():
    xi, xi_labels = generate_blob()
    with pytest.raises(ValueError) as e_err:
        # Twisted case violating the ttype "constraints"
        smplr.LabeledCostSampler(
            costs.NormCost(2, 2),
            xi, None,  # type: ignore
            .1,
            seed=SEED
        )
    assert "Please choose a cost that can sample labels" in str(e_err.value)

    list(map(assert_shapes, [
        smplr.ClassificationNormalNormalSampler(
            xi, xi_labels.unsqueeze(-1),
            seed=SEED, l_sigma=.1, sigma=.1
        ),
        smplr.ClassificationNormalBernouilliSampler(
            .1,
            xi, xi_labels.unsqueeze(-1),
            seed=SEED, sigma=.1
        ),
        smplr.ClassificationNormalIdSampler(
            xi, xi_labels.unsqueeze(-1),
            seed=SEED, sigma=.1
        ),
        smplr.LabeledCostSampler(
            DUMMY_COST,
            xi, xi_labels.unsqueeze(-1),
            .1,
            seed=SEED
        )
    ]))

def test_sampling_prob():
    xi, xi_labels = generate_blob()
    s = pt.tensor([.1])
    for sampler in [
        smplr.ClassificationNormalNormalSampler(
            xi, xi_labels.unsqueeze(-1),
            seed=SEED, l_sigma=s, sigma=s
        ),
        smplr.LabeledCostSampler(
            DUMMY_COST, xi, xi_labels.unsqueeze(-1),
            s, seed=SEED
        ),
        smplr.NoLabelsCostSampler(DUMMY_COST, xi, s, seed=SEED)
    ]:
        assert_probs(sampler, xi, xi_labels)

def test_grads_sampler():
    xi, xi_labels = generate_blob(rg=True)
    s = pt.tensor([.1], requires_grad=True)
    for sampler in [
        smplr.ClassificationNormalNormalSampler(
            xi, xi_labels.unsqueeze(-1),
            seed=SEED, l_sigma=s, sigma=s
        ),
        smplr.LabeledCostSampler(
            DUMMY_COST, xi, xi_labels.unsqueeze(-1),
            s, seed=SEED
        ),
        smplr.NoLabelsCostSampler(DUMMY_COST, xi, s, seed=SEED)
    ]:
        assert_grad(sampler, xi, xi_labels, s)

if __name__ == "__main__": test_size_samples_classif()
