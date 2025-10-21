# Author: Claude Sonnet 4.5 (Flo.V. vibe-coding purposeless coverage test)

import pytest
import torch as pt

from skwdro.solvers.oracle_torch import DualLoss
from skwdro.base.losses_torch.portfolio import SimplePortfolio
from skwdro.base.losses_torch.weber import SimpleWeber
from skwdro.base.losses_torch.logistic import BiDiffSoftMarginLoss
from skwdro.base.losses_torch import (
    Loss,
    LogisticLoss,
    NewsVendorLoss_torch,
    QuadraticLoss,
)
from skwdro.base.costs_torch import NormCost
from skwdro.base.samplers.torch import LabeledCostSampler, NoLabelsCostSampler


MOCK_COST = NormCost(2, 2)

@pytest.fixture
def mock_labeled_sampler():
    sampler = LabeledCostSampler(MOCK_COST, pt.randn(10, 5), pt.randn(10, 1), .1)
    return sampler


@pytest.fixture
def mock_no_labels_sampler():
    sampler = NoLabelsCostSampler(MOCK_COST, pt.randn(10, 5), .1)
    return sampler


class ConcreteLoss(Loss):
    def __init__(self, sampler, has_labels, **kwargs):
        super().__init__(sampler, has_labels, **kwargs)
        self._theta = pt.nn.Parameter(pt.randn(5, 1))
        self._intercept = pt.randn(1)

    @classmethod
    def default_sampler(cls, xi, xi_labels, epsilon, seed):
        return LabeledCostSampler(MOCK_COST, pt.randn(10, 5), pt.randn(10, 1), .1)

    @property
    def theta(self):
        return self._theta

    @property
    def intercept(self):
        return self._intercept


class TestLoss:
    def test_init_with_sampler(self, mock_labeled_sampler):
        loss = ConcreteLoss(mock_labeled_sampler, True)
        assert loss._sampler is mock_labeled_sampler
        assert loss.has_labels is True
        assert loss.l2reg is None

    def test_init_without_sampler_with_xi(self):
        xi = pt.randn(10, 5)
        loss = ConcreteLoss(None, False, xi=xi, sigma=0.2)
        assert loss._sampler is not None
        assert loss.has_labels is False

    def test_init_without_sampler_without_xi(self):
        loss = ConcreteLoss(None, False)
        assert loss._sampler is None

    def test_l2reg_positive(self, mock_labeled_sampler):
        loss = ConcreteLoss(mock_labeled_sampler, True, l2reg=0.5)
        assert loss.l2reg is not None
        assert loss.l2reg.item() == 0.5

    def test_l2reg_negative(self, mock_labeled_sampler):
        loss = ConcreteLoss(mock_labeled_sampler, True, l2reg=-0.5)
        assert loss.l2reg is None

    def test_l2reg_zero(self, mock_labeled_sampler):
        loss = ConcreteLoss(mock_labeled_sampler, True, l2reg=0.0)
        assert loss.l2reg is None

    def test_regularize_without_l2reg(self, mock_labeled_sampler):
        loss = ConcreteLoss(mock_labeled_sampler, True)
        input_loss = pt.tensor(5.0)
        result = loss.regularize(input_loss)
        assert result == input_loss

    def test_regularize_with_l2reg(self, mock_labeled_sampler):
        loss = ConcreteLoss(mock_labeled_sampler, True, l2reg=0.1)
        input_loss = pt.tensor(5.0)
        result = loss.regularize(input_loss)
        assert result > input_loss

    def test_value_old_raises(self, mock_labeled_sampler):
        loss = ConcreteLoss(mock_labeled_sampler, True)
        with pytest.raises(NotImplementedError):
            loss.value_old(pt.randn(5), pt.randn(10, 5))

    def test_value_raises(self, mock_labeled_sampler):
        loss = ConcreteLoss(mock_labeled_sampler, True)
        with pytest.raises(NotImplementedError):
            loss.value(pt.randn(10, 5), pt.randn(10, 1))

    def test_sample_pi0(self, mock_labeled_sampler):
        loss = ConcreteLoss(mock_labeled_sampler, True)
        samples, labels = loss.sample_pi0(10)
        assert samples.shape == pt.Size((10, 10, 5))
        assert labels is not None
        assert labels.shape == pt.Size((10, 10, 1))

    def test_sampler_getter(self, mock_labeled_sampler):
        loss = ConcreteLoss(mock_labeled_sampler, True)
        assert loss.sampler is mock_labeled_sampler

    def test_sampler_getter_raises(self):
        loss = ConcreteLoss(None, True)
        with pytest.raises(ValueError):
            _ = loss.sampler

    def test_sampler_setter(self, mock_labeled_sampler):
        loss = ConcreteLoss(None, True)
        loss.sampler = mock_labeled_sampler
        assert loss._sampler is mock_labeled_sampler

    def test_sampler_deleter(self, mock_labeled_sampler):
        loss = ConcreteLoss(mock_labeled_sampler, True)
        del loss.sampler
        with pytest.raises(AttributeError):
            _ = loss._sampler

    def test_forward(self, mock_labeled_sampler):
        loss = ConcreteLoss(mock_labeled_sampler, True)
        with pytest.raises(NotImplementedError):
            loss.forward(pt.randn(10, 5), pt.randn(10, 1))

    def test_wrapped(self):
        lss = ConcreteLoss(mock_labeled_sampler, True)
        input_tensor = pt.randn(10, 1)
        target = pt.randn(10, 1)
        epsilon = pt.tensor(0.1)
        model = DualLoss(lss, MOCK_COST, 1, epsilon, pt.tensor(0.1))
        assert model.default_sampler(input_tensor, target, epsilon, None) is not None


class TestBiDiffSoftMarginLoss:
    def test_init(self):
        loss = BiDiffSoftMarginLoss()
        assert loss.reduction == 'none'

    def test_forward(self):
        loss = BiDiffSoftMarginLoss()
        input_tensor = pt.randn(10, 1)
        target = pt.randn(10, 1)
        result = loss.forward(input_tensor, target)
        assert result.shape == input_tensor.shape

    def test_forward_shape_mismatch(self):
        loss = BiDiffSoftMarginLoss()
        input_tensor = pt.randn(10, 1)
        target = pt.randn(10)
        result = loss.forward(input_tensor, target)
        assert result.shape == input_tensor.shape


class TestLogisticLoss:
    @pytest.mark.parametrize("d,fit_intercept", [
        (5, True),
        (10, False),
        (3, True),
    ])
    def test_init(self, mock_labeled_sampler, d, fit_intercept):
        loss = LogisticLoss(mock_labeled_sampler, d=d, fit_intercept=fit_intercept)
        assert loss.linear.in_features == d
        assert loss.linear.out_features == 1
        assert loss.linear.bias is not None if fit_intercept else loss.linear.bias is None
        assert loss.l2reg is None

    def test_init_with_l2reg(self, mock_labeled_sampler):
        loss = LogisticLoss(mock_labeled_sampler, d=5, l2reg=0.5)
        assert loss.l2reg is not None
        assert loss.l2reg.item() == 0.5

    def test_init_invalid_dimension(self, mock_labeled_sampler):
        with pytest.raises(AssertionError):
            LogisticLoss(mock_labeled_sampler, d=0)

    def test_predict(self, mock_labeled_sampler):
        loss = LogisticLoss(mock_labeled_sampler, d=5)
        X = pt.randn(10, 5)
        predictions = loss.predict(X)
        assert predictions.shape == (10, 1)
        assert pt.all(predictions >= -1) and pt.all(predictions <= 1)

    def test_value(self, mock_labeled_sampler):
        loss = LogisticLoss(mock_labeled_sampler, d=5)
        xi = pt.randn(10, 5)
        xi_labels = pt.randn(10, 1)
        result = loss.value(xi, xi_labels)
        assert result.shape == (10, 1)

    def test_value_with_none_labels_raises(self, mock_labeled_sampler):
        loss = LogisticLoss(mock_labeled_sampler, d=5)
        xi = pt.randn(10, 5)
        with pytest.raises(AssertionError):
            loss.value(xi, None)

    def test_default_sampler(self):
        xi = pt.randn(10, 5)
        xi_labels = pt.randn(10, 1)
        sampler = LogisticLoss.default_sampler(xi, xi_labels, 0.1, 42)
        assert sampler is not None

    def test_theta_property(self, mock_labeled_sampler):
        loss = LogisticLoss(mock_labeled_sampler, d=5)
        assert loss.theta.shape == (1, 5)

    def test_intercept_property(self, mock_labeled_sampler):
        loss = LogisticLoss(mock_labeled_sampler, d=5, fit_intercept=True)
        assert loss.intercept.shape == (1,)


class TestNewsVendorLoss:
    @pytest.mark.parametrize("k,u,l2reg", [
        (5.0, 7.0, None),
        (10.0, 15.0, 0.1),
        (3.0, 5.0, 0.5),
    ])
    def test_init(self, mock_no_labels_sampler, k, u, l2reg):
        loss = NewsVendorLoss_torch(mock_no_labels_sampler, k=k, u=u, l2reg=l2reg)
        assert loss.k.item() == k
        assert loss.u.item() == u
        assert loss.name == "NewsVendor loss"

    def test_init_custom_name(self, mock_no_labels_sampler):
        loss = NewsVendorLoss_torch(mock_no_labels_sampler, name="Custom")
        assert loss.name == "Custom"

    def test_value_old(self, mock_no_labels_sampler):
        loss = NewsVendorLoss_torch(mock_no_labels_sampler)
        theta = pt.tensor(10.0)
        xi = pt.tensor(8.0)
        result = loss.value_old(theta, xi)
        expected = loss.k * theta - loss.u * pt.minimum(theta, xi)
        assert pt.isclose(result, expected)

    def test_value(self, mock_no_labels_sampler):
        loss = NewsVendorLoss_torch(mock_no_labels_sampler)
        xi = pt.randn(10, 2)
        xi_labels = pt.randn(10, 1)
        result = loss.value(xi, xi_labels)
        assert result.shape == (10, 1)

    def test_theta_property(self, mock_no_labels_sampler):
        loss = NewsVendorLoss_torch(mock_no_labels_sampler)
        assert loss.theta.shape == (1,)

    def test_intercept_property(self, mock_no_labels_sampler):
        loss = NewsVendorLoss_torch(mock_no_labels_sampler)
        assert loss.intercept is None

    def test_default_sampler(self):
        xi = pt.randn(10, 1)
        sampler = NewsVendorLoss_torch.default_sampler(xi, None, 0.1, 42)
        assert sampler is not None


class TestPortfolio:
    THRESHOLD_RECAST: float = 1e-8
    @pytest.mark.parametrize("d,risk_aversion,risk_level", [
        (5, 0.5, 0.05),
        (10, 1.0, 0.1),
        (3, 0.2, 0.01),
    ])
    def test_init(self, d, risk_aversion, risk_level):
        portfolio = SimplePortfolio(d, risk_aversion, risk_level)
        assert portfolio.assets.in_features == d
        assert portfolio.assets.out_features == 1
        assert abs(portfolio.eta.item() - risk_aversion) < self.THRESHOLD_RECAST
        assert abs(portfolio.alpha.item() == risk_level) < self.THRESHOLD_RECAST
        assert portfolio.tau.shape == ()

    def test_forward(self):
        portfolio = SimplePortfolio(5, 0.5, 0.05)
        xi = pt.randn(10, 5)
        result = portfolio.forward(xi)
        assert result.shape == (10, 1)

    def test_weights_sum_to_one(self):
        portfolio = SimplePortfolio(5, 0.5, 0.05)
        weights = portfolio.assets.weight
        assert pt.isclose(weights.sum(), pt.tensor(1.0))


class TestQuadraticLoss:
    @pytest.mark.parametrize("d,fit_intercept,l2reg", [
        (5, True, None),
        (10, False, 0.1),
        (3, True, 0.5),
    ])
    def test_init(self, mock_labeled_sampler, d, fit_intercept, l2reg):
        loss = QuadraticLoss(mock_labeled_sampler, d=d, fit_intercept=fit_intercept, l2reg=l2reg)
        assert loss.linear.in_features == d
        assert loss.linear.out_features == 1
        assert loss.linear.bias is not None if fit_intercept else loss.linear.bias is None

    def test_init_invalid_dimension(self, mock_labeled_sampler):
        with pytest.raises(AssertionError):
            QuadraticLoss(mock_labeled_sampler, d=0)

    def test_regression(self, mock_labeled_sampler):
        loss = QuadraticLoss(mock_labeled_sampler, d=5)
        X = pt.randn(10, 5)
        result = loss.regression(X)
        assert result.shape == (10, 1)

    def test_value(self, mock_labeled_sampler):
        loss = QuadraticLoss(mock_labeled_sampler, d=5)
        xi = pt.randn(10, 5)
        xi_labels = pt.randn(10, 1)
        result = loss.value(xi, xi_labels)
        assert result.shape == (10, 1)

    def test_value_with_none_labels_raises(self, mock_labeled_sampler):
        loss = QuadraticLoss(mock_labeled_sampler, d=5)
        xi = pt.randn(10, 5)
        with pytest.raises(AssertionError):
            loss.value(xi, None)

    def test_default_sampler(self):
        xi = pt.randn(10, 5)
        xi_labels = pt.randn(10, 1)
        sampler = QuadraticLoss.default_sampler(xi, xi_labels, 0.1, 42)
        assert sampler is not None

    def test_theta_property(self, mock_labeled_sampler):
        loss = QuadraticLoss(mock_labeled_sampler, d=5)
        assert loss.theta.shape == (1, 5)

    def test_intercept_property(self, mock_labeled_sampler):
        loss = QuadraticLoss(mock_labeled_sampler, d=5, fit_intercept=True)
        assert loss.intercept.shape == (1,)


class TestSimpleWeber:
    @pytest.mark.parametrize("d", [2, 3, 5])
    def test_init(self, d):
        weber = SimpleWeber(d)
        assert weber.pos.shape == (d,)
        assert weber.d == d
        assert pt.all(weber.pos == 0)

    def test_forward(self):
        weber = SimpleWeber(2)
        xi = pt.randn(10, 2)
        xi_labels = pt.randn(10, 1)
        result = weber.forward(xi, xi_labels)
        assert result.shape == (10, 1)
