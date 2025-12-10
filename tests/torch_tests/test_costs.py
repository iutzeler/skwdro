# Author: Claude Sonnet 4.5 (Flo.V. vibe-coding purposeless coverage test)

import pytest
import torch as pt
from unittest.mock import Mock, patch
import skwdro.distributions
from skwdro.base.costs_torch import (
    TorchCost,
    NormCost,
    NormLabelCost,
)
from skwdro.base.costs_torch.base_cost import ENGINES_NAMES
import skwdro.distributions as dst


@pytest.fixture
def mock_distribution():
    dist = skwdro.distributions.Normal(pt.randn(10, 5), pt.tensor(0.1))
    return dist


class ConcreteTorchCost(TorchCost):
    def value(self, xi, zeta, xi_labels=None, zeta_labels=None):
        return pt.norm(xi - zeta, p=2, dim=-1, keepdim=True)

    def _sampler_data(self, xi, epsilon):
        return mock_distribution()

    def _sampler_labels(self, xi_labels, epsilon):
        return mock_distribution() if xi_labels is not None else None

    def solve_max_series_exp(self, xi, xi_labels, rhs, rhs_labels):
        return xi, xi_labels


class TestTorchCostBase:
    def test_init(self):
        cost = ConcreteTorchCost(name="test_cost", engine="pt")
        assert cost.name == "test_cost"
        assert cost.engine == "pt"
        assert cost.power == 1.0

    def test_forward_calls_value(self):
        cost = ConcreteTorchCost()
        xi = pt.randn(5, 3)
        zeta = pt.randn(5, 3)
        result = cost(xi, zeta)
        assert isinstance(result, pt.Tensor)

    def test_str_representation(self):
        cost = ConcreteTorchCost(name="test", engine="pt")
        str_repr = str(cost)
        assert "test" in str_repr
        assert "PyTorch tensors" in str_repr

    @pytest.mark.parametrize("engine,expected_name", [
        ("pt", "PyTorch tensors"),
        ("jx", "Jax arrays"),
    ])
    def test_engines_names_mapping(self, engine, expected_name):
        assert ENGINES_NAMES[engine] == expected_name


class TestNormCost:
    @pytest.mark.parametrize("p,power", [
        (1.0, 1.0),
        (2.0, 1.0),
        (2.0, 2.0),
        (1.0, 2.0),
    ])
    def test_init(self, p, power):
        cost = NormCost(p=p, power=power)
        assert cost.p == p
        assert cost.power == power
        assert cost.engine == "pt"

    def test_init_custom_name(self):
        cost = NormCost(name="CustomNorm")
        assert cost.name == "CustomNorm"

    def test_init_default_name(self):
        cost = NormCost()
        assert cost.name == "Norm"

    @pytest.mark.parametrize("p,power,xi_shape", [
        (1.0, 1.0, (10, 5)),
        (2.0, 1.0, (8, 3)),
        (2.0, 2.0, (5, 4)),
    ])
    def test_value_without_labels(self, p, power, xi_shape):
        cost = NormCost(p=p, power=power)
        xi = pt.randn(xi_shape)
        zeta = pt.randn(xi_shape)
        result = cost.value(xi, zeta)
        assert result.shape == (xi_shape[0], 1)

    def test_value_computation_p2_power1(self):
        cost = NormCost(p=2.0, power=1.0)
        xi = pt.tensor([[1.0, 2.0], [3.0, 4.0]])
        zeta = pt.tensor([[1.0, 2.0], [0.0, 0.0]])
        result = cost.value(xi, zeta)
        expected = pt.tensor([[0.0], [5.0]])
        assert pt.allclose(result, expected, atol=1e-5)

    def test_value_computation_p1_power1(self):
        cost = NormCost(p=1.0, power=1.0)
        xi = pt.tensor([[1.0, 2.0], [3.0, 4.0]])
        zeta = pt.tensor([[1.0, 2.0], [0.0, 0.0]])
        result = cost.value(xi, zeta)
        expected = pt.tensor([[0.0], [7.0]])
        assert pt.allclose(result, expected, atol=1e-5)

    @pytest.mark.parametrize("p,power", [
        (1.0, 1.0),
        (2.0, 1.0),
        (2.0, 2.0),
    ])
    def test_sampler_data_power1(self, p, power):
        cost = NormCost(p=p, power=power)
        xi = pt.randn(10, 5)
        epsilon = pt.tensor(0.1)
        sampler = cost._sampler_data(xi, epsilon)
        assert sampler is not None

    def test_sampler_data_none_epsilon(self):
        cost = NormCost(p=2.0, power=1.0)
        xi = pt.randn(10, 5)
        sampler = cost._sampler_data(xi, None)
        assert sampler is not None

    def test_sampler_data_scalar_epsilon(self):
        cost = NormCost(p=2.0, power=1.0)
        xi = pt.randn(10, 5)
        sampler = cost._sampler_data(xi, pt.tensor(0.1))
        assert sampler is not None

    @pytest.mark.parametrize("p,power", [
        (1.0, 1.0),
        (2.0, 1.0),
        (2.0, 2.0),
        (float("inf"), 2.0),
        (float("inf"), 1.0),
    ])
    def test_sampler_data_p_power(self, p, power):
        cost = NormCost(p=p, power=power)
        xi = pt.randn(10, 5)
        epsilon = pt.tensor(0.1)
        if p == float('inf') and power == 2.0:
            with pytest.raises(NotImplementedError):
                sampler = cost._sampler_data(xi, epsilon)
        else:
            sampler = cost._sampler_data(xi, epsilon)
            assert sampler is not None

    def test_sampler_labels_with_labels(self):
        cost = NormCost()
        xi_labels = pt.randn(10, 1)
        epsilon = pt.tensor(0.1)
        sampler = cost._sampler_labels(xi_labels, epsilon)
        assert isinstance(sampler, dst.Dirac)

    def test_sampler_labels_without_labels(self):
        cost = NormCost()
        epsilon = pt.tensor(0.1)
        sampler = cost._sampler_labels(None, epsilon)
        assert sampler is None

    @pytest.mark.parametrize("p,power", [
        (2.0, 2.0),
    ])
    def test_solve_max_series_exp_with_labels(self, p, power):
        cost = NormCost(p=p, power=power)
        xi = pt.randn(10, 5)
        xi_labels = pt.randn(10, 1)
        rhs = pt.randn(10, 5)
        rhs_labels = pt.randn(10, 1)
        zeta, zeta_labels = cost.solve_max_series_exp(xi, xi_labels, rhs, rhs_labels)
        assert pt.allclose(zeta, xi + 0.5 * rhs)
        assert pt.allclose(zeta_labels, xi_labels)

    def test_solve_max_series_exp_without_labels(self):
        cost = NormCost(p=2.0, power=2.0)
        xi = pt.randn(10, 5)
        rhs = pt.randn(10, 5)
        zeta, zeta_labels = cost.solve_max_series_exp(xi, None, rhs, None)
        assert pt.allclose(zeta, xi + 0.5 * rhs)
        assert zeta_labels is None


class TestNormLabelCost:
    @pytest.mark.parametrize("p,power,kappa", [
        (2.0, 1.0, 1e4),
        (2.0, 2.0, 1e3),
        (1.0, 1.0, 100.0),
    ])
    def test_init(self, p, power, kappa):
        cost = NormLabelCost(p=p, power=power, kappa=kappa)
        assert cost.p == p
        assert cost.power == power
        assert cost.kappa == kappa

    def test_init_custom_name(self):
        cost = NormLabelCost(name="CustomLabelCost")
        assert cost.name == "CustomLabelCost"

    def test_init_default_name(self):
        cost = NormLabelCost()
        assert cost.name == "Kappa-norm"

    def test_init_negative_kappa_raises(self):
        with pytest.raises(AssertionError):
            NormLabelCost(kappa=-1.0)

    @pytest.mark.parametrize("p,power,kappa,xi_shape", [
        (2.0, 1.0, 1e4, (10, 5)),
        (2.0, 2.0, 1e3, (8, 3)),
        (1.0, 1.0, 100.0, (5, 4)),
    ])
    def test_value_with_labels(self, p, power, kappa, xi_shape):
        cost = NormLabelCost(p=p, power=power, kappa=kappa)
        xi = pt.randn(xi_shape)
        zeta = pt.randn(xi_shape)
        xi_labels = pt.randn(xi_shape[0], 1)
        zeta_labels = pt.randn(xi_shape[0], 1)
        result = cost.value(xi, zeta, xi_labels, zeta_labels)
        assert result.shape == (xi_shape[0], 1)

    def test_value_kappa_zero(self):
        cost = NormLabelCost(p=2.0, power=1.0, kappa=0.0)
        xi = pt.randn(5, 3)
        zeta = pt.randn(5, 3)
        xi_labels = pt.randn(5, 1)
        zeta_labels = pt.randn(5, 1)
        result = cost.value(xi, zeta, xi_labels, zeta_labels)
        assert result.shape == (5, 1)

    def test_value_kappa_inf(self):
        cost = NormLabelCost(p=2.0, power=1.0, kappa=float('inf'))
        xi = pt.randn(5, 3)
        zeta = pt.randn(5, 3)
        xi_labels = pt.randn(5, 1)
        zeta_labels = pt.randn(5, 1)
        result = cost.value(xi, zeta, xi_labels, zeta_labels)
        assert result.shape == (5, 1)

    def test_label_penalty(self):
        y = pt.tensor([[1.0, 2.0], [3.0, 4.0]])
        y_prime = pt.tensor([[1.0, 2.0], [0.0, 0.0]])
        result = NormLabelCost._label_penalty(y, y_prime, p=2.0)
        expected = pt.tensor([[0.0], [5.0]])
        assert pt.allclose(result, expected, atol=1e-5)

    def test_data_penalty(self):
        x = pt.tensor([[1.0, 2.0], [3.0, 4.0]])
        x_prime = pt.tensor([[1.0, 2.0], [0.0, 0.0]])
        result = NormLabelCost._data_penalty(x, x_prime, p=2.0)
        expected = pt.tensor([[0.0], [5.0]])
        assert pt.allclose(result, expected, atol=1e-5)

    @pytest.mark.parametrize("p,power,kappa", [
        (2.0, 1.0, 1e4),
        (1.0, 1.0, 100.0),
        (float('inf'), 1.0, 100.0),
    ])
    def test_sampler_labels_with_labels(self, p, power, kappa):
        cost = NormLabelCost(p=p, power=power, kappa=kappa)
        xi_labels = pt.randn(10, 1)
        epsilon = pt.tensor(0.1)
        sampler = cost._sampler_labels(xi_labels, epsilon)
        assert sampler is not None

    def test_sampler_labels_kappa_inf(self):
        cost = NormLabelCost(p=2.0, power=1.0, kappa=float('inf'))
        xi_labels = pt.randn(10, 1)
        epsilon = pt.tensor(0.1)
        sampler = cost._sampler_labels(xi_labels, epsilon)
        assert isinstance(sampler, dst.Dirac)

    def test_sampler_labels_none_epsilon(self):
        cost = NormLabelCost(p=2.0, power=1.0, kappa=1e4)
        xi_labels = pt.randn(10, 1)
        sampler = cost._sampler_labels(xi_labels, None)
        assert sampler is not None

    @pytest.mark.parametrize("p,power,kappa", [
        (2.0, 1.0, 1e4),
        (2.0, 2.0, 1e3),
    ])
    def test_solve_max_series_exp_with_labels(self, p, power, kappa):
        cost = NormLabelCost(p=p, power=power, kappa=kappa)
        xi = pt.randn(10, 5)
        xi_labels = pt.randn(10, 1)
        rhs = pt.randn(10, 5)
        rhs_labels = pt.randn(10, 1)
        if p == power == 2.:
            zeta, zeta_labels = cost.solve_max_series_exp(xi, xi_labels, rhs, rhs_labels)
            assert pt.allclose(zeta, xi + 0.5 * rhs)
            assert pt.allclose(zeta_labels, xi_labels + 0.5 * rhs_labels / kappa)
        else:
            with pytest.raises(NotImplementedError):
                zeta, zeta_labels = cost.solve_max_series_exp(xi, xi_labels, rhs, rhs_labels)

    def test_solve_max_series_exp_without_labels(self):
        cost = NormLabelCost(p=2.0, power=2.0, kappa=1e4)
        xi = pt.randn(10, 5)
        rhs = pt.randn(10, 5)
        zeta, zeta_labels = cost.solve_max_series_exp(xi, None, rhs, None)
        assert pt.allclose(zeta, xi + 0.5 * rhs)
        assert zeta_labels is None


class TestCostSampler:
    def test_sampler_returns_tuple(self):
        cost = NormCost(p=2.0, power=1.0)
        xi = pt.randn(10, 5)
        xi_labels = None
        epsilon = pt.tensor(0.1)
        result = cost.sampler(xi, xi_labels, epsilon)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.parametrize("p,power", [
        (1.0, 1.0),
        (2.0, 1.0),
        (2.0, 2.0),
    ])
    def test_sampler_data_component(self, p, power):
        cost = NormCost(p=p, power=power)
        xi = pt.randn(10, 5)
        xi_labels = None
        epsilon = pt.tensor(0.1)
        data_dist, _ = cost.sampler(xi, xi_labels, epsilon)
        assert data_dist is not None
