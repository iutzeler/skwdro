# Author: Claude Sonnet 4.5 (Flo.V. vibe-coding purposeless coverage test)
import pytest
import numpy as np
from skwdro.base.problems import (
    Distribution,
    EmpiricalDistributionWithoutLabels,
    EmpiricalDistributionWithLabels,
)


class TestDistribution:
    def test_init(self):
        dist = Distribution(100, "test_dist")
        assert dist.m == 100
        assert dist.name == "test_dist"
        assert dist._samples is None
        assert dist._samples_x is None
        assert dist._samples_y is None

    def test_samples_getter_without_labels(self):
        dist = Distribution(100, "test")
        dist.with_labels = False
        dist._samples = np.array([1, 2, 3])
        assert dist.samples is not None
        assert np.array_equal(dist.samples, np.array([1, 2, 3]))

    def test_samples_getter_with_labels_raises(self):
        dist = Distribution(100, "test")
        dist.with_labels = True
        with pytest.raises(AttributeError):
            _ = dist.samples

    def test_samples_setter_with_ndarray(self):
        dist = Distribution(100, "test")
        data = np.array([1, 2, 3])
        dist.samples = data
        assert dist._samples is not None
        assert np.array_equal(dist._samples, data)

    def test_samples_setter_with_invalid_type(self):
        dist = Distribution(100, "test")
        with pytest.raises(TypeError):
            dist.samples = [1, 2, 3]

    def test_samples_x_getter_with_labels(self):
        dist = Distribution(100, "test")
        dist.with_labels = True
        dist._samples_x = np.array([1, 2, 3])
        assert dist.samples_x is not None
        assert np.array_equal(dist.samples_x, np.array([1, 2, 3]))

    def test_samples_x_getter_without_labels_raises(self):
        dist = Distribution(100, "test")
        dist.with_labels = False
        with pytest.raises(AttributeError):
            _ = dist.samples_x

    def test_samples_x_setter_with_ndarray(self):
        dist = Distribution(100, "test")
        data = np.array([1, 2, 3])
        dist.samples_x = data
        assert dist._samples_x is not None
        assert np.array_equal(dist._samples_x, data)

    def test_samples_x_setter_with_invalid_type(self):
        dist = Distribution(100, "test")
        with pytest.raises(TypeError):
            dist.samples_x = [1, 2, 3]

    def test_samples_y_getter_with_labels(self):
        dist = Distribution(100, "test")
        dist.with_labels = True
        dist._samples_y = np.array([1, 2, 3])
        assert dist.samples_y is not None
        assert np.array_equal(dist.samples_y, np.array([1, 2, 3]))

    def test_samples_y_getter_without_labels_raises(self):
        dist = Distribution(100, "test")
        dist.with_labels = False
        with pytest.raises(AttributeError):
            _ = dist.samples_y

    def test_samples_y_setter_with_ndarray(self):
        dist = Distribution(100, "test")
        labels = np.array([1, 2, 3])
        dist.samples_y = labels
        assert dist._samples_y is not None
        assert np.array_equal(dist._samples_y, labels)

    def test_samples_y_setter_with_invalid_type(self):
        dist = Distribution(100, "test")
        with pytest.raises(TypeError):
            dist.samples_y = [1, 2, 3]


class TestEmpiricalDistributionWithoutLabels:
    def test_init(self):
        samples = np.array([1, 2, 3])
        dist = EmpiricalDistributionWithoutLabels(100, samples)
        assert dist.m == 100
        assert dist.name == "Empirical distribution"
        assert dist.empirical is True
        assert dist.with_labels is False
        assert dist._samples is not None
        assert np.array_equal(dist._samples, samples)

    def test_init_with_custom_name(self):
        samples = np.array([1, 2, 3])
        dist = EmpiricalDistributionWithoutLabels(100, samples, "custom")
        assert dist.name == "custom"


class TestEmpiricalDistributionWithLabels:
    def test_init(self):
        samples_x = np.array([1, 2, 3])
        samples_y = np.array([4, 5, 6])
        dist = EmpiricalDistributionWithLabels(100, samples_x, samples_y)
        assert dist.m == 100
        assert dist.name == "Empirical distribution"
        assert dist.empirical is True
        assert dist.with_labels is True
        assert dist._samples_x is not None and dist._samples_y is not None
        assert np.array_equal(dist._samples_x, samples_x)
        assert np.array_equal(dist._samples_y, samples_y)

    def test_init_with_custom_name(self):
        samples_x = np.array([1, 2, 3])
        samples_y = np.array([4, 5, 6])
        dist = EmpiricalDistributionWithLabels(100, samples_x, samples_y, "custom")
        assert dist.name == "custom"

    def test_init_copies_arrays(self):
        samples_x = np.array([1, 2, 3])
        samples_y = np.array([4, 5, 6])
        dist = EmpiricalDistributionWithLabels(100, samples_x, samples_y)
        samples_x[0] = 999
        samples_y[0] = 999
        assert dist._samples_x is not None and dist._samples_y is not None
        assert dist._samples_x[0] == 1
        assert dist._samples_y[0] == 4
