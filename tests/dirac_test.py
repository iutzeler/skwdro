import pytest
import torch as pt
from skwdro.distributions import Dirac
from skwdro.distributions.dirac_distribution import cast_to_size


class TestCastToSize:
    @pytest.mark.parametrize("input_shape,expected", [
        (pt.Size([2, 3]), pt.Size([2, 3])),
        ([2, 3], pt.Size([2, 3])),
        ((2, 3), pt.Size([2, 3])),
        (pt.Size([]), pt.Size([])),
        ([], pt.Size([])),
    ])
    def test_cast_to_size(self, input_shape, expected):
        result = cast_to_size(input_shape)
        assert result == expected
        assert isinstance(result, pt.Size)


class TestDiracInit:
    @pytest.mark.parametrize("loc_shape,n_batch_dims", [
        ((5,), 0),
        ((3, 5), 1),
        ((2, 3, 5), 2),
        ((5,), 1),
    ])
    def test_init(self, loc_shape, n_batch_dims):
        loc = pt.randn(loc_shape)
        dirac = Dirac(loc, n_batch_dims=n_batch_dims)
        assert dirac.loc.shape == loc_shape
        expected_batch_shape = loc_shape[:n_batch_dims]
        expected_event_shape = loc_shape[n_batch_dims:]
        assert dirac.batch_shape == expected_batch_shape
        assert dirac.event_shape == expected_event_shape

    def test_init_default_n_batch_dims(self):
        loc = pt.randn(5)
        dirac = Dirac(loc)
        assert dirac.batch_shape == pt.Size([])
        assert dirac.event_shape == pt.Size([5])


class TestDiracProperties:
    @pytest.mark.parametrize("loc_shape,n_batch_dims", [
        ((5,), 0),
        ((3, 5), 1),
        ((2, 3, 4), 2),
    ])
    def test_arg_constraints(self, loc_shape, n_batch_dims):
        loc = pt.randn(loc_shape)
        dirac = Dirac(loc, n_batch_dims=n_batch_dims)
        constraints = dirac.arg_constraints
        assert "loc" in constraints

    def test_support(self):
        loc = pt.randn(5)
        dirac = Dirac(loc)
        assert dirac.support is not None

    def test_has_rsample(self):
        loc = pt.randn(5)
        dirac = Dirac(loc)
        assert dirac.has_rsample is True

    @pytest.mark.parametrize("loc_shape,n_batch_dims", [
        ((5,), 0),
        ((3, 5), 1),
        ((2, 3, 4), 2),
    ])
    def test_mean(self, loc_shape, n_batch_dims):
        loc = pt.randn(loc_shape)
        dirac = Dirac(loc, n_batch_dims=n_batch_dims)
        mean = dirac.mean
        assert pt.allclose(mean, loc)

    @pytest.mark.parametrize("loc_shape,n_batch_dims", [
        ((5,), 0),
        ((3, 5), 1),
        ((2, 3, 4), 2),
    ])
    def test_mode(self, loc_shape, n_batch_dims):
        loc = pt.randn(loc_shape)
        dirac = Dirac(loc, n_batch_dims=n_batch_dims)
        mode = dirac.mode
        assert pt.allclose(mode, loc)

    @pytest.mark.parametrize("loc_shape,n_batch_dims", [
        ((5,), 0),
        ((3, 5), 1),
        ((2, 3, 4), 2),
    ])
    def test_variance(self, loc_shape, n_batch_dims):
        loc = pt.randn(loc_shape)
        dirac = Dirac(loc, n_batch_dims=n_batch_dims)
        variance = dirac.variance
        assert variance.shape == loc_shape
        assert pt.allclose(variance, pt.zeros_like(loc))

    def test_entropy(self):
        loc = pt.randn(5)
        dirac = Dirac(loc)
        entropy = dirac.entropy()
        assert pt.isinf(entropy) and entropy < 0

    def test_perplexity(self):
        loc = pt.randn(5)
        dirac = Dirac(loc)
        perplexity = dirac.perplexity()
        assert perplexity == 0.0


class TestDiracRsample:
    @pytest.mark.parametrize("loc_shape,n_batch_dims,sample_shape", [
        ((5,), 0, ()),
        ((5,), 0, (3,)),
        ((3, 5), 1, ()),
        ((3, 5), 1, (2,)),
        ((2, 3, 4), 2, (5,)),
    ])
    def test_rsample(self, loc_shape, n_batch_dims, sample_shape):
        loc = pt.randn(loc_shape)
        dirac = Dirac(loc, n_batch_dims=n_batch_dims)
        samples = dirac.rsample(sample_shape)
        expected_shape = sample_shape + dirac.batch_shape + dirac.event_shape
        assert samples.shape == expected_shape

    @pytest.mark.parametrize("loc_shape,n_batch_dims", [
        ((5,), 0),
        ((3, 5), 1),
        ((2, 3, 4), 2),
    ])
    def test_rsample_values_match_loc(self, loc_shape, n_batch_dims):
        loc = pt.randn(loc_shape)
        dirac = Dirac(loc, n_batch_dims=n_batch_dims)
        samples = dirac.rsample()
        assert pt.allclose(samples, loc)

    @pytest.mark.parametrize("sample_shape_input", [
        pt.Size([3]),
        [3],
        (3,),
    ])
    def test_rsample_with_different_shape_types(self, sample_shape_input):
        loc = pt.randn(5)
        dirac = Dirac(loc)
        samples = dirac.rsample(sample_shape_input)
        assert samples.shape == pt.Size([3, 5])


class TestDiracLogProb:
    @pytest.mark.parametrize("loc_shape,n_batch_dims", [
        ((5,), 0),
        ((3, 5), 1),
        ((2, 3, 4), 2),
    ])
    def test_log_prob_at_loc(self, loc_shape, n_batch_dims):
        loc = pt.randn(loc_shape)
        dirac = Dirac(loc, n_batch_dims=n_batch_dims)
        log_prob = dirac.log_prob(loc)
        assert log_prob == 0.0

    @pytest.mark.parametrize("loc_shape,n_batch_dims", [
        ((5,), 0),
        ((3, 5), 1),
        ((2, 3, 4), 2),
    ])
    def test_log_prob_away_from_loc(self, loc_shape, n_batch_dims):
        loc = pt.randn(loc_shape)
        dirac = Dirac(loc, n_batch_dims=n_batch_dims)
        value = loc + 1.0
        log_prob = dirac.log_prob(value)
        assert pt.isinf(log_prob) and log_prob < 0

    def test_log_prob_zero_difference(self):
        loc = pt.tensor([1.0, 2.0, 3.0])
        dirac = Dirac(loc)
        value = pt.tensor([1.0, 2.0, 3.0])
        log_prob = dirac.log_prob(value)
        assert log_prob == 0.0

    def test_log_prob_small_difference(self):
        loc = pt.tensor([1.0, 2.0, 3.0])
        dirac = Dirac(loc)
        value = loc + 1e-7
        log_prob = dirac.log_prob(value)
        assert pt.isinf(log_prob) and log_prob < 0


class TestDiracExpand:
    @pytest.mark.parametrize("loc_shape,n_batch_dims,new_batch_shape", [
        ((5,), 0, (3,)),
        ((5,), 0, (2, 3)),
        ((3, 5), 1, (7, 3)),
    ])
    def test_expand(self, loc_shape, n_batch_dims, new_batch_shape):
        loc = pt.randn(loc_shape)
        dirac = Dirac(loc, n_batch_dims=n_batch_dims)
        expanded = dirac.expand(new_batch_shape)
        assert expanded.batch_shape == new_batch_shape
        assert expanded.event_shape == dirac.event_shape

    @pytest.mark.parametrize("new_batch_shape_input", [
        pt.Size([3]),
        [3],
        (3,),
    ])
    def test_expand_with_different_shape_types(self, new_batch_shape_input):
        loc = pt.randn(5)
        dirac = Dirac(loc)
        expanded = dirac.expand(new_batch_shape_input)
        assert expanded.batch_shape == pt.Size([3])

    def test_expand_returns_dirac_instance(self):
        loc = pt.randn(5)
        dirac = Dirac(loc)
        expanded = dirac.expand((3,))
        assert isinstance(expanded, Dirac)


class TestDiracEnumerateSupport:
    @pytest.mark.parametrize("loc_shape,n_batch_dims", [
        ((5,), 0),
        ((3, 5), 1),
        ((2, 3, 4), 2),
    ])
    def test_enumerate_support_expand_true(self, loc_shape, n_batch_dims):
        loc = pt.randn(loc_shape)
        dirac = Dirac(loc, n_batch_dims=n_batch_dims)
        support = dirac.enumerate_support(expand=True)
        assert support.shape == pt.Size([1]) + dirac.batch_shape + dirac.event_shape

    def test_enumerate_support_expand_false_raises(self):
        loc = pt.randn(5)
        dirac = Dirac(loc)
        with pytest.raises(NotImplementedError):
            dirac.enumerate_support(expand=False)

    @pytest.mark.parametrize("loc_shape,n_batch_dims", [
        ((5,), 0),
        ((3, 5), 1),
    ])
    def test_enumerate_support_values_match_loc(self, loc_shape, n_batch_dims):
        loc = pt.randn(loc_shape)
        dirac = Dirac(loc, n_batch_dims=n_batch_dims)
        support = dirac.enumerate_support(expand=True)
        assert pt.allclose(support.squeeze(0), loc)


class TestDiracValidateArgs:
    @pytest.mark.parametrize("validate_args", [True, False, None])
    def test_init_with_validate_args(self, validate_args):
        loc = pt.randn(5)
        dirac = Dirac(loc, validate_args=validate_args)
        if validate_args is not None:
            assert dirac._validate_args == validate_args
