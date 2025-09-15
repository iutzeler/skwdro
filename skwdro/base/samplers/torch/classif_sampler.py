from typing import Optional, Union
import torch as pt

import skwdro.distributions as dst
from .base_samplers import IsOptionalCovarianceSampler, LabeledSampler


class ClassificationNormalNormalSampler(LabeledSampler, IsOptionalCovarianceSampler):
    data_s: dst.MultivariateNormal
    labels_s: dst.MultivariateNormal

    def __init__(
        self,
        xi: pt.Tensor, xi_labels: pt.Tensor,
        *,
        sigma: Optional[Union[float, pt.Tensor]] = None,
        tril: Optional[pt.Tensor] = None,
        prec: Optional[pt.Tensor] = None,
        cov: Optional[pt.Tensor] = None,
        l_sigma: Optional[Union[float, pt.Tensor]] = None,
        l_tril: Optional[pt.Tensor] = None,
        l_prec: Optional[pt.Tensor] = None,
        l_cov: Optional[pt.Tensor] = None,
        seed: Optional[int] = None,
    ):
        """
        Example of an available sampler for classification problems.

        - inputs are sampled from a gaussian distribution
        - targets from another one.

        Specify the parameters of the distributions as keywords arguments.

        Parameters
        ----------
        xi: pt.Tensor
            mean for inputs
        xi_labels: pt.Tensor
            mean for targets
        sigma: float|Tensor
            scalar standard deviation shared through dimensions, for inputs.
        l_sigma: float|Tensor
            scalar standard deviation shared through dimensions, for targets.

        See :py:class:`skwdro.base.samplers.torch.base_samplers.IsOptionalCovarianceSampler`
        for other arguments.
        """
        assert len(xi.size()) >= 2
        assert len(xi_labels.size()) >= 2
        covar = self.init_covar(xi.size(-1), sigma, tril, prec, cov)
        labels_covar = self.init_covar(
            xi_labels.size(-1), l_sigma, l_tril, l_prec, l_cov
        )
        super(ClassificationNormalNormalSampler, self).__init__(
            dst.MultivariateNormal(
                loc=xi,
                **covar  # type: ignore
            ),
            dst.MultivariateNormal(
                loc=xi_labels,
                **labels_covar  # type: ignore
            ),
            seed
        )

    def reset_mean(self, xi, xi_labels):
        self.__init__(
            xi,
            xi_labels,
            seed=self.seed,
            tril=self.data_s._unbroadcasted_scale_tril,
            l_tril=self.labels_s._unbroadcasted_scale_tril
        )


class ClassificationNormalIdSampler(LabeledSampler, IsOptionalCovarianceSampler):
    data_s: dst.MultivariateNormal
    # Just a placeholder to remember the mean. Dirac does not exist in torch...
    labels_s: dst.MultivariateNormal

    def __init__(
        self,
        xi: pt.Tensor, xi_labels: pt.Tensor,
        *,
        sigma: Optional[Union[float, pt.Tensor]] = None,
        tril: Optional[pt.Tensor] = None,
        prec: Optional[pt.Tensor] = None,
        cov: Optional[pt.Tensor] = None,
        seed: Optional[int],
    ):
        """
        Example of an available sampler for classification problems.

        - inputs are sampled from a gaussian distribution
        - targets are not noisy.

        Specify the parameters of the distributions as keywords arguments.

        Parameters
        ----------
        xi: pt.Tensor
            mean for inputs
        xi_labels: pt.Tensor
            mean for targets
        sigma: float|Tensor
            scalar standard deviation shared through dimensions, for inputs.

        See :py:class:`skwdro.base.samplers.torch.base_samplers.IsOptionalCovarianceSampler`
        for other arguments.
        """
        covar = self.init_covar(xi.size(-1), sigma, tril, prec, cov)
        super(ClassificationNormalIdSampler, self).__init__(
            dst.MultivariateNormal(
                loc=xi,
                **covar  # type: ignore
            ),
            dst.Dirac(
                loc=xi_labels,
                n_batch_dims=1,
                validate_args=True
            ),
            seed
        )

    def sample_labels(self, n_sample: int) -> pt.Tensor:
        """
        Just get as many labels as data points (n_sample).
        """
        return self.data_s.mean.unsqueeze(0).expand(n_sample, -1, -1)

    def reset_mean(self, xi, xi_labels):
        self.__init__(
            xi,
            xi_labels,
            seed=self.seed,
            tril=self.data_s._unbroadcasted_scale_tril,
        )


class ClassificationNormalBernouilliSampler(LabeledSampler, IsOptionalCovarianceSampler):
    data_s: dst.MultivariateNormal
    labels_s: dst.TransformedDistribution

    def __init__(
        self,
        p: float,
        xi: pt.Tensor, xi_labels: pt.Tensor,
        *,
        sigma: Optional[Union[float, pt.Tensor]] = None,
        tril: Optional[pt.Tensor] = None,
        prec: Optional[pt.Tensor] = None,
        cov: Optional[pt.Tensor] = None,
        seed: Optional[int],
    ):
        r"""
        Example of an available sampler for classification problems.

        - inputs are sampled from a gaussian distribution
        - targets are sampled randomly in :math:`\{0, 1\}`.

        Specify the parameters of the distributions as keywords arguments.

        Parameters
        ----------
        p: float|Tensor
            Probability of switch for bernouilli.
        xi: pt.Tensor
            mean for inputs
        xi_labels: pt.Tensor
            mean for targets
        sigma: float|Tensor
            scalar standard deviation shared through dimensions, for inputs.

        See :py:class:`skwdro.base.samplers.torch.base_samplers.IsOptionalCovarianceSampler`
        for other arguments.
        """
        assert 0. <= p <= 1.
        covar = self.init_covar(xi.size(-1), sigma, tril, prec, cov)
        self.p = p
        super(ClassificationNormalBernouilliSampler, self).__init__(
            dst.MultivariateNormal(
                loc=xi,
                **covar  # type: ignore
            ),
            dst.TransformedDistribution(
                dst.Bernoulli(
                    p
                ),
                dst.transforms.AffineTransform(
                    loc=-xi_labels,
                    scale=2 * xi_labels
                )
            ),
            seed
        )

    def sample_labels(self, n_sample: int) -> pt.Tensor:
        """
        Overrides w/ ``sample`` to prevent ``rsample`` from crashing since bernouilli
        isn't reparametrizeable.
        """
        zeta_labels = self.labels_s.sample(pt.Size((n_sample,)))
        assert isinstance(zeta_labels, pt.Tensor)
        return zeta_labels

    def reset_mean(self, xi, xi_labels):
        self.__init__(
            self.p,
            xi,
            xi_labels,
            seed=self.seed,
            tril=self.data_s._unbroadcasted_scale_tril,
        )
