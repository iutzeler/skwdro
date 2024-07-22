from typing import Optional, Union
import torch as pt

import skwdro.distributions as dst
from .base_samplers import IsOptionalCovarianceSampler, LabeledSampler


class ClassificationNormalNormalSampler(LabeledSampler, IsOptionalCovarianceSampler):
    data_s: dst.MultivariateNormal
    labels_s: dst.MultivariateNormal

    def __init__(self, xi: pt.Tensor, xi_labels: pt.Tensor, seed: int, *,
                 sigma: Optional[Union[float, pt.Tensor]] = None,
                 tril: Optional[pt.Tensor] = None,
                 prec: Optional[pt.Tensor] = None,
                 cov: Optional[pt.Tensor] = None,
                 l_sigma: Optional[Union[float, pt.Tensor]] = None,
                 l_tril: Optional[pt.Tensor] = None,
                 l_prec: Optional[pt.Tensor] = None,
                 l_cov: Optional[pt.Tensor] = None
                 ):
        assert len(xi.size()) >= 2
        assert len(xi_labels.size()) >= 2
        covar = self.init_covar(xi.size(-1), sigma, tril, prec, cov)
        labels_covar = self.init_covar(
            xi_labels.size(-1), l_sigma, l_tril, l_prec, l_cov)
        super(ClassificationNormalNormalSampler, self).__init__(
            dst.MultivariateNormal(
                loc=xi,
                **covar
            ),
            dst.MultivariateNormal(
                loc=xi_labels,
                **labels_covar
            ),
            seed
        )

    def reset_mean(self, xi, xi_labels):
        self.__init__(
            xi,
            xi_labels,
            self.seed,
            tril=self.data_s._unbroadcasted_scale_tril,
            l_tril=self.labels_s._unbroadcasted_scale_tril
        )


class ClassificationNormalIdSampler(LabeledSampler, IsOptionalCovarianceSampler):
    data_s: dst.MultivariateNormal
    # Just a placeholder to remember the mean. Dirac does not exist in torch...
    labels_s: dst.MultivariateNormal

    def __init__(self, xi: pt.Tensor, xi_labels: pt.Tensor, seed: int, *,
                 sigma: Optional[Union[float, pt.Tensor]] = None,
                 tril: Optional[pt.Tensor] = None,
                 prec: Optional[pt.Tensor] = None,
                 cov: Optional[pt.Tensor] = None
                 ):
        covar = self.init_covar(xi.size(-1), sigma, tril, prec, cov)
        super(ClassificationNormalIdSampler, self).__init__(
            dst.MultivariateNormal(
                loc=xi,
                **covar
            ),
            dst.Dirac(
                loc=xi_labels,
                n_batch_dims=1,
                validate_args=True
            ),
            seed
        )

    def sample_labels(self, n_sample: int):
        """
        Just get as many labels as data points (n_sample).
        """
        return self.data_s.mean.unsqueeze(0).expand(n_sample, -1, -1)

    def reset_mean(self, xi, xi_labels):
        self.__init__(
            xi,
            xi_labels,
            self.seed,
            tril=self.data_s._unbroadcasted_scale_tril,
        )


class ClassificationNormalBernouilliSampler(LabeledSampler, IsOptionalCovarianceSampler):
    data_s: dst.MultivariateNormal
    labels_s: dst.TransformedDistribution

    def __init__(self, xi: pt.Tensor, xi_labels: pt.Tensor, seed: int, *,
                 p: float,
                 sigma: Optional[Union[float, pt.Tensor]] = None,
                 tril: Optional[pt.Tensor] = None,
                 prec: Optional[pt.Tensor] = None,
                 cov: Optional[pt.Tensor] = None
                 ):
        assert 0. <= p <= 1.
        covar = self.init_covar(xi.size(-1), sigma, tril, prec, cov)
        self.p = p
        super(ClassificationNormalBernouilliSampler, self).__init__(
            dst.MultivariateNormal(
                loc=xi,
                **covar
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

    def sample_labels(self, n_sample: int):
        """
        Overrides w/ ``sample`` to prevent ``rsample`` from crashing since bernouilli
        isn't reparametrizeable.
        """
        return self.labels_s.sample(pt.Size((n_sample,)))

    def reset_mean(self, xi, xi_labels):
        self.__init__(
            xi,
            xi_labels,
            self.seed,
            tril=self.data_s._unbroadcasted_scale_tril,
            p=self.p
        )
