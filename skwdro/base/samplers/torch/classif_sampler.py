from typing import Optional, Union
import torch as pt
import torch.distributions as dst

from .base_samplers import IsOptionalCovarianceSampler, LabeledSampler

class ClassificationNormalNormalSampler(LabeledSampler, IsOptionalCovarianceSampler):
    data_s: dst.MultivariateNormal
    labels_s: dst.MultivariateNormal
    def __init__(self, xi: pt.Tensor, xi_labels: pt.Tensor, *,
                 sigma: Optional[Union[float, pt.Tensor]]=None,
                 tril: Optional[pt.Tensor]=None,
                 prec: Optional[pt.Tensor]=None,
                 cov: Optional[pt.Tensor]=None,
                 l_sigma: Optional[Union[float, pt.Tensor]]=None,
                 l_tril: Optional[pt.Tensor]=None,
                 l_prec: Optional[pt.Tensor]=None,
                 l_cov: Optional[pt.Tensor]=None
                 ):
        assert len(xi.size()) >= 2
        assert len(xi_labels.size()) >= 2
        covar = self.init_covar(xi.size(-1), sigma, tril, prec, cov)
        labels_covar = self.init_covar(xi_labels.size(-1), l_sigma, l_tril, l_prec, l_cov)
        super(ClassificationNormalNormalSampler, self).__init__(
                dst.MultivariateNormal(
                    loc=xi,
                    **covar
                ),
                dst.MultivariateNormal(
                    loc=xi_labels,
                    **labels_covar
                )
            )

    def reset_mean(self, xi, xi_labels):
        self.__init__(
                xi,
                xi_labels,
                tril=self.data_s._unbroadcasted_scale_tril,
                l_tril=self.labels_s._unbroadcasted_scale_tril
                )

class ClassificationNormalBernouilliSampler(LabeledSampler, IsOptionalCovarianceSampler):
    data_s: dst.MultivariateNormal
    labels_s: dst.TransformedDistribution
    def __init__(self, xi: pt.Tensor, xi_labels: pt.Tensor, *,
                 p: float,
                 sigma: Optional[Union[float, pt.Tensor]]=None,
                 tril: Optional[pt.Tensor]=None,
                 prec: Optional[pt.Tensor]=None,
                 cov: Optional[pt.Tensor]=None
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
                        scale=2*xi_labels
                    )
                )
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
                tril=self.data_s._unbroadcasted_scale_tril,
                p=self.p
                )
