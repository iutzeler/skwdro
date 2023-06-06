from typing import Optional, Union
import torch as pt
import torch.distributions as dst

from skwdro.base.samplers.torch.base_samplers import IsOptionalCovarianceSampler, NoLabelsSampler

class NewsVendorNormalSampler(NoLabelsSampler, IsOptionalCovarianceSampler):
    data_s: dst.MultivariateNormal
    def __init__(self, xi, *,
                 sigma: Optional[Union[pt.Tensor, float]]=None,
                 tril: Optional[pt.Tensor]=None,
                 prec: Optional[pt.Tensor]=None,
                 cov: Optional[pt.Tensor]=None
                 ):
        super(NewsVendorNormalSampler, self).__init__(
                dst.MultivariateNormal(
                    loc=xi,
                    **self.init_covar(xi.size(-1), sigma, tril, prec, cov)
                )
            )

    def reset_mean(self, xi, xi_labels):
        self.__init__(xi, tril=self.data_s._unbroadcasted_scale_tril)
