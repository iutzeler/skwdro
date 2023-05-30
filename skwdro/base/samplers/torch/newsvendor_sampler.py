from typing import Optional
import torch as pt
import torch.distributions as dst

from base_samplers import IsOptionalCovarianceSampler, NoLabelsSampler

class NewsVendorNormalSampler(NoLabelsSampler, IsOptionalCovarianceSampler):
    def __init__(self, xi, *,
                 sigma: Optional[float]=None,
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
