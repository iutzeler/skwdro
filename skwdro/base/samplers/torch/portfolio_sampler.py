from typing import Optional, Union
import torch as pt
import torch.distributions as dst
from mv_laplace import MvLaplaceSampler

from skwdro.base.samplers.torch.base_samplers import IsOptionalCovarianceSampler, NoLabelsSampler

class PortfolioNormalSampler(NoLabelsSampler, IsOptionalCovarianceSampler):
    data_s: dst.MultivariateNormal
    def __init__(self, xi, seed: int, *,
                 sigma: Optional[Union[pt.Tensor, float]]=None,
                 tril: Optional[pt.Tensor]=None,
                 prec: Optional[pt.Tensor]=None,
                 cov: Optional[pt.Tensor]=None
                 ):
        super(PortfolioNormalSampler, self).__init__(
                dst.MultivariateNormal(
                    loc=xi,
                    **self.init_covar(xi.size(-1), sigma, tril, prec, cov)
                ),
                seed
            )

    def reset_mean(self, xi, xi_labels):
        self.__init__(xi, self.seed, tril=self.data_s._unbroadcasted_scale_tril)

class PortfolioLaplaceSampler(NoLabelsSampler, IsOptionalCovarianceSampler):
    data_s: dst.MultivariateNormal #To adapt (there is a priori no MultivariateLaplace in Pytorch)
    def __init__(self, xi, seed: int, *,
                 sigma: Optional[Union[pt.Tensor, float]]=None,
                 tril: Optional[pt.Tensor]=None,
                 prec: Optional[pt.Tensor]=None,
                 cov: Optional[pt.Tensor]=None
                 ):
        super(PortfolioLaplaceSampler, self).__init__(
                dst.MultivariateNormal( #Idem
                    loc=xi,
                    **self.init_covar(xi.size(-1), sigma, tril, prec, cov)
                ),
                seed
            )

    def reset_mean(self, xi, xi_labels):
        self.__init__(xi, self.seed, tril=self.data_s._unbroadcasted_scale_tril)
