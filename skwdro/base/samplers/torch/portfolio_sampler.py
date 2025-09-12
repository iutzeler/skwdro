from typing import Optional, Union
import torch as pt

import skwdro.distributions as dst
from skwdro.base.samplers.torch.base_samplers import IsOptionalCovarianceSampler, NoLabelsSampler


class PortfolioNormalSampler(NoLabelsSampler, IsOptionalCovarianceSampler):
    data_s: dst.MultivariateNormal

    def __init__(
        self,
        xi,
        *,
        sigma: Optional[Union[pt.Tensor, float]] = None,
        tril: Optional[pt.Tensor] = None,
        prec: Optional[pt.Tensor] = None,
        cov: Optional[pt.Tensor] = None,
        seed: Optional[int] = None,
    ):
        """
        Example of an available sampler for portfolio management problems.

        - inputs are sampled from a gaussian distribution

        Specify the parameters of the distributions as keywords arguments.

        Parameters
        ----------
        xi: pt.Tensor
            mean for inputs
        sigma: float|Tensor
            scalar standard deviation shared through dimensions, for inputs.

        See :py:class:`skwdro.base.samplers.torch.base_samplers.IsOptionalCovarianceSampler`
        for other arguments.
        """
        super(PortfolioNormalSampler, self).__init__(
            dst.MultivariateNormal(
                loc=xi,
                **self.init_covar(
                    xi.size(-1),
                    sigma, tril, prec, cov
                )  # type: ignore
            ),
            seed
        )

    def reset_mean(self, xi, xi_labels):
        del xi_labels
        self.__init__(
            xi, seed=self.seed, tril=self.data_s._unbroadcasted_scale_tril
        )


class PortfolioLaplaceSampler(NoLabelsSampler, IsOptionalCovarianceSampler):
    # To adapt (there is a priori no MultivariateLaplace in Pytorch)
    data_s: dst.MultivariateNormal

    def __init__(
        self,
        xi,
        *,
        sigma: Optional[Union[pt.Tensor, float]] = None,
        tril: Optional[pt.Tensor] = None,
        prec: Optional[pt.Tensor] = None,
        cov: Optional[pt.Tensor] = None,
        seed: Optional[int] = None,
    ):
        """
        Example of an available sampler for portfolio management problems.

        - inputs are sampled from a Laplace distribution

        Specify the parameters of the distributions as keywords arguments.

        Parameters
        ----------
        xi: pt.Tensor
            mean for inputs
        sigma: float|Tensor
            scalar standard deviation shared through dimensions, for inputs.

        See :py:class:`skwdro.base.samplers.torch.base_samplers.IsOptionalCovarianceSampler`
        for other arguments.
        """
        super(PortfolioLaplaceSampler, self).__init__(
            dst.MultivariateNormal(  # Idem
                loc=xi,
                **self.init_covar(
                    xi.size(-1),
                    sigma, tril, prec, cov
                )  # type: ignore
            ),
            seed
        )

    def reset_mean(self, xi, xi_labels):
        del xi_labels
        self.__init__(
            xi, seed=self.seed, tril=self.data_s._unbroadcasted_scale_tril)
