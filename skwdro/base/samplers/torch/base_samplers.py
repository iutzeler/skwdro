from typing import Dict, Optional, Tuple, Union
from abc import ABC, abstractmethod

import random
import torch as pt

import skwdro.distributions as dst


class BaseSampler(ABC):
    seed: int

    def __init__(self, seed: int):
        self.seed = seed

        # Set seed
        pt.manual_seed(seed)
        random.seed(seed)

    @abstractmethod
    def sample(
        self, n_samples: int
    ) -> Tuple[pt.Tensor, Optional[pt.Tensor]]:
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample(1)

    @property
    @abstractmethod
    def produces_labels(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def reset_mean(self, xi, xi_labels):
        raise NotImplementedError()

    @abstractmethod
    def log_prob(
            self,
            zeta: pt.Tensor,
            zeta_labels: Optional[pt.Tensor]
    ) -> pt.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def log_prob_recentered(
            self,
            xi: pt.Tensor,
            xi_labels: Optional[pt.Tensor],
            zeta: pt.Tensor,
            zeta_labels: Optional[pt.Tensor]
    ) -> pt.Tensor:
        raise NotImplementedError()


class NoLabelsSampler(BaseSampler, ABC):
    def __init__(self, data_sampler: dst.Distribution, seed: int):
        super(NoLabelsSampler, self).__init__(seed)
        self.data_s = data_sampler

    def sample(self, n_samples: int):
        return self.data_s.rsample(pt.Size((n_samples,))), None

    @property
    def produces_labels(self):
        return False

    def log_prob(
            self,
            zeta: pt.Tensor,
            zeta_labels: Optional[pt.Tensor]
    ) -> pt.Tensor:
        assert zeta_labels is None
        return self.data_s.log_prob(zeta).sum(-1, keepdim=True)

    def log_prob_recentered(
            self,
            xi: pt.Tensor,
            xi_labels: Optional[pt.Tensor],
            zeta: pt.Tensor,
            zeta_labels: Optional[pt.Tensor]
    ) -> pt.Tensor:
        assert xi_labels is None and zeta_labels is None
        return self.data_s.log_prob(zeta - xi + self.data_s.mean).sum(-1, keepdim=True)


class LabeledSampler(BaseSampler, ABC):
    def __init__(self, data_sampler: dst.Distribution, labels_sampler: dst.Distribution, seed: int):
        super(LabeledSampler, self).__init__(seed)
        self.data_s = data_sampler
        self.labels_s = labels_sampler

    def sample(self, n_samples: int):
        zeta = self.sample_data(n_samples)
        zeta_labels = self.sample_labels(n_samples)
        return zeta, zeta_labels

    def sample_data(self, n_sample: int):
        return self.data_s.rsample(pt.Size((n_sample,)))

    def sample_labels(self, n_sample: int):
        return self.labels_s.rsample(pt.Size((n_sample,)))

    @property
    def produces_labels(self):
        return True

    def log_prob(
            self,
            zeta: pt.Tensor,
            zeta_labels: Optional[pt.Tensor]
    ) -> pt.Tensor:
        assert zeta_labels is not None
        lp_zeta = self.data_s.log_prob(zeta).sum(-1, keepdim=True)
        lp_zeta_labels = self.labels_s.log_prob(zeta_labels).sum(-1, keepdim=True)
        lp = lp_zeta + lp_zeta_labels
        return lp

    def log_prob_recentered(
            self,
            xi: pt.Tensor,
            xi_labels: Optional[pt.Tensor],
            zeta: pt.Tensor,
            zeta_labels: Optional[pt.Tensor]
    ) -> pt.Tensor:
        assert zeta_labels is not None and xi_labels is not None
        lp_zeta = self.data_s.log_prob(zeta).sum(-1, keepdim=True)
        lp_zeta_labels = self.labels_s.log_prob(zeta_labels).sum(-1, keepdim=True)
        lp = lp_zeta + lp_zeta_labels
        return lp

# Helper class ########################


class IsOptionalCovarianceSampler(ABC):
    def init_covar(
        self,
        d: int,
        sigma: Optional[Union[float, pt.Tensor]] = None,
        tril: Optional[pt.Tensor] = None,
        prec: Optional[pt.Tensor] = None,
        cov: Optional[pt.Tensor] = None
    ) -> Dict[str, pt.Tensor]:
        """
        Sets up the covariance matrix in the correct format to give as a kwarg to torch distributions.
        Order of importance for non-None values:
        * sigma: defines Id/sigma**2 as cov matrix, given as L^T@L
        * tril: defines L s.t. C=L^TL
        * cov: defines the full C matrix
        * prec: defines the precision matrix C^-1, only useful for fast CDF computation and bad otherwise
        """
        if sigma is not None:
            return {"scale_tril": pt.eye(d) * sigma}
        elif tril is not None:
            return {"scale_tril": tril}
        elif cov is not None:
            return {"covariance_matrix": cov}
        elif prec is not None:
            return {"precision_matrix": prec}
        else:
            raise ValueError(' '.join([
                "Please provide",
                "a valid covariance",
                "matrix for the"
                "constructor of",
                str(self.__class__.__name__)
            ]))
