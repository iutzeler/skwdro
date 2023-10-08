from typing import Optional, Dict

import torch as pt
import torch.distributions as dst
import torch.distributions.constraints as cstr

class Dirac(dst.ExponentialFamily):
    @property
    def arg_constraints(self) -> Dict[str, cstr.Constraint]:
        return {
            "loc": cstr.real_vector
        }
    @property
    def support(self) -> Optional[cstr.Constraint]:
        return cstr.real_vector # type: ignore
    has_rsample = True

    def __init__(
            self,
            loc: pt.Tensor,
            n_batch_dims: int=0,
            validate_args: Optional[bool]=None):
        locshape = loc.size()
        batch_shape = locshape[:n_batch_dims]
        event_shape = locshape[n_batch_dims:]
        super().__init__(batch_shape, event_shape, validate_args)

        self.loc: pt.Tensor = loc

    def expand(self, batch_shape: pt.Size, _instance=None):
        new: Dirac = self._get_checked_instance(Dirac, _instance)
        batch_shape = pt.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        new.loc = self.loc.expand(loc_shape)
        super(Dirac, new).__init__(batch_shape, self.event_shape, validate_args=False) # type: ignore
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> pt.Tensor:
        return self.loc

    @property
    def mode(self) -> pt.Tensor:
        return self.loc

    @property
    def variance(self) -> pt.Tensor:
        return pt.zeros_like(self.loc)

    def r_sample(self, sample_shape: pt.Size) -> pt.Tensor:
        return self.loc.expand(self._extended_shape(sample_shape))

    def log_prob(self, value: pt.Tensor) -> pt.Tensor:
        return pt.tensor(pt.inf) if (value - self.loc).abs().sum() == 0. else pt.tensor(-pt.inf)

    def enumerate_support(self, expand: bool=True) -> pt.Tensor:
        if expand:
            return self.r_sample(pt.Size((1,)))
        else:
            raise NotImplementedError

    def entropy(self) -> pt.Tensor:
        return pt.tensor(-pt.inf)

    def perplexity(self) -> pt.Tensor:
        return pt.tensor(0.)
