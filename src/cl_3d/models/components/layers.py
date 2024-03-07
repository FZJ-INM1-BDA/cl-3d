from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn

from cl_3d.models.components.distributed import sync_ddp_if_available


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process, supporting backward propagation.

    From https://github.com/Spijkervet/SimCLR/blob/04bcf2baa1fb5631a0a636825aabe469865ad8a9/simclr/modules/gather.py
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[torch.distributed.get_rank()]
        return grad_out


class RunningNorm2D(nn.Module):
    """
    Compute running statistics on the fly over whole training

    Parts are copied from https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d

    Args:
        num_features (int): numver of input features (or channels)
        max_iter (int): maximum iterations after which statistics are not updated anymore and kept constant
    """

    def __init__(self, num_features, max_iter=1024, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(RunningNorm2D, self).__init__()
        self.num_features = num_features
        self.max_iter = max_iter
        self.register_buffer('num_batches_tracked',
                             torch.tensor(0, dtype=torch.long,
                                          **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
        self.num_batches_tracked: Optional[Tensor]
        self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
        self.register_buffer('running_var', torch.zeros(num_features, **factory_kwargs))
        self.running_mean: Optional[Tensor]
        self.running_var: Optional[Tensor]

    def forward(self, input):
        self._check_input_dim(input)

        if self.training and self.num_batches_tracked < self.max_iter:
            # Detach input since no gradient is required for running stats
            self._update_stats(input.detach())

        if self.num_batches_tracked == 0:
            input_norm = input
        else:
            input_norm = (input - self.running_mean[:, None, None]) / \
                         (torch.sqrt(self.running_var)[:, None, None] + 1e-8)

        return input_norm

    def reset_running_stats(self) -> None:
        self.running_mean.zero_()
        self.running_var.zero_()
        self.num_batches_tracked.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

    def _update_stats(self, input):
        self.num_batches_tracked += 1

        n_mean = torch.mean(input, dim=(0, 2, 3))
        n_mean = sync_ddp_if_available(n_mean, reduce_op='mean')
        self.running_mean += (n_mean - self.running_mean) / float(self.num_batches_tracked)

        n_var = torch.mean((input - self.running_mean[:, None, None]) ** 2, dim=(0, 2, 3))
        n_var = sync_ddp_if_available(n_var, reduce_op='mean')
        self.running_var += (n_var - self.running_var) / float(self.num_batches_tracked)


class PLIMods(nn.Module):

    def __init__(self):
        """
        Returns transmittance, direction, retardation as 0 and 1 frequency of a DFT transformed signal.
        Returns 3 channels.

        IMPORTANT: The input for direction is expected to be in radians!
        """
        super(PLIMods, self).__init__()


    def forward(self, trans, dir, ret):
        """
        :param trans: Transmittance of shape N1HW or NHW
        :param dir: Direction of shape N1HW or NHW
        :param ret: Retardation of shape N1HW or NHW
        :return:
        """
        assert trans.shape == dir.shape == ret.shape, \
            f"Differing shapes found for input modalities {trans.shape}, {dir.shape}, {ret.shape}"
        
        if len(trans.shape) == 3:
            trans = trans[:, None]
            dir = dir[:, None]
            ret = ret[:, None]

        dft0 = trans
        dft1 = ret * torch.cos(2 * dir)
        dft2 = ret * torch.sin(2 * dir)
        dft = torch.cat((dft0, dft1, dft2), dim=1)

        return dft