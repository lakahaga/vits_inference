from typing import Optional

import torch
import torch.nn as nn

import numpy as np


class MakePadMask(nn.Module):
    def __init__(self, max_seq_len=512, flip=True):
        super().__init__()
        if flip:
            self.mask_pad = torch.Tensor(1 - np.tri(max_seq_len)).type(torch.bool)
        else:
            self.mask_pad = torch.Tensor(np.tri(max_seq_len)).type(torch.bool)
    
    def forward(self, lengths, xs=None, length_dim=-1, maxlen=None):
        """Make mask tensor containing indices of padded part.
        This implementation creates the same mask tensor with original make_pad_mask,
        which can be converted into onnx format.
        Dimension length of xs should be 2 or 3.
        """
        if length_dim == 0:
            raise ValueError("length_dim cannot be 0: {}".format(length_dim))

        if xs is not None and len(xs.shape) == 3:
            if length_dim == 1:
                lengths = lengths.unsqueeze(1).expand(
                    *xs.transpose(1, 2).shape[:2])
            else:
                lengths = lengths.unsqueeze(1).expand(*xs.shape[:2])

        if maxlen is not None:
            m = maxlen
        elif xs is not None:
            m = xs.shape[-1]
        else:
            m = torch.max(lengths)

        mask = self.mask_pad[lengths - 1][..., :m].type(torch.float32)

        if length_dim == 1:
            return mask.transpose(1, 2)
        else:
            return mask


def normalize(input: torch.Tensor, p: float = 2.0, dim: int = 1, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if out is None:
        denom = input.norm(p, dim, keepdim=True).expand_as(input)
        return input / denom
    else:
        denom = input.norm(p, dim, keepdim=True).expand_as(input)
        return torch.div(input, denom, out=out)


def subsequent_mask(size: torch.Tensor):
    return torch.ones(size, size).tril()
