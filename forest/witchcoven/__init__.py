"""Interface for poison recipes."""
from .witch_matching import WitchGradientMatching
from .witch_bullseye import WitchBullsEye

import torch


def Witch(args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Implement Main interface."""
    if args.recipe == 'gradient-matching':
        return WitchGradientMatching(args, setup)
    elif args.recipe == 'bullseye':
        return WitchBullsEye(args, setup)
    else:
        raise NotImplementedError()


__all__ = ['Witch']
