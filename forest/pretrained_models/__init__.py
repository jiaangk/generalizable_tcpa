from .clip import *
from .convnext import *
from .dino import *
from .efficientnet import *
from .inception import *
from .resnet import *
from .swin import *
from .align import *
import torch
from pathlib import Path

def bypass_last_layer(model):
    layer_cake = list(model.children())
    headless_model = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())  # this works most of the time all of the time :<

    return headless_model

def pretrained_picker(name, args):
    if name == 'custom':
        model = torch.load(args.pretrained_path / Path('model.pth'), map_location=torch.device('cpu'))
    else:
        model = globals()[name]()
    model.eval()
    model.requires_grad_(False)

    if not ('clip' in name or 'dino' in name or 'align' in name):
        model = bypass_last_layer(model)

    return model

__all__ = ['pretrained_picker']