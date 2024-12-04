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

def transform_picker(name, args):
    if name == 'custom':
        transform = torch.load(args.pretrained_path / Path('transform.pth'))
        return transform
    
    if 'dinov2' in name:
        return make_classification_eval_transform()
    
    return globals()['%s_transform' % name]()

__all__ = ['transform_picker']