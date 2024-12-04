"""Model definitions."""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models import AlexNet, resnet18
from pathlib import Path

from collections import OrderedDict

from .mobilenet import MobileNetV2
from .vgg import VGG
from .swin import Swin
from .levit import levit_picker
from ..consts import DATASET_SETTING, PRETRAINED_MODELS

class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=1024, num_class=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_class)
        self.droupout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.droupout(x)
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)

def get_model(model_name, dataset_name, args, pretrained=None):
    """Retrieve an appropriate architecture."""
    in_channels = DATASET_SETTING[dataset_name]['in_channels']
    num_classes = DATASET_SETTING[dataset_name]['num_classes']
    
    if pretrained:
        match model_name:
            case 'Linear':
                if pretrained == 'custom':
                    dim = torch.load(Path(args.pretrained_path) / 'dim.pth')
                    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(dim, num_classes))
                else:
                    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(PRETRAINED_MODELS[pretrained], num_classes))
            case 'MLP':
                if pretrained == 'custom':
                    dim = torch.load(Path(args.pretrained_path) / 'dim.pth')
                    model = MLP(dim, num_classes)
                else:
                    model = MLP(PRETRAINED_MODELS[pretrained], num_class=num_classes)
            case '_':
                raise ValueError('When using a pretrained model as a feature extractor, the training model can only be linear model.')
        return model
    
    if model_name == 'std_ResNet18':
        model = resnet18(num_classes=num_classes)
    elif 'ResNet' in model_name:
        model = resnet_picker(model_name, dataset_name, in_channels, num_classes)
    elif 'VGG' in model_name:
        model = VGG(model_name)
    elif 'LeViT' in model_name:
        model = levit_picker(model_name, dataset_name, num_classes)
    else:
        match model_name:
            case 'Swin':
                model = Swin(num_classes=num_classes)
            case 'alexnet':
                model = AlexNet(num_classes=num_classes)
            case 'ConvNet':
                model = convnet(width=32, in_channels=in_channels, num_classes=num_classes)
            case 'ConvNet64':
                model = convnet(width=64, in_channels=in_channels, num_classes=num_classes)
            case 'ConvNet128':
                model = convnet(width=64, in_channels=in_channels, num_classes=num_classes)
            case 'ConvNetBN':
                model = ConvNetBN(width=64, in_channels=in_channels, num_classes=num_classes)
            case 'Linear':
                model = linear_model(dataset_name, num_classes=num_classes)
            case 'MLP':
                model = mlp_model(dataset_name, num_classes=num_classes)
            case 'alexnet-mp':
                model = alexnet_metapoison(in_channels=in_channels, num_classes=num_classes, batchnorm=False)
            case 'alexnet-mp-bn':
                model = alexnet_metapoison(in_channels=in_channels, num_classes=num_classes, batchnorm=True)
            case 'MobileNetV2':
                model = MobileNetV2(num_classes=num_classes, train_dp=0, test_dp=0, droplayer=0, bdp=0)
            case _:
                raise ValueError(f'Architecture {model_name} not implemented for dataset {dataset_name}.')
        
    return model


def linear_model(dataset, num_classes=10):
    """Define the simplest linear model."""
    if 'cifar' in dataset.lower():
        dimension = 3072
    elif 'mnist' in dataset.lower():
        dimension = 784
    elif 'imagenet' in dataset.lower():
        dimension = 150528
    elif 'tinyimagenet' in dataset.lower():
        dimension = 64**2 * 3
    else:
        raise ValueError('Linear model not defined for dataset.')
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(dimension, num_classes))

def mlp_model(dataset, num_classes=10):
    """Define the simplest linear model."""
    if 'cifar' in dataset.lower():
        dimension = 3072
    elif 'mnist' in dataset.lower():
        dimension = 784
    elif 'imagenet' in dataset.lower():
        dimension = 150528
    elif 'tinyimagenet' in dataset.lower():
        dimension = 64**2 * 3
    else:
        raise ValueError('Linear model not defined for dataset.')
    return torch.nn.Sequential(torch.nn.Flatten(), MLP(input_size=dimension, num_class=num_classes))

def convnet(width=32, in_channels=3, num_classes=10, **kwargs):
    """Define a simple ConvNet. This architecture only really works for CIFAR10."""
    model = torch.nn.Sequential(OrderedDict([
        ('conv0', torch.nn.Conv2d(in_channels, 1 * width, kernel_size=3, padding=1)),
        ('relu0', torch.nn.ReLU()),
        ('conv1', torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
        ('relu1', torch.nn.ReLU()),
        ('conv2', torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
        ('relu2', torch.nn.ReLU()),
        ('conv3', torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
        ('relu3', torch.nn.ReLU()),
        ('pool3', torch.nn.MaxPool2d(3)),
        ('conv4', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
        ('relu4', torch.nn.ReLU()),
        ('pool4', torch.nn.MaxPool2d(3)),
        ('flatten', torch.nn.Flatten()),
        ('linear', torch.nn.Linear(36 * width, num_classes))
    ]))
    return model


def alexnet_metapoison(widths=[16, 32, 32, 64, 64], in_channels=3, num_classes=10, batchnorm=False):
    """AlexNet variant as used in MetaPoison."""
    def convblock(width_in, width_out):
        if batchnorm:
            bn = torch.nn.BatchNorm2d(width_out)
        else:
            bn = torch.nn.Identity()
        return torch.nn.Sequential(torch.nn.Conv2d(width_in, width_out, kernel_size=3, padding=1),
                                   torch.nn.ReLU(),
                                   bn,
                                   torch.nn.MaxPool2d(2, 2))
    blocks = []
    width_in = in_channels
    for width in widths:
        blocks.append(convblock(width_in, width))
        width_in = width

    model = torch.nn.Sequential(*blocks, torch.nn.Flatten(), torch.nn.Linear(widths[-1], num_classes))
    return model


class ConvNetBN(torch.nn.Module):
    """ConvNetBN."""

    def __init__(self, width=64, num_classes=10, in_channels=3):
        """Init with width and num classes."""
        super().__init__()
        self.model = torch.nn.Sequential(OrderedDict([
            ('conv0', torch.nn.Conv2d(in_channels, 1 * width, kernel_size=3, padding=1)),
            ('bn0', torch.nn.BatchNorm2d(1 * width)),
            ('relu0', torch.nn.ReLU()),

            ('conv1', torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn1', torch.nn.BatchNorm2d(2 * width)),
            ('relu1', torch.nn.ReLU()),

            ('conv2', torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn2', torch.nn.BatchNorm2d(2 * width)),
            ('relu2', torch.nn.ReLU()),

            ('conv3', torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn3', torch.nn.BatchNorm2d(4 * width)),
            ('relu3', torch.nn.ReLU()),

            ('conv4', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn4', torch.nn.BatchNorm2d(4 * width)),
            ('relu4', torch.nn.ReLU()),

            ('conv5', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn5', torch.nn.BatchNorm2d(4 * width)),
            ('relu5', torch.nn.ReLU()),

            ('pool0', torch.nn.MaxPool2d(3)),

            ('conv6', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', torch.nn.BatchNorm2d(4 * width)),
            ('relu6', torch.nn.ReLU()),

            ('conv7', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn7', torch.nn.BatchNorm2d(4 * width)),
            ('relu7', torch.nn.ReLU()),

            ('pool1', torch.nn.MaxPool2d(3)),

            ('flatten', torch.nn.Flatten()),
            ('linear', torch.nn.Linear(36 * width, num_classes))
        ]))

    def forward(self, input):
        return self.model(input)


def resnet_picker(arch, dataset, in_channels, num_classes):
    """Pick an appropriate resnet architecture for MNIST/CIFAR."""

    if 'CIFAR10' in dataset or 'MNIST' in dataset:
        initial_conv = [3, 1, 1]
    else:
        initial_conv = [7, 2, 3]

    if arch == 'ResNet20':
        return ResNet(BasicBlock, [3, 3, 3], in_channels=in_channels, num_classes=num_classes, base_width=16, initial_conv=initial_conv)
    elif 'ResNet20-' in arch and arch[-1].isdigit():
        width_factor = int(arch[-1])
        return ResNet(BasicBlock, [3, 3, 3], in_channels=in_channels, num_classes=num_classes, base_width=16 * width_factor, initial_conv=initial_conv)
    elif arch == 'ResNet28-10':
        return ResNet(BasicBlock, [4, 4, 4], in_channels=in_channels, num_classes=num_classes, base_width=16 * 10, initial_conv=initial_conv)
    elif arch == 'ResNet32':
        return ResNet(BasicBlock, [5, 5, 5], in_channels=in_channels, num_classes=num_classes, base_width=16, initial_conv=initial_conv)
    elif arch == 'ResNet32-10':
        return ResNet(BasicBlock, [5, 5, 5], in_channels=in_channels, num_classes=num_classes, base_width=16 * 10, initial_conv=initial_conv)
    elif arch == 'ResNet44':
        return ResNet(BasicBlock, [7, 7, 7], in_channels=in_channels, num_classes=num_classes, base_width=16, initial_conv=initial_conv)
    elif arch == 'ResNet56':
        return ResNet(BasicBlock, [9, 9, 9], in_channels=in_channels, num_classes=num_classes, base_width=16, initial_conv=initial_conv)
    elif arch == 'ResNet110':
        return ResNet(BasicBlock, [18, 18, 18], in_channels=in_channels, num_classes=num_classes, base_width=16, initial_conv=initial_conv)
    elif arch == 'ResNet18':
        return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes, base_width=64, initial_conv=initial_conv)
    elif 'ResNet18-' in arch:  # this breaks the usual notation, but is nicer for now!!
        new_width = int(arch.split('-')[1])
        return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes, base_width=new_width, initial_conv=initial_conv)
    elif arch == 'ResNet34':
        return ResNet(BasicBlock, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes, base_width=64, initial_conv=initial_conv)
    elif arch == 'ResNet50':
        return ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes, base_width=64, initial_conv=initial_conv)
    elif arch == 'ResNet101':
        return ResNet(Bottleneck, [3, 4, 23, 3], in_channels=in_channels, num_classes=num_classes, base_width=64, initial_conv=initial_conv)
    elif arch == 'ResNet152':
        return ResNet(Bottleneck, [3, 8, 36, 3], in_channels=in_channels, num_classes=num_classes, base_width=64, initial_conv=initial_conv)
    else:
        raise ValueError(f'Invalid ResNet [{dataset}] model chosen: {arch}.')


class ResNet(torchvision.models.ResNet):
    """ResNet generalization for CIFAR-like thingies.

    This is a minor modification of
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py,
    adding additional options.
    """

    def __init__(self, block, layers, in_channels=3, num_classes=10, zero_init_residual=False,
                 groups=1, base_width=64, replace_stride_with_dilation=[False, False, False, False],
                 norm_layer=torch.nn.BatchNorm2d, strides=[1, 2, 2, 2], initial_conv=[3, 1, 1]):
        """Initialize as usual. Layers and strides are scriptable."""
        super(torchvision.models.ResNet, self).__init__()  # torch.nn.Module
        self._norm_layer = norm_layer

        self.dilation = 1
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups

        self.inplanes = base_width
        self.base_width = 64  # Do this to circumvent BasicBlock errors. The value is not actually used.
        self.conv1 = torch.nn.Conv2d(in_channels, self.inplanes, kernel_size=initial_conv[0],
                                     stride=initial_conv[1], padding=initial_conv[2], bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)

        layer_list = []
        width = self.inplanes
        for idx, layer in enumerate(layers):
            layer_list.append(self._make_layer(block, width, layer, stride=strides[idx], dilate=replace_stride_with_dilation[idx]))
            width *= 2
        self.layers = torch.nn.Sequential(*layer_list)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(width // 2 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the arch by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    torch.nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    torch.nn.init.constant_(m.bn2.weight, 0)


    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layers(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
