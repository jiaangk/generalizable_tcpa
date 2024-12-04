"""Super-classes of common datasets to extract id information per image."""
import torch
import torchvision
import json
from pathlib import Path
from torchvision.datasets.folder import pil_loader

from ..consts import *   # import all mean/std constants
from ..transforms import transform_picker

import torchvision.transforms as transforms
from PIL import Image
from ..efficient import TinyDataset
import random

from .cifar import CIFAR10, CIFAR100, CIFAR100_20

# Block ImageNet corrupt EXIF warnings
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class MNIST(torchvision.datasets.MNIST):
    """Super-class MNIST to return image ids with images."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.annotations = self.targets
        self.subpopulations = self.classes

    def __getitem__(self, index):
        """_getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html#MNIST.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.

        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = int(self.targets[index])

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index
    
    def get_annotation(self, index):
        annotation = self.annotations[index]

        return annotation, index
    
    def change_transform(self, transform):
        self.transform = transform

SELECTED_CLASSES = list(range(1000))

class ImageNet(torchvision.datasets.ImageNet):
    """Overwrite torchvision ImageNet to limit it to less than 1mio examples.

    [limit/per class, due to automl restrictions].
    """

    def __init__(self, root, train=True, class_limit=10, img_limit=1000, transform=None, **kwargs):
        """As torchvision.datasets.ImageNet except for additional keyword 'limit'."""
        split = 'train' if train else 'val'
        super().__init__(root, split, transform=transform)

        # Dictionary, mapping ImageNet1k ids to ImageNet ids:
        self.full_imagenet_id = dict()
        # Remove samples above limit.
        selected_classes = SELECTED_CLASSES[:class_limit]
        idx_transform = {j: i for i, j in enumerate(selected_classes)}
        examples_per_class = [0 for _ in range(class_limit)]
        new_imgs = []
        new_idx = 0

        if img_limit is None:
            img_limit = len(self.imgs)

        for full_idx, (path, target) in enumerate(self.imgs):
            if target in selected_classes:
                target = idx_transform[target]
                if examples_per_class[target] < img_limit:
                    examples_per_class[target] += 1
                    item = path, target
                    new_imgs.append(item)
                    self.full_imagenet_id[new_idx] = full_idx
                    new_idx += 1

        self.imgs = new_imgs
        self.targets = [target for path, target in self.imgs]

        self.old_classes = self.classes
        self.classes = [self.old_classes[i][0] for i in selected_classes]
        self.selected_classes = selected_classes
        print(f'Size of {self.split} dataset reduced to {len(self.imgs)}.')

        self.subpopulations = self.classes
        self.annotations = [j for i, j in self.imgs]
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

    def get_target(self, index):
        _, target = self.imgs[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index

    def get_annotation(self, index):
        annotation = self.annotations[index]

        return annotation, index
    
    def change_transform(self, transform):
        self.transform = transform

class CelebA(torchvision.datasets.VisionDataset):
    def __init__(self, root, train, download=False, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.root = Path(root)
        self.imgs = []

        with open(self.root / 'list_eval_partition.txt') as f:
            while True:
                s = f.readline()
                if len(s) > 0:
                    match train:
                        case True:
                            if s.split()[1] == '0':
                                self.imgs.append(s.split()[0])
                        case False:
                            if s.split()[1] != '0':
                                self.imgs.append(s.split()[0])
                else:
                    break
        
        imgs_set = set(self.imgs)
        self.targets = []
        
        with open(self.root / 'list_attr_celeba.txt') as f:
            f.readline()
            f.readline()

            while True:
                s = f.readline()
                if len(s) > 0:
                    infos = s.split()
                    if infos[0] in imgs_set and infos[21] == '-1':
                        self.targets.append(0)
                    if infos[0] in imgs_set and infos[21] == '1':
                        self.targets.append(1)
                else:
                    break
        
        self.classes = ['female', 'male']

        self.annotations = self.targets.copy()
        self.subpopulations = self.classes.copy()
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path = self.root / self.imgs[index]
        target = self.targets[index]

        sample = pil_loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

    def get_target(self, index):
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index

    def get_annotation(self, index):
        annotation = self.annotations[index]
        
        return annotation, index
    
    def change_transform(self, transform):
        self.transform = transform

DATASETS = {
    'CIFAR10': CIFAR10,
    'CIFAR100': CIFAR100,
    'CIFAR100_20': CIFAR100_20,
    'MNIST': MNIST,
    'ImageNet': ImageNet,
    'CelebA': CelebA
}

def compute_mean_std(trainset, channels):
    """Compute mean and standard deviation for a dataset."""
    if channels == 1:
        cc = torch.cat([torch.mean(trainset[i][0].reshape(-1), dim=0) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        cc = torch.stack([torch.mean(trainset[i][0].reshape(3, -1), dim=1) for i in range(len(trainset))], dim=1)
        data_mean = torch.mean(cc, dim=1).tolist()
        data_std = torch.std(cc, dim=1).tolist()
        
    return data_mean, data_std

def construct_custom_datasets(dataset, data_path, args, pretrained=None, normalize=True):
    with open(data_path / 'config.json') as f:
        infos = json.load(f)

    trainset = TinyDataset(data_path / 'train', infos['train_targets'], infos['train_annotations'], transform=transforms.ToTensor(),
                           classes=infos['classes'])
    
    if pretrained:
        if 'clip' == pretrained:
            data_mean = clip_mean
            data_std = clip_std
        else:
            data_mean = imagenet_mean
            data_std = imagenet_std
    elif 'imagenet' in dataset.lower():
        data_mean = imagenet_mean
        data_std = imagenet_std
    else:
        if normalize:
            data_mean, data_std = compute_mean_std(trainset, 3)
            print(f'Data mean is {data_mean}, \nData std  is {data_std}.')
        else:
            data_mean, data_std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
            print('Normalization disabled.')

    # Define transforms
    if pretrained:
        transform = transform_picker(pretrained, args)
    else:
        common_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)
        ]
        if not DATASET_SETTING[dataset]['resolution_is_fix']:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                *common_transforms
            ])
        else:
            transform = transforms.Compose(common_transforms)
        
    # Update trainset transform
    trainset.transform = transform

    # Create validation set
    validset = TinyDataset(data_path / 'valid', infos['valid_targets'], infos['valid_annotations'], transform=transform,
                           classes=infos['classes'])

    # Assign mean and std to the datasets
    for ds in (trainset, validset):
        ds.data_mean = data_mean
        ds.data_std = data_std

    return trainset, validset

def construct_datasets(dataset, data_path, args, pretrained=None, normalize=True):
    """Construct datasets with appropriate transforms."""
    if dataset == 'custom':
        return construct_custom_datasets(dataset, Path(data_path), args, pretrained, normalize)

    if dataset not in DATASETS:
        raise ValueError(f'Invalid dataset {dataset} given.')

    if dataset == 'ImageNet':
        global SELECTED_CLASSES
        random.shuffle(SELECTED_CLASSES)

    dataset_class = DATASETS[dataset]
    channels = DATASET_SETTING[dataset]['in_channels']

    # Compute mean and std if not provided
    trainset = dataset_class(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    
    if pretrained:
        if 'clip' == pretrained:
            data_mean = clip_mean
            data_std = clip_std
        else:
            data_mean = imagenet_mean
            data_std = imagenet_std
    elif 'imagenet' in dataset.lower() or 'celeba' in dataset.lower():
        data_mean = imagenet_mean
        data_std = imagenet_std
    else:
        if normalize:
            data_mean, data_std = compute_mean_std(trainset, channels)
            print(f'Data mean is {data_mean}, \nData std  is {data_std}.')
        else:
            data_mean, data_std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
            print('Normalization disabled.')

    # Define transforms
    if pretrained:
        transform = transform_picker(pretrained, args)
    else:
        common_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)
        ]
        if not DATASET_SETTING[dataset]['resolution_is_fix']:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                *common_transforms
            ])
        else:
            transform = transforms.Compose(common_transforms)
        
    # Update trainset transform
    trainset.transform = transform

    # Create validation set
    validset = dataset_class(root=data_path, train=False, download=True, transform=transform)

    # Assign mean and std to the datasets
    for ds in (trainset, validset):
        ds.data_mean = data_mean
        ds.data_std = data_std

    return trainset, validset