"""Write a PyTorch dataset into RAM."""

import torch

from torchvision.datasets import VisionDataset
from PIL import Image
from pathlib import Path

from .consts import PIN_MEMORY, NON_BLOCKING

class Subset(torch.utils.data.Subset):
    """Overwrite subset class to provide class methods of main class."""

    def get_target(self, index):
        return getattr(self.dataset, 'get_target')(self.indices[index])
    
    def get_annotation(self, index):
        return getattr(self.dataset, 'get_annotation')(self.indices[index])

    def __getattr__(self, name):
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)
    

class TinyDataset(VisionDataset):
    def __init__(self, root, target, annotations=None, transform=None, target_transform=None, classes=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.root = Path(root)
        self.images = list(self.root.rglob('*'))

        self.data = dict()
        self.classes = classes
        self.target = target
        self.annotations = annotations
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        if isinstance(self.target, list):
            target = self.target[index]
        else:
            target = self.target
        
        if index not in self.data:
            self.data[index] = Image.open(self.images[index])

        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
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
        if isinstance(self.target, list):
            target = self.target[index]
        else:
            target = self.target

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index
    
    def get_annotation(self, index):
        if self.annotations is None:
            annotation = self.get_target(index)[0]
        else:
            annotation = self.annotations[index]

        return annotation, index

    def change_transform(self, transform):
        self.transform = transform
    

class PureImageDataset(VisionDataset):
    def __init__(self, images, targets, annotations, transform=None, target_transform=None):
        super().__init__('', transform=transform, target_transform=target_transform)
        self.images = images
        self.targets = targets
        self.annotations = annotations
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        target = self.targets[index]
        img = self.images[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

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


class FeatureDataset(torch.utils.data.Dataset):
    """Cache a given dataset."""

    @torch.no_grad()
    def __init__(self, kettle, dataset, feature_extractor, setup, poison_delta=None, num_workers=200):
        """Initialize with a given pytorch dataset."""
        self.dataset = dataset
        self.cache = []
        batch_size = max(1, min(len(dataset) // max(num_workers, 1), 32))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, drop_last=False, num_workers=num_workers,
                                             pin_memory=False)

        if feature_extractor is None:
            self.cache = None
            return

        # Allocate memory:
        
        temp_input = self.dataset[0][0].to(**setup)
        temp_input.unsqueeze_(0)
        temp_output = feature_extractor(temp_input)
        temp_output.squeeze_(0)
        self.cache = torch.empty((len(self.dataset), *temp_output.shape), pin_memory=PIN_MEMORY)

        pointer = 0
        for inputs, labels, ids in loader:
            batch_length = len(inputs)
            inputs = inputs.to(**kettle.setup)
            labels = labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)

            if poison_delta is not None:
                poison_slices, batch_positions = [], []
                for batch_id, image_id in enumerate(ids.tolist()):
                    lookup = kettle.poison_lookup.get(image_id)
                    if lookup is not None:
                        poison_slices.append(lookup)
                        batch_positions.append(batch_id)
                # Python 3.8:
                # twins = [(b, l) for b, i in enumerate(ids.tolist()) if l:= kettle.poison_lookup.get(i)]
                # poison_slices, batch_positions = zip(*twins)

                if batch_positions:
                    inputs[batch_positions] += poison_delta[poison_slices].to(**kettle.setup)
            
            feature = feature_extractor(inputs)
            self.cache[pointer: pointer + batch_length] = feature.cpu()
            pointer += batch_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.cache is None:
            return self.dataset[index]
        
        sample = self.cache[index]
        target, index = self.dataset.get_target(index)
        
        return sample, target, index

    def get_target(self, index):
        return self.dataset.get_target(index)
    

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensors, targets):
        self.tensors = tensors.detach().cpu()
        self.targets = targets
    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.tensors[index], self.targets[index], index

    def get_target(self, index):
        return self.targets[index], index