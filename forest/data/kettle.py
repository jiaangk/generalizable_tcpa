"""Data class, holding information about dataloaders and poison ids."""

import torch
import numpy as np
from io import BytesIO

import pickle
import json

import datetime
import os
import warnings
import random
import PIL
import numpy as np

from pathlib import Path
from sklearn.cluster import KMeans
from argparse import Namespace
import torchvision.transforms as transforms

from .datasets import construct_datasets
from ..efficient import Subset, FeatureDataset, TensorDataset, TinyDataset, PureImageDataset

from .diff_data_augmentation import RandomTransform
from ..transforms import transform_picker

from ..consts import PIN_MEMORY, BENCHMARK, DISTRIBUTED_BACKEND, SHARING_STRATEGY
from ..utils import get_num_workers
from ..pretrained_models import pretrained_picker
torch.backends.cudnn.benchmark = BENCHMARK
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


class Kettle():
    """Brew poison with given arguments.

    Data class.
    Attributes:
    - trainloader
    - validloader
    - poisonloader
    - poison_ids
    - trainset/poisonset/targetset

    Most notably .poison_lookup is a dictionary that maps image ids to their slice in the poison_delta tensor.

    Initializing this class will set up all necessary attributes.

    Other data-related methods of this class:
    - initialize_poison
    - export_poison

    """

    def __init__(self, args, batch_size, augmentations, feature_extractor, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize with given specs..."""
        self.args, self.setup = args, setup
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.num_workers = get_num_workers()
        self.trainset, self.validset = self.prepare_data(normalize=True)
        self.feature_extractor = feature_extractor

        if self.args.subpopulation == 'cluster':
            self._dataset_clustering()
        if self.args.subpopulation == 'external_annotation':
            self._load_annotation()

        if self.args.subpopulation == 'external':
            self.external_construction()
        else:
            self.random_construction()

        self.datasets = Namespace(trainset=self.trainset, validset=self.validset,
                                  targetset=self.targetset, poisonset=self.poisonset,
                                  target_validset=self.target_validset)

        self.dataloader_construction()
        # Ablation on a subset?
        if args.ablation < 1.0:
            self.sample = random.sample(range(len(self.trainset)), int(self.args.ablation * len(self.trainset)))
            self.partialset = Subset(self.trainset, self.sample)
            self.partialloader = torch.utils.data.DataLoader(self.partialset, batch_size=min(self.batch_size, len(self.partialset)),
                                                             shuffle=True, drop_last=False, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
        self.print_status()


    """ STATUS METHODS """
    def print_status(self):
        class_names = self.datasets.trainset.classes
        print(
            f'Poisoning setup generated for threat model {self.args.threatmodel} and '
            f'budget of {self.args.budget * 100}% - {len(self.datasets.poisonset)} images:')
        print(f'--Target images drawn from class {class_names[self.datasets.targetset[0][1]]}.')
        print(f'--Target images assigned intended class {class_names[self.poison_setup["intended_class"]]}.')
        print(f'--Target validset contains {len(self.target_validset)} samples.')

        if self.poison_setup["poison_class"] is not None:
            print(f'--Poison images drawn from class {class_names[self.poison_setup["poison_class"]]}.')
        else:
            print(f'--Poison images drawn from all classes.')

        if self.args.ablation < 1.0:
            print(f'--Partialset is {len(self.partialset)/len(self.datasets.trainset):2.2%} of full training set')
            num_p_poisons = len(np.intersect1d(self.poison_ids.cpu().numpy(), np.array(self.sample)))
            print(f'--Poisons in partialset are {num_p_poisons} ({num_p_poisons/len(self.poison_ids):2.2%})')

    """ CONSTRUCTION METHODS """
    def dataloader_construction(self):
        # Generate loaders:
        if self.feature_extractor is None:
            self.validset = self.datasets.validset
        else:
            self.validset = FeatureDataset(self, self.datasets.validset, self.feature_extractor, self.setup, num_workers=self.num_workers)
        self.validloader = torch.utils.data.DataLoader(self.validset, batch_size=min(self.batch_size, len(self.validset)),
                                                       shuffle=False, drop_last=False, num_workers=self.num_workers, pin_memory=PIN_MEMORY)

        if self.feature_extractor is None:
            self.targetset = self.datasets.targetset
        else:
            self.targetset = FeatureDataset(self, self.datasets.targetset, self.feature_extractor, self.setup, num_workers=self.num_workers)

        if self.feature_extractor is None:
            self.target_validset = self.datasets.target_validset
        else:
            self.target_validset = FeatureDataset(self, self.datasets.target_validset, self.feature_extractor, self.setup, num_workers=self.num_workers)
        self.target_validloader = torch.utils.data.DataLoader(self.target_validset, batch_size=min(self.batch_size, len(self.target_validset)),
                                                              shuffle=False, drop_last=False, num_workers=self.num_workers, pin_memory=PIN_MEMORY)

        validated_batch_size = max(min(self.args.pbatch, len(self.poisonset)), 1)
        self.poisonloader = torch.utils.data.DataLoader(self.poisonset, batch_size=validated_batch_size,
                                                        shuffle=False, drop_last=False, num_workers=self.num_workers,
                                                        pin_memory=PIN_MEMORY)
                                                        
        self.trainset_feature_reconstruction()

    def trainset_feature_reconstruction(self, poison_delta=None):
        if self.augmentations or self.feature_extractor is None:
            self.trainset = self.datasets.trainset
        else:
            self.trainset = FeatureDataset(self, self.datasets.trainset, self.feature_extractor, self.setup, poison_delta=poison_delta, num_workers=self.num_workers)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=min(self.batch_size, len(self.trainset)),
                                                       shuffle=True, drop_last=False, num_workers=self.num_workers, pin_memory=PIN_MEMORY)

    def prepare_data(self, normalize=True):
        trainset, validset = construct_datasets(self.args.dataset, self.args.data_path, self.args, self.args.pretrained, normalize)

        # Prepare data mean and std for later:
        self.dm = torch.tensor(trainset.data_mean)[None, :, None, None].to(**self.setup)
        self.ds = torch.tensor(trainset.data_std)[None, :, None, None].to(**self.setup)

        # Train augmentations are handled separately as they possibly have to be backpropagated
        if self.args.pretrained is not None:
            params = dict(source_size=224, target_size=224, shift=224 // 4, fliplr=True)
        elif 'CIFAR' in self.args.dataset:
            params = dict(source_size=32, target_size=32, shift=8, fliplr=True)
        elif 'MNIST' in self.args.dataset:
            params = dict(source_size=28, target_size=28, shift=4, fliplr=True)
        elif 'TinyImageNet' in self.args.dataset:
            params = dict(source_size=64, target_size=64, shift=64 // 4, fliplr=True)
        elif 'ImageNet' in self.args.dataset:
            params = dict(source_size=224, target_size=224, shift=224 // 4, fliplr=True)
        elif 'CelebA' in self.args.dataset:
            params = dict(source_size=224, target_size=224, shift=224 // 4, fliplr=True)

        self.augment = RandomTransform(**params, mode='bilinear', setup=self.setup)

        return trainset, validset

    def deterministic_construction(self):
        """Construct according to the triplet input key.

        The triplet key, e.g. 5-3-1 denotes in order:
        target_class - poison_class - target_id

        Poisons are always the first n occurences of the given class.
        [This is the same setup as in metapoison]
        """
        if self.args.threatmodel != 'single-class':
            raise NotImplementedError()

        split = self.args.poisonkey.split('-')
        if len(split) != 3:
            raise ValueError('Invalid poison triplet supplied.')
        else:
            target_class, poison_class, target_id = [int(s) for s in split]
        self.init_seed = self.args.poisonkey
        print(f'Initializing Poison data (chosen images, examples, targets, labels) as {self.args.poisonkey}')

        self.poison_setup = dict(poison_budget=self.args.budget,
                                 target_num=self.args.targets, poison_class=poison_class, target_class=target_class,
                                 intended_class=[poison_class])
        self.poisonset, self.targetset, self.validset = self._choose_poisons_deterministic(target_id)

    def benchmark_construction(self, setup_dict):
        """Construct according to the benchmark."""
        target_class, poison_class = setup_dict['target class'], setup_dict['base class']

        budget = len(setup_dict['base indices']) / len(self.trainset)
        self.poison_setup = dict(poison_budget=budget,
                                 target_num=self.args.targets, poison_class=poison_class, target_class=target_class,
                                 intended_class=[poison_class])
        self.init_seed = self.args.poisonkey
        self.poisonset, self.targetset, self.validset = self._choose_poisons_benchmark(setup_dict)

    def _choose_poisons_benchmark(self, setup_dict):
        # poisons
        class_ids = setup_dict['base indices']
        poison_num = len(class_ids)
        self.poison_ids = class_ids

        # the target
        self.target_ids = [setup_dict['target index']]
        # self.target_ids = setup_dict['target index']

        targetset = Subset(self.validset, indices=self.target_ids)
        valid_indices = []
        for index in range(len(self.validset)):
            _, idx = self.validset.get_target(index)
            if idx not in self.target_ids:
                valid_indices.append(idx)
        validset = Subset(self.validset, indices=valid_indices)
        poisonset = Subset(self.trainset, indices=self.poison_ids)

        # Construct lookup table
        self.poison_lookup = dict(zip(self.poison_ids, range(poison_num)))

        return poisonset, targetset, validset

    def _choose_poisons_deterministic(self, target_id):
        # poisons
        class_ids = []
        for index in range(len(self.trainset)):  # we actually iterate this way not to iterate over the images
            target, idx = self.trainset.get_target(index)
            if target == self.poison_setup['poison_class']:
                class_ids.append(idx)

        poison_num = int(np.ceil(self.args.budget * len(self.trainset)))
        if len(class_ids) < poison_num:
            warnings.warn(f'Training set is too small for requested poison budget.')
            poison_num = len(class_ids)
        self.poison_ids = class_ids[:poison_num]

        # the target
        # class_ids = []
        # for index in range(len(self.validset)):  # we actually iterate this way not to iterate over the images
        #     target, idx = self.validset.get_target(index)
        #     if target == self.poison_setup['target_class']:
        #         class_ids.append(idx)
        # self.target_ids = [class_ids[target_id]]
        # Disable for now for benchmark sanity check. This is a breaking change.
        self.target_ids = [target_id]

        targetset = Subset(self.validset, indices=self.target_ids)
        valid_indices = []
        for index in range(len(self.validset)):
            _, idx = self.validset.get_target(index)
            if idx not in self.target_ids:
                valid_indices.append(idx)
        validset = Subset(self.validset, indices=valid_indices)
        poisonset = Subset(self.trainset, indices=self.poison_ids)

        # Construct lookup table
        self.poison_lookup = dict(zip(self.poison_ids, range(poison_num)))
        dict(zip(self.poison_ids, range(poison_num)))
        return poisonset, targetset, validset
    
    def external_construction(self):
        """Construct according to external information.

        The setup can be repeated from its key (which initializes the random generator).
        This method sets
         - poison_setup
         - poisonset / targetset / validset

        """
        dataset_path = Path(self.args.external_path)

        with open(dataset_path / 'config.json') as f:
            data = json.load(f)
            target_class = data['target_class']
            
            # intended_class = data['intended_class']
            num_classes = len(self.trainset.classes)
            list_intentions = list(range(num_classes))
            list_intentions.remove(target_class)
            intended_class = np.random.choice(list_intentions)

            if self.args.subpopulation_target is not None:
                subpopulation = self.args.subpopulation_target
            else:
                subpopulation = np.random.randint(data['num'])
            
        self.poison_setup = dict(poison_budget=self.args.budget, poison_class=intended_class, target_class=target_class, intended_class=intended_class,
                                 target_subpopulation=subpopulation)

        # Construct poison set
        class_ids = []
        for index in range(len(self.trainset)):  # we actually iterate this way not to iterate over the images
            target, idx = self.trainset.get_target(index)
            if target == self.poison_setup['poison_class']:
                class_ids.append(idx)

        poison_num = int(np.ceil(self.args.budget * len(self.trainset)))
        if len(class_ids) < poison_num:
            warnings.warn(f'Training set is too small for requested poison budget. \n'
                            f'Budget will be reduced to maximal size {len(class_ids)}')
            poison_num = len(class_ids)
        self.poison_ids = torch.tensor(np.random.choice(
            class_ids, size=poison_num, replace=False), dtype=torch.long)
        
        self.poisonset = Subset(self.trainset, indices=self.poison_ids)

        # Construct lookup table
        self.poison_lookup = dict(zip(self.poison_ids.tolist(), range(poison_num)))
        self.poison_origin_img = torch.stack([self.trainset[idx][0] for idx in self.poison_ids], dim=0)

        # Construct targetset
        self.targetset = TinyDataset(dataset_path / ('target/%d' % subpopulation), target_class, transform=self.trainset.transform)
        self.target_validset = TinyDataset(dataset_path / ('valid/%d' % subpopulation), target_class, transform=self.trainset.transform)

        self.args.targets = len(self.targetset)

    def random_construction(self):
        """Construct according to random selection.

        The setup can be repeated from its key (which initializes the random generator).
        This method sets
         - poison_setup
         - poisonset / targetset / validset

        """
        # Parse threat model
        self.poison_setup = self._parse_threats_randomly()
        self.poisonset, self.targetset, self.validset, self.target_validset = self._choose_poisons_randomly()

    def _parse_threats_randomly(self):
        """Parse the different threat models.

        The threat-models are [In order of expected difficulty]:

        single-class replicates the threat model of feature collision attacks,
        third-party draws all poisons from a class that is unrelated to both target and intended label.
        random-subset draws poison images from all classes.
        random-subset draw poison images from all classes and draws targets from different classes to which it assigns
        different labels.
        """
        num_classes = len(self.trainset.classes)

        target_class = np.random.randint(num_classes)
        if self.args.assign_target is not None:
            target_class = self.args.assign_target
        list_intentions = list(range(num_classes))
        list_intentions.remove(target_class)
        intended_class = np.random.choice(list_intentions)
        
        annotation_ids = set()
        n = len(self.validset)
        for i in range(n):
            target, idx = self.validset.get_target(i)
            if target == target_class:
                annotation, _ = self.validset.get_annotation(i)
                annotation_ids.add(annotation)
        
        target_subpopulation = np.random.choice(list(annotation_ids))

        if self.args.targets < 1:
            poison_setup = dict(poison_budget=0, target_num=0,
                                poison_class=np.random.randint(num_classes), target_class=None,
                                intended_class=[np.random.randint(num_classes)])
            warnings.warn('Number of targets set to 0.')
            return poison_setup

        if self.args.threatmodel == 'single-class':
            poison_class = intended_class
            poison_setup = dict(poison_budget=self.args.budget, target_num=self.args.targets,
                                poison_class=poison_class, target_class=target_class, intended_class=intended_class,
                                target_subpopulation=target_subpopulation)
        elif self.args.threatmodel == 'random-subset':
            poison_class = None
            intended_class = None
            poison_setup = dict(poison_budget=self.args.budget,
                                target_num=self.args.targets, poison_class=None, target_class=target_class,
                                intended_class=intended_class, target_subpopulation=target_subpopulation)
        else:
            raise NotImplementedError('Unknown threat model.')

        return poison_setup

    def _choose_poisons_randomly(self):
        """Subconstruct poison and targets.

        The behavior is different for poisons and targets. We still consider poisons to be part of the original training
        set and load them via trainloader (And then add the adversarial pattern Delta)
        The targets are fully removed from the validation set and returned as a separate dataset, indicating that they
        should not be considered during clean validation using the validloader

        """
        # Poisons:
        if self.poison_setup['poison_class'] is not None:
            class_ids = []
            for index in range(len(self.trainset)):  # we actually iterate this way not to iterate over the images
                target, idx = self.trainset.get_target(index)
                if target == self.poison_setup['poison_class']:
                    class_ids.append(idx)

            poison_num = int(np.ceil(self.args.budget * len(self.trainset)))
            if len(class_ids) < poison_num:
                warnings.warn(f'Training set is too small for requested poison budget. \n'
                              f'Budget will be reduced to maximal size {len(class_ids)}')
                poison_num = len(class_ids)
            self.poison_ids = torch.tensor(np.random.choice(
                class_ids, size=poison_num, replace=False), dtype=torch.long)
        else:
            total_ids = []
            for index in range(len(self.trainset)):  # we actually iterate this way not to iterate over the images
                _, idx = self.trainset.get_target(index)
                total_ids.append(idx)
            poison_num = int(np.ceil(self.args.budget * len(self.trainset)))
            if len(total_ids) < poison_num:
                warnings.warn(f'Training set is too small for requested poison budget. \n'
                              f'Budget will be reduced to maximal size {len(total_ids)}')
                poison_num = len(total_ids)
            self.poison_ids = torch.tensor(np.random.choice(
                total_ids, size=poison_num, replace=False), dtype=torch.long)

        # Targets:
        if self.args.valid_knowledge:
            dataset = self.validset
        else:
            dataset = self.trainset

        class_ids = []
        for index in range(len(dataset)):  # we actually iterate this way not to iterate over the images
            target, idx = dataset.get_target(index)
            annotation, _ = dataset.get_annotation(index)
            if target == self.poison_setup['target_class'] and annotation == self.poison_setup['target_subpopulation']:
                class_ids.append(idx)

        if len(class_ids) < self.args.targets:
            warnings.warn(f'Training set is too small for requested targets. \n'
                          f'Targets will be reduced to maximal size {len(class_ids)}')
            self.args.targets = len(class_ids)

        self.cluster_ids = class_ids

        self.target_ids = np.random.choice(class_ids, size=self.args.targets, replace=False)
        targetset = Subset(dataset, indices=self.target_ids)

        # Targets valid set:
        class_ids = []
        for index in range(len(self.validset)):
            target, idx = self.validset.get_target(index)
            annotation, _ = self.validset.get_annotation(index)
            if target == self.poison_setup['target_class'] and annotation == self.poison_setup['target_subpopulation']:
                class_ids.append(idx)
        target_validset = Subset(self.validset, indices=class_ids)

        valid_indices = [idx for idx in range(len(self.validset)) if idx not in class_ids]

        validset = Subset(self.validset, indices=valid_indices)
        poisonset = Subset(self.trainset, indices=self.poison_ids)

        # Construct lookup table
        self.poison_lookup = dict(zip(self.poison_ids.tolist(), range(poison_num)))
        self.poison_origin_img = torch.stack([self.trainset[idx][0] for idx in self.poison_ids], dim=0)
        
        return poisonset, targetset, validset, target_validset

    def initialize_poison(self, initializer=None):
        """Initialize according to args.init.

        Propagate initialization in distributed settings.
        """
        if initializer is None:
            initializer = self.args.init

        # ds has to be placed on the default (cpu) device, not like self.ds
        ds = self.ds
        if initializer == 'zero':
            init = torch.zeros(len(self.poison_ids), *self.poisonset[0][0].shape)
            init = init.to(**self.setup)
        elif initializer == 'rand':
            init = (torch.rand(len(self.poison_ids), *self.poisonset[0][0].shape) - 0.5) * 2
            init = init.to(**self.setup)
            init *= self.args.eps / ds / 255
        elif initializer == 'randn':
            init = torch.randn(len(self.poison_ids), *self.poisonset[0][0].shape)
            init = init.to(**self.setup)
            init *= self.args.eps / ds / 255
        elif initializer == 'normal':
            init = torch.randn(len(self.poison_ids), *self.poisonset[0][0].shape)
            init = init.to(**self.setup)
        else:
            raise NotImplementedError()

        init.data = torch.max(torch.min(init, self.args.eps / ds / 255), -self.args.eps / ds / 255)

        return init

    def realistic_process(self, poison_delta):
        dm = torch.tensor(self.datasets.trainset.data_mean)[:, None, None]
        ds = torch.tensor(self.datasets.trainset.data_std)[:, None, None]

        def _torch_to_PIL(image_tensor):
            """Torch->PIL pipeline as in torchvision.utils.save_image."""
            image_denormalized = torch.clamp(image_tensor * ds + dm, 0, 1)
            image_torch_uint8 = image_denormalized.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8)
            image_PIL = PIL.Image.fromarray(image_torch_uint8.numpy())
            return image_PIL

        def _jpeg(image, rate=10):
            outputIoStream = BytesIO()
            image.save(outputIoStream, "JPEG", quality=rate, optimice=True)
            outputIoStream.seek(0)
            return PIL.Image.open(outputIoStream)

        def _save_image(input, idx, train=True):
            """Save input image to given location, add poison_delta if necessary."""
            lookup = self.poison_lookup.get(idx)
            if (lookup is not None) and train:
                input += poison_delta[lookup, :, :, :]
            image = _torch_to_PIL(input)

            if self.args.defense == 'jpeg':
                image = _jpeg(image)
            return image

        def incorperate_dataset(dataset, train):
            images = []
            labels = []
            annotations = []        
            for input, label, idx in self.datasets.trainset:
                annotation, _ = self.datasets.trainset.get_annotation(idx)
                image = _save_image(input, idx, train=train)
                images.append(image)
                labels.append(label)
                annotations.append(annotation)
            dataset = PureImageDataset(images, labels, annotations, dataset.transform)
            return dataset

        trainset = incorperate_dataset(self.datasets.trainset, train=True)
        print("Finished realistic preprocess.")

        if self.args.defense == 'bdr':
            trainset.transform = transforms.Compose([transforms.RandomPosterize(bits=2, p=1), trainset.transform])
        if self.args.defense == 'gaussian':
            trainset.transform = transforms.Compose([transforms.GaussianBlur(3, sigma=0.1), trainset.transform])
        self.datasets.trainset = trainset
        
        self.dataloader_construction()

    """ EXPORT METHODS """

    def export_poison(self, poison_delta, path=None, mode='automl'):
        """Export poisons in either packed mode (just ids and raw data) or in full export mode, exporting all images.

        In full export mode, export data into folder structure that can be read by a torchvision.datasets.ImageFolder

        In automl export mode, export data into a single folder and produce a csv file that can be uploaded to
        google storage.
        """
        if path is None:
            path = self.args.poison_path

        dm = torch.tensor(self.datasets.trainset.data_mean)[:, None, None]
        ds = torch.tensor(self.datasets.trainset.data_std)[:, None, None]

        def _torch_to_PIL(image_tensor):
            """Torch->PIL pipeline as in torchvision.utils.save_image."""
            image_denormalized = torch.clamp(image_tensor * ds + dm, 0, 1)
            image_torch_uint8 = image_denormalized.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8)
            image_PIL = PIL.Image.fromarray(image_torch_uint8.numpy())
            return image_PIL

        def _save_image(input, label, idx, location, train=True):
            """Save input image to given location, add poison_delta if necessary."""
            filename = os.path.join(location, str(idx) + '.png')

            lookup = self.poison_lookup.get(idx)
            if (lookup is not None) and train:
                input += poison_delta[lookup, :, :, :]
            _torch_to_PIL(input).save(filename)

        # Save either into packed mode, ImageDataSet Mode or google storage mode
        if mode == 'packed':
            data = dict()
            data['poison_setup'] = self.poison_setup
            data['poison_delta'] = poison_delta
            data['poison_ids'] = self.poison_ids
            data['target_images'] = [data for data in self.datasets.targetset]
            name = f'{path}poisons_packed_{datetime.date.today()}.pth'
            torch.save([poison_delta, self.poison_ids], os.path.join(path, name))

        elif mode == 'limited':
            # Save training set
            names = self.datasets.trainset.classes
            for name in names:
                os.makedirs(os.path.join(path, 'train', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'targets', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'clean', name), exist_ok=True)
            for input, label, idx in self.datasets.trainset:
                lookup = self.poison_lookup.get(idx)
                if lookup is not None:
                    _save_image(input, label, idx, location=os.path.join(path, 'train', names[label]), train=True)
            print('Poisoned training images exported ...')
            
            for input, label, idx in self.datasets.trainset:
                lookup = self.poison_lookup.get(idx)
                if lookup is not None:
                    _save_image(input, label, idx, location=os.path.join(path, 'clean', names[label]), train=False)
            print('Clean training images exported ...')

            # Save secret targets
            for enum, (target, _, idx) in enumerate(self.datasets.targetset):
                intended_class = self.poison_setup['intended_class']
                _save_image(target, intended_class, idx, location=os.path.join(path, 'targets', names[intended_class]), train=False)
                if enum + 1 == len(self.datasets.targetset):
                    break
            print('Target images exported with intended class labels ...')

        elif mode == 'full':
            # Save training set
            names = self.datasets.trainset.classes
            for name in names:
                os.makedirs(os.path.join(path, 'train', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'test', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'targets', name), exist_ok=True)
            for input, label, idx in self.datasets.trainset:
                _save_image(input, label, idx, location=os.path.join(path, 'train', names[label]), train=True)
            print('Poisoned training images exported ...')

            for input, label, idx in self.datasets.validset:
                _save_image(input, label, idx, location=os.path.join(path, 'test', names[label]), train=False)
            print('Unaffected validation images exported ...')

            # Save secret targets
            for enum, (target, _, idx) in enumerate(self.datasets.targetset):
                intended_class = self.poison_setup['intended_class'][enum]
                _save_image(target, intended_class, idx, location=os.path.join(path, 'targets', names[intended_class]), train=False)
            print('Target images exported with intended class labels ...')

        elif mode == 'numpy':
            _, h, w = self.datasets.trainset[0][0].shape
            training_data = np.zeros([len(self.datasets.trainset), h, w, 3])
            labels = np.zeros(len(self.datasets.trainset))
            for input, label, idx in self.datasets.trainset:
                lookup = self.poison_lookup.get(idx)
                if lookup is not None:
                    input += poison_delta[lookup, :, :, :]
                training_data[idx] = np.asarray(_torch_to_PIL(input))
                labels[idx] = label

            np.save(os.path.join(path, 'poisoned_training_data.npy'), training_data)
            np.save(os.path.join(path, 'poisoned_training_labels.npy'), labels)

        elif mode == 'kettle-export':
            with open(f'kette_{self.args.dataset}{self.args.model}.pkl', 'wb') as file:
                pickle.dump([self, poison_delta], file, protocol=pickle.HIGHEST_PROTOCOL)

        elif mode == 'benchmark':
            foldername = f'{self.args.name}_{"_".join(self.args.net)}'
            sub_path = os.path.join(path, 'benchmark_results', foldername, str(self.args.benchmark_idx))
            os.makedirs(sub_path, exist_ok=True)

            # Poisons
            benchmark_poisons = []
            for lookup, key in enumerate(self.poison_lookup.keys()):  # This is a different order than we usually do for compatibility with the benchmark
                input, label, _ = self.datasets.trainset[key]
                input += poison_delta[lookup, :, :, :]
                benchmark_poisons.append((_torch_to_PIL(input), int(label)))

            with open(os.path.join(sub_path, 'poisons.pickle'), 'wb+') as file:
                pickle.dump(benchmark_poisons, file, protocol=pickle.HIGHEST_PROTOCOL)

            # Target
            target, target_label, _ = self.datasets.targetset[0]
            with open(os.path.join(sub_path, 'target.pickle'), 'wb+') as file:
                pickle.dump((_torch_to_PIL(target), target_label), file, protocol=pickle.HIGHEST_PROTOCOL)

            # Indices
            with open(os.path.join(sub_path, 'base_indices.pickle'), 'wb+') as file:
                pickle.dump(self.poison_ids, file, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            raise NotImplementedError()

        print('Dataset fully exported.')

    def _dataset_clustering(self):
        if self.args.cluster_model:
            feature_extractor = pretrained_picker(self.args.cluster_model, self.args)
            feature_extractor.to(**self.setup)

            transform = self.trainset.transform
            new_transform = transform_picker(self.args.cluster_model, self.args)
            self.trainset.change_transform(new_transform)
            self.validset.change_transform(new_transform)
        elif self.feature_extractor:
            feature_extractor = self.feature_extractor
        else:
            raise ValueError('There is no feature extractor for clustering!')
    
        clusters = self.args.clusters
        featureset = FeatureDataset(self, self.trainset, feature_extractor, self.setup, num_workers=self.num_workers)
        n = len(self.trainset.classes)
        
        estimators = []
        offset = 0
        for target in range(n):
            features = None
            idxs = []
            for feature, label, idx in featureset:
                if label == target:
                    if features is None:
                        features = feature.unsqueeze(0).numpy()
                    else:
                        features = np.concatenate((features, feature.unsqueeze(0).numpy()), axis=0)
                    idxs.append(idx)
            
            estimator = KMeans(clusters, max_iter=1000, random_state=random.randint(0, 99999)).fit(features)
            for label, idx in zip(estimator.labels_, idxs):
                self.trainset.annotations[idx] = int(label) + offset
            estimators.append(estimator)
            offset += clusters
        
        self.trainset.subpopulations = list(range(offset))
        
        featureset = FeatureDataset(self, self.validset, feature_extractor, self.setup, num_workers=self.num_workers)
        offset = 0
        for target in range(n):
            estimator = estimators[target]
            for feature, label, idx in featureset:
                if label == target:
                    annotation = estimator.predict(feature.unsqueeze(0).numpy())[0]
                    self.validset.annotations[idx] = int(annotation) + offset

            offset += clusters

        self.validset.subpopulations = self.trainset.subpopulations

        if self.args.cluster_model:
            self.trainset.change_transform(transform)
            self.validset.change_transform(transform)

            del feature_extractor
            torch.cuda.empty_cache()
        
        if self.args.external_annotation is not None:
            with open(self.args.external_annotation, 'w') as f:
                data = {
                    'train': self.trainset.annotations,
                    'valid': self.validset.annotations
                }
                json.dump(data, f)
    
    def _load_annotation(self):
        with open(self.args.external_annotation) as f:
            data = json.load(f)
        self.trainset.annotations = data['train']
        self.validset.annotations = data['valid']