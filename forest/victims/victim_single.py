"""Single model default victim class."""

import torch
import numpy as np
from collections import defaultdict

from ..consts import BENCHMARK
from ..utils import set_random_seed
from .levit import Residual
torch.backends.cudnn.benchmark = BENCHMARK

from .victim_base import _VictimBase

class _VictimSingle(_VictimBase):
    """Implement model-specific code and behavior for a single model on a single GPU.

    This is the simplest victim implementation.

    """

    """ Methods to initialize a model."""
    def initialize(self, seed=None, silent=False):
        if self.args.modelkey is None:
            if seed is None:
                self.model_init_seed = np.random.randint(0, 2**32 - 1)
            else:
                self.model_init_seed = seed
        else:
            self.model_init_seed = self.args.modelkey
        set_random_seed(self.model_init_seed)
        self.model, self.defs, self.criterion, self.optimizer, self.scheduler = self._initialize_model(self.args.net[0])
        self.epoch = 0
        self.stats = defaultdict(list)

        self.model.to(**self.setup)
        
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        if not silent:
            print(f'{self.args.net[0]} model initialized with random key {self.model_init_seed}.')

    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""

    def _iterate(self, kettle, poison_delta, max_epoch=None):
        """Validate a given poison by training the model and checking target accuracy."""
        if max_epoch is None:
            max_epoch = self.defs.epochs

        def loss_fn(model, outputs, labels):
            return self.criterion(outputs, labels)

        single_setup = (self.model, self.feature_extractor, self.defs, self.criterion, self.optimizer, self.scheduler)
        for self.epoch in range(max_epoch):
            self._step(kettle, poison_delta, loss_fn, self.epoch, self.stats, *single_setup)
            if self.args.dryrun:
                break

        return self.stats

    def step(self, kettle, poison_delta, poison_targets, true_classes):
        """Step through a model epoch. Optionally: minimize target loss."""
        def loss_fn(model, outputs, labels):
            normal_loss = self.criterion(outputs, labels)
            model.eval()
            if self.args.adversarial != 0:
                target_loss = 1 / self.defs.batch_size * self.criterion(model(poison_targets), true_classes)
            else:
                target_loss = 0
            model.train()
            return normal_loss + self.args.adversarial * target_loss

        stats = self.stats

        single_setup = (self.model, self.feature_extractor, self.defs, self.criterion, self.optimizer, self.scheduler)
        self._step(kettle, poison_delta, loss_fn, self.epoch, stats, *single_setup, silent=True)

        self.epoch += 1
        if self.epoch >= self.defs.epochs:
            self.initialize(silent=True)

        return stats

    """ Various Utilities."""

    def eval(self, dropout=False):
        """Switch everything into evaluation mode."""
        def apply_dropout(m):
            """https://discuss.pytorch.org/t/dropout-at-test-time-in-densenet/6738/6."""
            if isinstance(m, torch.nn.Dropout):
                m.train()
            if isinstance(m, Residual):
                m.training = True

        self.model.eval()
        if dropout:
            self.model.apply(apply_dropout)

    def reset_learning_rate(self):
        """Reset scheduler object to initial state."""
        _, _, self.optimizer, self.scheduler = self._initialize_model()

    def gradient(self, samples, labels, criterion=None):
        """Compute the gradient of criterion(model) w.r.t to given data."""
        if criterion is None:
            loss = self.criterion(self.model(samples), labels)
        else:
            loss = criterion(self.model(samples), labels)
        gradients = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True, only_inputs=True)
        grad_norm = 0
        for grad in gradients:
            if grad is not None:
                grad_norm += grad.detach().pow(2).sum()
        grad_norm = grad_norm.sqrt()
        return gradients, grad_norm

    def compute(self, function, *args):
        r"""Compute function on the given optimization problem, defined by criterion \circ model.

        Function has arguments: model, criterion
        """
        return function(self.model, self.feature_extractor, self.criterion, self.optimizer, *args)
