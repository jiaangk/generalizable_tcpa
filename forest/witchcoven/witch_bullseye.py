"""Main class, holding information about models and training/testing routines."""

import torch
from torch.nn import MSELoss
from torchvision.models.feature_extraction import create_feature_extractor
from ..consts import BENCHMARK
from ..utils import cw_loss
torch.backends.cudnn.benchmark = BENCHMARK

from .witch_base import _Witch

class WitchBullsEye(_Witch):
    """Brew poison frogs variant with averaged feature matching instead of sums of feature matches.

    This is also known as BullsEye Polytope Attack.

    """

    def _define_objective(self, inputs, labels, targets):
        """Implement the closure here."""
        def closure(model, feature_extractor, criterion, optimizer, target_grad, target_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            if feature_extractor is None:
                model_input = inputs
            else:
                model_input = feature_extractor(inputs)

            if not self.args.extractor_collision:
                headless_model = self.bypass_last_layer(model)
                
                outputs = headless_model(model_input)
                target_outputs = headless_model(targets)
            else:
                outputs = model_input
                target_outputs = targets

            prediction = (model(model_input).argmax(dim=1) == labels).sum()

            feature_loss = MSELoss()(outputs.mean(dim=0), target_outputs.mean(dim=0))
            feature_loss.backward(retain_graph=self.retain)

            return feature_loss.detach().cpu(), prediction.detach().cpu()
        return closure


    @staticmethod
    def bypass_last_layer(model):
        """Hacky way of separating features and classification head for many models.

        Patch this function if problems appear.
        """
        if hasattr(model, 'bullseye'):
            def headless_model(*args, **kwargs):
                return model.bullseye(*args, **kwargs)

            return headless_model

        layer_cake = list(model.children())
        # last_layer = layer_cake[-1]
        headless_model = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())  # this works most of the time all of the time :<

        return headless_model
        # name = list(model.named_children())[-2][0]
        # headless = create_feature_extractor(model, return_nodes=[name])

        # def headless_model(*args, **kwargs):
        #     return torch.nn.Flatten()(headless(*args, **kwargs)[name])
        
        # return headless_model
