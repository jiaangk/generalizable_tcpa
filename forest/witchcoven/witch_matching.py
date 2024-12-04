"""Main class, holding information about models and training/testing routines."""

import torch
from ..consts import BENCHMARK
from ..utils import cw_loss
torch.backends.cudnn.benchmark = BENCHMARK

from .witch_base import _Witch

class WitchGradientMatching(_Witch):
    """Brew passenger poison with given arguments.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """

    def _define_objective(self, inputs, labels, targets, display=False):
        """Implement the closure here."""
        def closure(model, feature_extractor, criterion, optimizer, target_grad, target_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            if feature_extractor is None:
                model_input = inputs
            else:
                model_input = feature_extractor(inputs)
            outputs = model(model_input)
            if self.args.target_criterion in ['cw', 'carlini-wagner']:
                criterion = cw_loss
            else:
                pass  # use the default for untargeted or targeted cross entropy
            
            poison_loss = criterion(outputs, labels)
            prediction = (outputs.data.argmax(dim=1) == labels).sum()
            poison_grad = torch.autograd.grad(poison_loss, model.parameters(), allow_unused=True, retain_graph=True, create_graph=True)

            if self.args.loss == 'COSMSE':
                passenger_loss, cosine_similarity = self._cos_mse(poison_grad, target_grad, target_gnorm, display=display)
                if display == False:
                    passenger_loss.backward(retain_graph=self.retain)

                return cosine_similarity.detach().cpu(), prediction.detach().cpu()

            passenger_loss = self._passenger_loss(poison_grad, target_grad, target_gnorm, display=display)

            if self.args.centreg != 0:
                passenger_loss = passenger_loss + self.args.centreg * poison_loss
            if display == False:
                passenger_loss.backward(retain_graph=self.retain)
            return passenger_loss.detach().cpu(), prediction.detach().cpu()
        return closure
    
    def _cos_mse(self, poison_grad, target_grad, target_gnorm, display=False):
        passenger_loss = 0
        cosine_similarity = 0
        poison_norm = 0

        indices = torch.arange(len(target_grad))
        
        indices = [i for i in indices if poison_grad[i] is not None]

        for i in indices:
            passenger_loss += torch.nn.functional.mse_loss(poison_grad[i], target_grad[i], reduction='sum')
            cosine_similarity += (target_grad[i] * poison_grad[i]).sum()
            poison_norm += poison_grad[i].pow(2).sum()
        
        cosine_similarity = cosine_similarity / target_gnorm
        cosine_similarity = cosine_similarity / poison_norm.sqrt()
        if display:
            print((passenger_loss.sqrt() / target_gnorm).item(), ',', cosine_similarity.item(), sep='')
        passenger_loss = passenger_loss * (1.0 - cosine_similarity)

        return passenger_loss, 1.0 - cosine_similarity


    def _passenger_loss(self, poison_grad, target_grad, target_gnorm, display=False):
        """Compute the blind passenger loss term."""
        passenger_loss = 0
        poison_norm = 0
        cosine_similarity = 0
        mse_sum = 0

        SIM_TYPE = ['similarity', 'similarity-narrow', 'top5-similarity', 'top10-similarity', 'top20-similarity']
        if self.args.loss == 'top10-similarity':
            _, indices = torch.topk(torch.stack([p.norm() for p in target_grad], dim=0), 10)
        elif self.args.loss == 'top20-similarity':
            _, indices = torch.topk(torch.stack([p.norm() for p in target_grad], dim=0), 20)
        elif self.args.loss == 'top5-similarity':
            _, indices = torch.topk(torch.stack([p.norm() for p in target_grad], dim=0), 5)
        else:
            indices = torch.arange(len(target_grad))
        
        indices = [i for i in indices if poison_grad[i] is not None]

        for i in indices:
            if self.args.loss in ['scalar_product', *SIM_TYPE]:
                passenger_loss -= (target_grad[i] * poison_grad[i]).sum()
            elif self.args.loss == 'cosine1':
                passenger_loss -= torch.nn.functional.cosine_similarity(target_grad[i].flatten(), poison_grad[i].flatten(), dim=0)
            elif self.args.loss == 'SE':
                passenger_loss += 0.5 * (target_grad[i] - poison_grad[i]).pow(2).sum()
            elif self.args.loss == 'MSE':
                passenger_loss += torch.nn.functional.mse_loss(target_grad[i], poison_grad[i])

            poison_norm += poison_grad[i].pow(2).sum()

            if display:
                mse_sum += (target_grad[i] - poison_grad[i]).pow(2).sum()
                cosine_similarity += (target_grad[i] * poison_grad[i]).sum()

        if self.args.repel != 0:
            for i in indices:
                if self.args.loss in ['scalar_product', *SIM_TYPE]:
                    passenger_loss += self.args.repel * (target_grad[i] * poison_grad[i]).sum()
                elif self.args.loss == 'cosine1':
                    passenger_loss -= self.args.repel * torch.nn.functional.cosine_similarity(target_grad[i].flatten(), poison_grad[i].flatten(), dim=0)
                elif self.args.loss == 'SE':
                    passenger_loss -= 0.5 * self.args.repel * (target_grad[i] - poison_grad[i]).pow(2).sum()
                elif self.args.loss == 'MSE':
                    passenger_loss -= self.args.repel * torch.nn.functional.mse_loss(target_grad[i], poison_grad[i])

        if display:
            cosine_similarity = cosine_similarity / target_gnorm
            cosine_similarity = cosine_similarity / poison_norm.sqrt()
            print((mse_sum.sqrt() / target_gnorm).item(), ',', cosine_similarity.item(), sep='')

        passenger_loss = passenger_loss / target_gnorm  # this is a constant

        if self.args.loss in SIM_TYPE:
            passenger_loss = 1 + passenger_loss / poison_norm.sqrt()
        if self.args.normreg != 0:
            passenger_loss = passenger_loss + self.args.normreg * poison_norm.sqrt()

        if self.args.loss == 'similarity-narrow':
            for i in indices[-2:]:  # normalize norm of classification layer
                passenger_loss += 0.5 * poison_grad[i].pow(2).sum() / target_gnorm

        return passenger_loss