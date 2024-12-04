"""Repeatable code parts concerning optimization and training schedules."""


import torch

import datetime
from .utils import print_and_save_stats, save_stats, pgd_step

from ..consts import NON_BLOCKING, BENCHMARK, DEBUG_TRAINING
torch.backends.cudnn.benchmark = BENCHMARK


def get_optimizers(model, args, defs):
    """Construct optimizer as given in defs."""
    if defs.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=defs.lr, momentum=0.9,
                                    weight_decay=defs.weight_decay, nesterov=True)
    elif defs.optimizer == 'SGD-basic':
        optimizer = torch.optim.SGD(model.parameters(), lr=defs.lr, momentum=0.0,
                                    weight_decay=defs.weight_decay, nesterov=False)
    elif defs.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=defs.lr, weight_decay=defs.weight_decay)

    if defs.scheduler == 'cyclic':
        effective_batches = (50_000 // defs.batch_size) * defs.epochs
        print(f'Optimization will run over {effective_batches} effective batches in a 1-cycle policy.')
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=defs.lr / 100, max_lr=defs.lr,
                                                      step_size_up=effective_batches // 2,
                                                      cycle_momentum=True if defs.optimizer in ['SGD'] else False)
    elif defs.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=defs.milestones, gamma=0.1)
    elif defs.scheduler == 'none':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[10_000, 15_000, 25_000], gamma=1)

        # Example: epochs=160 leads to drops at 60, 100, 140.
    return optimizer, scheduler


def run_step(kettle, poison_delta, loss_fn, epoch, stats, model, feature_extractor, defs, criterion, optimizer, scheduler, ablation=True, silent=False):

    epoch_loss, total_preds, correct_preds = 0, 0, 0
    if DEBUG_TRAINING:
        data_timer_start = torch.cuda.Event(enable_timing=True)
        data_timer_end = torch.cuda.Event(enable_timing=True)
        forward_timer_start = torch.cuda.Event(enable_timing=True)
        forward_timer_end = torch.cuda.Event(enable_timing=True)
        backward_timer_start = torch.cuda.Event(enable_timing=True)
        backward_timer_end = torch.cuda.Event(enable_timing=True)

        stats['data_time'] = 0
        stats['forward_time'] = 0
        stats['backward_time'] = 0

        data_timer_start.record()

    if kettle.args.ablation < 1.0:
        # run ablation on a subset of the training set
        loader = kettle.partialloader
    else:
        loader = kettle.trainloader

    if poison_delta is not None:
        poison_delta = poison_delta.to(**kettle.setup)
        if not (defs.augmentations or (feature_extractor is None)):
            poison_origin_img = kettle.poison_origin_img.to(**kettle.setup)
            poison_imgs = poison_origin_img + poison_delta
    
    for batch, (inputs, labels, ids) in enumerate(loader):
        # Prep Mini-Batch
        model.train()
        optimizer.zero_grad()

        # Transfer to GPU
        inputs = inputs.to(**kettle.setup)
        labels = labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)

        if DEBUG_TRAINING:
            data_timer_end.record()
            forward_timer_start.record()

        # Add adversarial pattern
        with torch.no_grad():
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
                    if defs.augmentations or (feature_extractor is None):
                        inputs[batch_positions] += poison_delta[poison_slices]
                    else:
                        delta_imgs = poison_imgs[poison_slices]
                        delta_features = feature_extractor(delta_imgs)
                        
                        inputs[batch_positions] = delta_features

            if defs.augmentations:
                inputs = kettle.augment(inputs)

                if feature_extractor is not None:
                    inputs = feature_extractor(inputs)

        # Get loss
        outputs = model(inputs)
        loss = loss_fn(model, outputs, labels)
        if DEBUG_TRAINING:
            forward_timer_end.record()
            backward_timer_start.record()

        loss.backward()

        # Enforce batch-wise privacy if necessary
        # This is a defense discussed in Hong et al., 2020
        # We enforce privacy on mini batches instead of instances to cope with effects on batch normalization
        # This is reasonble as Hong et al. discuss that defense against poisoning mostly arises from the addition
        # of noise to the gradient signal
        with torch.no_grad():
            if defs.privacy['clip'] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), defs.privacy['clip'])
            if defs.privacy['noise'] is not None:
                # generator = torch.distributions.laplace.Laplace(torch.as_tensor(0.0).to(**kettle.setup),
                #                                                 kettle.defs.privacy['noise'])
                for param in model.parameters():
                    # param.grad += generator.sample(param.shape)
                    noise_sample = torch.randn_like(param) * defs.privacy['clip'] * defs.privacy['noise']
                    param.grad += noise_sample


        optimizer.step()

        predictions = torch.argmax(outputs.data, dim=1)
        total_preds += labels.size(0)
        correct_preds += (predictions == labels).sum().item()
        epoch_loss += loss.item()

        if DEBUG_TRAINING:
            backward_timer_end.record()
            torch.cuda.synchronize()
            stats['data_time'] += data_timer_start.elapsed_time(data_timer_end)
            stats['forward_time'] += forward_timer_start.elapsed_time(forward_timer_end)
            stats['backward_time'] += backward_timer_start.elapsed_time(backward_timer_end)

            data_timer_start.record()

        if defs.scheduler == 'cyclic':
            scheduler.step()
        if kettle.args.dryrun:
            break
    if defs.scheduler == 'linear':
        scheduler.step()

    if epoch % defs.validate == 0 or epoch == (defs.epochs - 1):
        valid_acc, valid_loss = run_validation(model, criterion, kettle.validloader, kettle.setup, kettle.args.dryrun)
        targets = check_targets(
            model, criterion, kettle.target_validloader, kettle.poison_setup['intended_class'],
            kettle.poison_setup['target_class'],
            kettle.setup)
    else:
        valid_acc, valid_loss = None, None
        targets = None
    current_lr = optimizer.param_groups[0]['lr']
    if not silent:
        print_and_save_stats(epoch, stats, current_lr, epoch_loss / (batch + 1), correct_preds / total_preds,
                             valid_acc, valid_loss, targets)
    else:
        save_stats(stats, epoch_loss / (batch + 1), correct_preds / total_preds, valid_acc, valid_loss,
                   targets)

    if DEBUG_TRAINING:
        print(f"Data processing: {datetime.timedelta(milliseconds=stats['data_time'])}, "
              f"Forward pass: {datetime.timedelta(milliseconds=stats['forward_time'])}, "
              f"Backward Pass and Gradient Step: {datetime.timedelta(milliseconds=stats['backward_time'])}")
        stats['data_time'] = 0
        stats['forward_time'] = 0
        stats['backward_time'] = 0


def run_validation(model, criterion, dataloader, setup, dryrun=False):
    """Get accuracy of model relative to dataloader."""
    model.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(dataloader):
            inputs = inputs.to(**setup)
            targets = targets.to(device=setup['device'], dtype=torch.long, non_blocking=NON_BLOCKING)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss += criterion(outputs, targets).item()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            if dryrun:
                break

    accuracy = correct / total
    loss_avg = loss / (i + 1)
    return accuracy, loss_avg

def check_targets(model, criterion, dataloader, intended_class, original_class, setup):
    """Get accuracy and loss for all targets on their intended class."""
    model.eval()
    total = 0
    correct_clean = 0
    correct_intended = 0
    loss_clean = 0
    loss_intended = 0

    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(dataloader):
            batch_length = len(inputs)
            inputs = inputs.to(**setup)
            targets = targets.to(device=setup['device'], dtype=torch.long, non_blocking=NON_BLOCKING)

            outputs = model(inputs)
            predictions = torch.argmax(outputs.data, dim=1)
            loss_clean += criterion(outputs, targets).item()
            
            correct_clean += (predictions == targets).sum().item()

            if intended_class is not None:
                targets = torch.tensor([intended_class for _ in range(batch_length)]).to(device=setup['device'], dtype=torch.long, non_blocking=NON_BLOCKING)
                loss_intended += criterion(outputs, targets).item()

                correct_intended += (predictions == targets).sum().item()
            else:
                loss_intended = -loss_clean
                correct_intended += (predictions != targets).sum().item()
            
            total += batch_length

    accuracy_clean = correct_clean / total
    accuracy_intend = correct_intended / total
    loss_clean_avg = loss_clean / (i + 1)
    loss_intend_avg = loss_intended / (i + 1)
    return accuracy_intend, loss_intend_avg, accuracy_clean, loss_clean_avg

def check_targets_old(model, criterion, targetset, intended_class, original_class, setup):
    """Get accuracy and loss for all targets on their intended class."""
    model.eval()
    if len(targetset) > 0:

        target_images = torch.stack([data[0] for data in targetset]).to(**setup)
        intended_labels = torch.tensor([intended_class for _ in range(len(targetset))]).to(device=setup['device'], dtype=torch.long)
        original_labels = torch.stack([torch.as_tensor(data[1], device=setup['device'], dtype=torch.long) for data in targetset])
        with torch.no_grad():
            outputs = model(target_images)
            predictions = torch.argmax(outputs, dim=1)

            loss_intended = criterion(outputs, intended_labels)
            accuracy_intended = (predictions == intended_labels).sum().float() / predictions.size(0)
            loss_clean = criterion(outputs, original_labels)
            predictions_clean = torch.argmax(outputs, dim=1)
            accuracy_clean = (predictions == original_labels).sum().float() / predictions.size(0)

            # print(f'Raw softmax output is {torch.softmax(outputs, dim=1)}, intended: {intended_class}')

        return accuracy_intended.item(), loss_intended.item(), accuracy_clean.item(), loss_clean.item()
    else:
        return 0, 0, 0, 0