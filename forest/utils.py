"""Various utilities."""

import os
import csv
import socket
import datetime

from collections import defaultdict
from pathlib import Path

import torch
import random
import numpy as np
import sqlite3

from .consts import NON_BLOCKING, MAX_THREADING, DB_CONSTRUCT, DB_INSERT


def system_startup(args=None, defs=None):
    """Decide and print GPU / CPU / hostname info."""
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    setup = dict(device=device, dtype=torch.float, non_blocking=NON_BLOCKING)
    print('Currently evaluating -------------------------------:')
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    if args is not None:
        print(args)
    if defs is not None:
        print(repr(defs))
    print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')

    if torch.cuda.is_available():
        print(f'GPU : {torch.cuda.get_device_name(device=device)}')

    return setup


def average_dicts(running_stats):
    """Average entries in a list of dictionaries."""
    average_stats = defaultdict(list)
    for stat in running_stats[0]:
        if isinstance(running_stats[0][stat], list):
            for i, _ in enumerate(running_stats[0][stat]):
                average_stats[stat].append(np.mean([stat_dict[stat][i] for stat_dict in running_stats]))
        else:
            average_stats[stat] = np.mean([stat_dict[stat] for stat_dict in running_stats])
    return average_stats


def cw_loss(outputs, intended_classes, clamp=-100):
    """Carlini-Wagner loss for brewing [Liam's version]."""
    top_logits, _ = torch.max(outputs, 1)
    intended_logits = torch.stack([outputs[i, intended_classes[i]] for i in range(outputs.shape[0])])
    difference = torch.clamp(top_logits - intended_logits, min=clamp)
    return torch.mean(difference)

def _label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.shape[0], num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cw_loss2(outputs, intended_classes, confidence=0, clamp=-100):
    """CW variant 2. This is assert-level equivalent."""
    one_hot_labels = _label_to_onehot(intended_classes, num_classes=outputs.shape[1])
    target_logit = (outputs * one_hot_labels).sum(dim=1)
    second_logit, _ = (outputs - outputs * one_hot_labels).max(dim=1)
    cw_indiv = torch.clamp(second_logit - target_logit + confidence, min=clamp)
    return cw_indiv.mean()



def save_to_table(out_dir, name, dryrun, **kwargs):
    """Save keys to .csv files."""
    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, f'table_{name}.csv')
    fieldnames = list(kwargs.keys())

    # Read or write header
    try:
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = [line for line in reader][0]
    except Exception as e:
        print('Creating a new .csv table...')
        with open(fname, 'w') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writeheader()
    if not dryrun:
        # Add row for this experiment
        with open(fname, 'a') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writerow(kwargs)
        print('\nResults saved to ' + fname + '.')
    else:
        print(f'Would save results to {fname}.')


def record_results(args, kettle, stats_clean, stats_results, vnet):
    path = Path(args.db_path)

    conn = sqlite3.connect(path)
    cur = conn.cursor()

    res = cur.execute('SELECT name FROM sqlite_master')
    if res.fetchone() is None:
        cur.execute(DB_CONSTRUCT)
    
    feature_extractor = args.cluster_model
    if feature_extractor is None:
        feature_extractor = args.pretrained

    train_aug = 0 if args.noaugment else 1
    poison_aug = 1 if args.paugment else 0
    realistic = 1 if args.realistic else 0
    
    retrain = args.retrain_iter if args.retrain else None

    output = (args.pretrained, args.loss, str(args.net), vnet, args.dataset, args.poisonkey, args.recipe, realistic, args.defense,
              train_aug, poison_aug, args.pbatch, args.epochs, args.eps, args.budget, args.restarts, retrain, args.clusters, 
              feature_extractor, args.subpopulation_target, kettle.args.targets, len(kettle.datasets.target_validset), 
              stats_clean['target_losses'][-1], stats_clean['target_accs'][-1], stats_clean['target_losses_clean'][-1], 
              stats_clean['target_accs_clean'][-1], stats_results['target_losses'][-1], stats_results['target_accs'][-1], 
              stats_results['target_losses_clean'][-1], stats_results['target_accs_clean'][-1], stats_clean['valid_accs'][-1], stats_results['valid_accs'][-1])

    cur.execute(DB_INSERT, output)
    conn.commit()
    conn.close()

def set_random_seed(seed=233):
    """233 = 144 + 89 is my favorite number."""
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)

def set_deterministic():
    """Switch pytorch into a deterministic computation mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_num_workers():
    """Check devices and set an appropriate number of workers."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        max_num_workers = 4 * num_gpus
    else:
        max_num_workers = 4
    if torch.get_num_threads() > 1 and MAX_THREADING > 0:
        worker_count = min(min(2 * torch.get_num_threads(), max_num_workers), MAX_THREADING)
    else:
        worker_count = 0
    # worker_count = 200
    print(f'Data is loaded with {worker_count} workers.')
    return worker_count