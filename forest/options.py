"""Implement an ArgParser common to both brew_poison.py and dist_brew_poison.py ."""

import argparse

from .consts import DATASET_SETTING
from .consts import PRETRAINED_MODELS

def options():
    """Construct the central argument parser, filled with useful defaults.

    The first block is essential to test poisoning in different scenarios.
    The options following afterwards change the algorithm in various ways and are set to reasonable defaults.
    """
    parser = argparse.ArgumentParser(description='Construct poisoned training data for the given network and dataset')


    ###########################################################################
    # Central:
    parser.add_argument('--net', default='ResNet18', type=lambda s: [str(item) for item in s.split(',')])
    parser.add_argument('--dataset', default='CIFAR10', type=str, choices=list(DATASET_SETTING.keys()))
    parser.add_argument('--recipe', default='gradient-matching', type=str, choices=['gradient-matching', 'bullseye'])
    parser.add_argument('--threatmodel', default='single-class', type=str, choices=['single-class', 'random-subset'])

    # Reproducibility management:
    parser.add_argument('--poisonkey', default=None, type=str, help='Initialize poison setup with this key.')  # Also takes a triplet 0-3-1
    parser.add_argument('--modelkey', default=None, type=int, help='Initialize the model with this key.')

    # Poison properties / controlling the strength of the attack:
    parser.add_argument('--eps', default=16, type=float)
    parser.add_argument('--budget', default=0.01, type=float, help='Fraction of training data that is poisoned')
    parser.add_argument('--targets', default=1, type=int, help='Number of target instances selected from training set.'
                                                               'Note that there is always only one target subpopulation.')
    parser.add_argument('--assign_target', default=None, type=int)
    parser.add_argument('--subpopulation', default='annotation', type=str, choices=['annotation', 'cluster', 'external', 'external_annotation'], help='The method to generate subpopulations.')
    parser.add_argument('--subpopulation_target', default=None, type=int)
    parser.add_argument('--external_path', default='.', type=str, help='The folder of testing dataset.')
    parser.add_argument('--external_annotation', default=None, type=str)
    parser.add_argument('--clusters', default=1, type=int, help='When subpopulationg setting is cluster, the cluster per class.')
    parser.add_argument('--cluster_model', default=None, type=str, choices=['custom'] + list(PRETRAINED_MODELS.keys()),
                        help='The model used to extract features that be used to generate clusters. If cluster_model not given, the feature extractor will be pretrained model.')

    # Files and folders
    parser.add_argument('--db_path', default='./poisoned_result.db', type=str)
    parser.add_argument('--poison_path', default='poisons/', type=str)
    parser.add_argument('--data_path', default='~/data/', type=str)
    ###########################################################################

    parser.add_argument('--valid_knowledge', action='store_true', help='The attacker know the knowledge of valid subpopulation.')

    # Poison brewing:
    parser.add_argument('--attackoptim', default='signAdam', type=str)
    parser.add_argument('--attackiter', default=250, type=int)
    parser.add_argument('--init', default='randn', type=str)  # randn / rand
    parser.add_argument('--tau', default=0.1, type=float)
    parser.add_argument('--scheduling', action='store_false', help='Disable step size decay.')
    parser.add_argument('--target_criterion', default='cross-entropy', type=str, help='Loss criterion for target loss')
    parser.add_argument('--restarts', default=1, type=int, help='How often to restart the attack.')

    parser.add_argument('--pbatch', default=512, type=int, help='Poison batch size during optimization')
    parser.add_argument('--pshuffle', action='store_true', help='Shuffle poison batch during optimization')
    parser.add_argument('--paugment', action='store_false', help='Do not augment poison batch during optimization')
    parser.add_argument('--data_aug', type=str, default='default', help='Mode of diff. data augmentation.')

    # Poisoning algorithm changes
    parser.add_argument('--full_data', action='store_true', help='Use full train data (instead of just the poison images)')
    parser.add_argument('--adversarial', default=0, type=float, help='Adversarial PGD for poisoning.')
    parser.add_argument('--ensemble', default=1, type=int, help='Ensemble of networks to brew the poison on')
    parser.add_argument('--memory_cache', action='store_true', help='Save ensemble models in the memory in order to save video memory.')
    parser.add_argument('--stagger', action='store_true', help='Stagger the network ensemble if it exists')
    parser.add_argument('--step', action='store_true', help='Optimize the model for one epoch.')
    parser.add_argument('--max_epoch', default=None, type=int, help='Train only up to this epoch before poisoning.')
    parser.add_argument('--extractor_collision', action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--retrain_iter', default=50, type=int, help='Start retraining every <retrain_iter> iterations')

    # Use only a subset of the dataset:
    parser.add_argument('--ablation', default=1.0, type=float, help='What percent of data (including poisons) to use for validation')

    # Gradient Matching - Specific Options
    parser.add_argument('--loss', default='similarity', type=str)  # similarity is stronger in  difficult situations

    # These are additional regularization terms for gradient matching. We do not use them, but it is possible
    # that scenarios exist in which additional regularization of the poisoned data is useful.
    parser.add_argument('--centreg', default=0, type=float)
    parser.add_argument('--normreg', default=0, type=float)
    parser.add_argument('--repel', default=0, type=float)

    # Validation behavior
    parser.add_argument('--vruns', default=1, type=int, help='How often to re-initialize and check target after retraining')
    parser.add_argument('--vnet', default=None, type=lambda s: [str(item) for item in s.split(',')], help='Evaluate poison on this victim model. Defaults to --net')
    parser.add_argument('--retrain_from_init', action='store_true', help='Additionally evaluate by retraining on the same model initialization.')

    # Optimization setup
    parser.add_argument('--pretrained', default=None, type=str, choices=['custom'] + list(PRETRAINED_MODELS.keys()), help='Using a pretrained model as feature extractor. In this case, only Linear and MLP is available for --net argument.')
    parser.add_argument('--pretrained_path', default='./', type=str)
    parser.add_argument('--transfer', action='store_true', help='conduct end-to-end transfer experiment')
    parser.add_argument('--optimization', default='conservative', type=str, help='Optimization Strategy')
    # Strategy overrides:
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--noaugment', action='store_true', help='Do not use data augmentation during training.')
    parser.add_argument('--gradient_noise', default=None, type=float, help='Add custom gradient noise during training.')
    parser.add_argument('--gradient_clip', default=None, type=float, help='Add custom gradient clip during training.')

    # Debugging:
    parser.add_argument('--dryrun', action='store_true')
    parser.add_argument('--save', default=None, help='Export poisons into a given format. Options are full/limited/automl/numpy.')

    # Defenses:
    parser.add_argument('--realistic', action='store_true', help='Store the poison samples and then load them.')
    parser.add_argument('--defense', default=None, type=str, choices=['jpeg', 'bdr', 'gaussian'])

    return parser
