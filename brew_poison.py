"""General interface script to launch poisoning jobs."""
import torch

import datetime
import time

import forest
import random
from forest.utils import record_results
torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)

# Parse input arguments
args = forest.options().parse_args()
if args.poisonkey is None:
    args.poisonkey = str(random.randint(0, 10000000))
forest.utils.set_random_seed(int(args.poisonkey))
forest.utils.set_deterministic()

if __name__ == "__main__":

    setup = forest.utils.system_startup(args)
    epochs = args.epochs

    augment_backup = args.noaugment
    args.noaugment = not args.paugment
    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations, model.feature_extractor, setup=setup)
    witch = forest.Witch(args, setup=setup)

    start_time = time.time()
    stats_clean = model.train(data, max_epoch=args.max_epoch)
    train_time = time.time()
    poison_delta = witch.brew(model, data)
    brew_time = time.time()

    train_net = args.net

    nets = args.net if args.vnet is None else args.vnet
    args.noaugment = augment_backup
    if args.realistic:
        data.realistic_process(poison_delta)
        poison_delta = None

    for net in nets:
        args.net = [net]
        if args.vruns > 0:
            args.epochs = epochs
            model = forest.Victim(args, setup=setup)
            stats_results = model.validate(data, poison_delta)
        else:
            stats_results = None
        args.net = train_net
        record_results(args, data, stats_clean, stats_results, net)

    test_time = time.time()

    
    timestamps = dict(train_time=str(datetime.timedelta(seconds=train_time - start_time)).replace(',', ''),
                      brew_time=str(datetime.timedelta(seconds=brew_time - train_time)).replace(',', ''),
                      test_time=str(datetime.timedelta(seconds=test_time - brew_time)).replace(',', ''))

    # Export
    if args.save is not None:
        data.export_poison(poison_delta, path=args.poison_path, mode=args.save)

    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with train time: {str(datetime.timedelta(seconds=train_time - start_time))}')
    print(f'--------------------------- brew time: {str(datetime.timedelta(seconds=brew_time - train_time))}')
    print(f'--------------------------- test time: {str(datetime.timedelta(seconds=test_time - brew_time))}')
    print('-------------Job finished.-------------------------')