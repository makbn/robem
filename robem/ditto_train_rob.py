from argparse import Namespace
import random

import numpy as np
import torch

from em.ditto.ditto_light.ditto import train
from em.ditto.ditto_light.dataset import DittoDataset
from em.ditto.ditto_light.knowledge import ProductDKInjector, GeneralDKInjector


if __name__ == "__main__":

    hp = Namespace(task="",
                   run_id=0,
                   batch_size=64,
                   max_len=256,
                   lr=3e-5,
                   n_epochs=40,
                   finetuning=True,
                   save_model=True,
                   logdir='ditto_checkpoints/rob_shuff/',
                   lm='roberta-base',
                   fp16=True,
                   da=None,
                   alpha_aug=0.8,
                   dk='general',
                   summarize=True,
                   size=None)
    datasets = [
         #'itunes_amazon',
         #'abt_buy',
         #'amazon_google',
         #'beer_rates',
         #'dblp_acm',
         #'dblp_scholar',
         #'fodors_zagats',
         #'walmart_amazon',
         #'dirty_itunes_amazon',
         #'dirty_dblp_acm',
         #'dirty_dblp_scholar',
         #'dirty_walmart_amazon',
    ]

    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    results = {}
    for ds in datasets:
        hp.__setattr__('task', ds)
        print(f'train on: {ds}')
        testset = f'../data/augmented/ditto/shuff/1/{ds}/test.txt'
        trainset = f'../data/augmented/ditto/shuff/1/{ds}/train.txt'
        validset = f'../data/augmented/ditto/shuff/1/{ds}/valid.txt'

        if hp.dk is not None:
            if hp.dk == 'product':
                injector = ProductDKInjector(None, hp.dk)
            else:
                injector = GeneralDKInjector(None, hp.dk)

            trainset = injector.transform_file(trainset)
            validset = injector.transform_file(validset)
            testset = injector.transform_file(testset)

        train_dataset = DittoDataset(trainset,
                                     lm=hp.lm,
                                     max_len=hp.max_len,
                                     size=hp.size,
                                     da=hp.da)
        valid_dataset = DittoDataset(validset, lm=hp.lm)
        test_dataset = DittoDataset(testset, lm=hp.lm)
        f1s = train(train_dataset, valid_dataset, test_dataset, hp)

        results[ds] = f1s['t_f1']

    print(results)
