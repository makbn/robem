import argparse
from argparse import Namespace


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--dataset_name', default='itunes-amazon', choices=['beer-rates', 'itunes-amazon', 'amazon-google',
                                                                'abt-buy', 'fodors-zagats', 'dblp-acm', 'dblp-scholar',
                                                                'walmart-amazon'],
            help="['beer-rates', 'itunes-amazon', 'amazon-google', 'company', 'abt-buy', 'fodors-zagats', 'dblp-acm', "
                 "'dblp-scholar', 'walmart-amazon']")
    #_nonspecialchar
    add_arg('--lm', default="roberta-base", choices=['bert', 'bert-large', 'roberta-base'],
            help='bert or bert-large or roberta-base')
    add_arg('--lr', default=3e-5, type=float, help="learning rate")
    add_arg('--da', default=False, type=bool, help="enable data augmentation")
    add_arg('--ditto_aug', default='all', type=str, choices=['del', 'drop_col', 'append_col', 'drop_token', 'drop_len',
            'drop_sym', 'drop_same', 'swap', 'ins', 'all'], help="enable ditto data augmentation")
    add_arg('--deep', default=True, type=bool, help="enable deep classifier")
    add_arg('--addsep', default=False, type=bool, help="add special token for attr separation to tokenizer and model")
    add_arg('--wd', default=0, type=float, help="weight decay")
    add_arg('--num_epochs', default=40, type=int, help="number of epochs")
    add_arg('--train_batch_size', default=64, type=int, help="train batch size")
    add_arg('--test_batch_size', default=64, type=int, help="train batch size")
    add_arg('--sentence_size', default=256, type=int, help="sentence size")
    add_arg('--fp16', default=True, type=bool, help="use fp16")
    add_arg('--wandb', default=True, type=bool, help="log results in wandb")
    add_arg('--save_dir', default='../checkpoint/', type=str, help="save directory for model checkpoint")
    add_arg('--loss', default='wce', choices=['ce', 'wce', 'asl'],
            help="wce: cross-entropy and wce: weighted cross-entropy asl: asymmetric loss")
    add_arg('--neg_weight', default=0.20, type=float, help="wce & asl loss weight for negative samples")
    add_arg('--pos_weight', default=0.80, type=float, help="wce & asl loss weight for negative samples")
    add_arg('--folds', default=2, type=int, help="cross validation k-folds >= 1")
    add_arg('--mcn_test', default=True, type=int, help="McNemar’s test for out of domain experiments")
    add_arg('--seed', default=0, type=int, help="random and dataloader seed (-1 if you dont want to set)")

    args = parser.parse_args()

    if args.da is True and args.ditcdto_aug is True:
        raise ValueError('both data augmentation are enabled! choose one of da or ditto_da methods!')
    return parser.parse_args()

