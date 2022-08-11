import random
import warnings

import numpy as np
import os

import torch
from torch.utils.data import DataLoader
from em.em_utils import calc_f1, get_criterion
from em.config import parse_args
from em.data.em_data_utils import get_aug_dataset
from robem.robem_main_single import do_epoch
from em.models.em_base_model import BaseModel

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def get_load_path(args, ds_train_name):
    name = "{}-{}-{}-{}-{}-model.pt".format(args.lm, ds_train_name, 'deep' if args.deep else 'simple', args.loss
                                            , 'da' if args.da else 'no')
    full_path = os.path.join(args.save_dir, name)
    return full_path


def do_eval(dataset):
    args = parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if args.seed != -1:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    tokenizer, type_tokenizer = BaseModel.get_tokenizers(args.lm, add_special_token=args.addsep)
    supem_model = torch.load(get_load_path(args, dataset)).to(device)
    criterion = get_criterion(args, device)
    alter = [None, 'SFF', 'DRP', 'MIS', 'EXT', 'TYP']
    results = {}
    print(f'ROBEM: Run alters on dataset: {dataset}')
    for tf in alter:
        total_f1 = 0
        try_count = 20
        for j in range(try_count):
            _, _, ds_test = get_aug_dataset(dataset, args.sentence_size, tokenizer, type_tokenizer, sets=['test'],
                                            aug_size=1, alter=tf)
            test_dl = DataLoader(dataset=ds_test, shuffle=False, batch_size=args.test_batch_size, num_workers=8)
            test_loss, test_itr, test_pred_labels, _ = do_epoch(phase='test', data_loader=test_dl, model=supem_model,
                                                                device=device, criterion=criterion, epoch_num=0,
                                                                optimizer=None, scheduler=None, args=None,
                                                                best_ckpt=None, amp_scaler=None)
            f1, th = calc_f1(test_pred_labels)
            total_f1 += f1

        results[tf] = (total_f1/try_count) * 100
        print(f'Alters:{tf} best F1: {(total_f1/try_count) * 100: .4f}\n')

    return results


def run_test():
    # SFF: Shuffling columns. For a pair of entities, we shuffle only one of them.
    # DRP: Dropping a non-key column (This is different from missing values).
    #      We randomly remove one or more non-key columns.
    # MIS: Replace a non-key column with missing value.
    # EXT: Adding one or more irrelevant columns. The new columns can be of type integer,
    #      floating-point, or string. For text columns, I suggest using columns from other
    #      datasets rather than random words. We can have multiple test data for this.
    # TYP: For numerical columns, convert their types to different formats. For instance,
    #      for price, we can convert it to string or add a dollar sign or divide it by 1000
    #      like 9000->9K.

    for ds in ['itunes-amazon']:
            ds_results = do_eval(dataset=ds)


if __name__ == '__main__':
    run_test()
