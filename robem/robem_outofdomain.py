import random
import warnings

import numpy as np
import os
import pickle
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from em.em_utils import calc_f1, get_criterion
from em.config import parse_args
from em.data.em_data_utils import get_dataset
from robem.robem_main_single import do_epoch
from em.models.em_base_model import BaseModel

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
writer = SummaryWriter(log_dir='../tb_run')


def get_load_path(args, ds_train_name):
    name = "{}-{}-{}-{}-{}-model.pt".format(args.lm, ds_train_name, 'deep' if args.deep else 'simple', args.loss
                                            , 'da' if args.da else 'no')
    full_path = os.path.join(args.save_dir, name)
    return full_path


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('OUT OF DOMAIN:{} with lm: {}'.format(device, args.lm))
    if args.seed != -1:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    tokenizer = BaseModel.get_tokenizers(args.lm, add_special_token=args.addsep)

    criterion = get_criterion(args, device)

    train_datasets = [
         'itunes-amazon',
         'abt-buy',
         'amazon-google',
         'beer-rates',
         'dblp-acm',
         'dblp-scholar',
         'fodors-zagats',
         'walmart-amazon',
         'dirty-itunes-amazon',
         'dirty-dblp-acm',
         'dirty-dblp-scholar',
         'dirty-walmart-amazon',
    ]

    datasets = [
         'itunes-amazon',
         'abt-buy',
         'amazon-google',
         'beer-rates',
         'dblp-acm',
         'dblp-scholar',
         'fodors-zagats',
         'walmart-amazon',
         'dirty-itunes-amazon',
         'dirty-dblp-acm',
         'dirty-dblp-scholar',
         'dirty-walmart-amazon',
    ]

    results = {}
    mctest_all = {}
    for ds_train_name in train_datasets:
        print(f'\n\n\nloading model trained on {ds_train_name}')
        supem_model = torch.load(get_load_path(args, ds_train_name))
        for ds_test_name in datasets:
            _, _, ds_test = get_dataset(ds_test_name, args.sentence_size, tokenizer, args.da,
                                        sets=['test'])
            test_dl = DataLoader(dataset=ds_test, shuffle=False, batch_size=args.test_batch_size, num_workers=8)

            test_loss, test_itr, test_pred_labels, _ = do_epoch(phase='test', data_loader=test_dl, model=supem_model,
                                                                device=device, criterion=criterion, epoch_num=0,
                                                                optimizer=None, scheduler=None, args=None,
                                                                best_ckpt=None, amp_scaler=None)

            f1, th = calc_f1(test_pred_labels)

            mctest_list = []
            for yp, ya in zip(test_pred_labels[0], test_pred_labels[1]):
                ypl = yp > th

                if ypl == ya:
                    mctest_list.append(True)
                else:
                    mctest_list.append(False)

            mctest_all[(ds_train_name, ds_test_name)] = mctest_list
            results[ds_train_name + ' on ' + ds_test_name] = f1 * 100
            print(f'{ds_test_name} best F1: {f1 * 100: .4f}')

    print(results)
    pickle.dump(mctest_all, open('mc_test.pckl', 'wb'))
