import random
import warnings

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from em.em_utils import calc_f1, get_criterion
from torch.utils.tensorboard import SummaryWriter
import wandb
from em.config import parse_args
from em.data.em_data_utils import get_dataset, set_to_device
from em.models.em_base_model import BaseModel
from em.models.unsupem_model import UNSUPEmModel

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
writer = SummaryWriter(log_dir='../tb_run')


def do_epoch(phase, data_loader, model, device, criterion, epoch_num, optimizer=None, scheduler=None, args=None,
             best_ckpt=None, amp_scaler=None):
    # # Set model to evaluate mode
    if phase == 'train':
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_step = 0

    all_pred = []
    all_labels = []
    if phase == 'train':
        scheduler.step()

    pbar = tqdm.tqdm(data_loader, desc='{}({})...'.format(phase, epoch_num),
                     colour='YELLOW' if phase == 'train' else 'GREEN')
    for dl_record in pbar:
        x1, x2, concat, labels = dl_record[0]
        with torch.set_grad_enabled(phase == 'train'):
            x1 = set_to_device(x1, device)
            x2 = set_to_device(x2, device)
            labels = labels.to(device)
            with torch.cuda.amp.autocast(enabled=(amp_scaler is not None)):
                outputs = model(x1, x2)

            all_pred += outputs.detach().cpu().numpy().tolist()
            all_labels += labels.cpu().numpy().tolist()

            total_step += 1
            f1 = calc_f1((all_pred, all_labels))[0] * 100
            pbar.set_description(f'{phase}({epoch_num})\t\tL: {total_loss / total_step: .4f}\t'
                                 f'F1: {f1: .3f}')

    if phase == 'valid' and args and (f1 > model.best_f1):
        model.best_f1 = f1
        model.best_loss = total_loss
        best_ckpt = model.save_to_file(args=args, overwrite=True)

    return total_loss, total_step, (all_pred, all_labels), best_ckpt


def main():
    args = parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print('UNSUPEM for dataset: {} and lm: {}'.format(args.dataset_name, args.lm))


    if args.seed != -1:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    tokenizer, type_tokenizer = BaseModel.get_tokenizers(args.lm, add_special_token=args.addsep)

    model = UNSUPEmModel(pretrained_lm=args.lm, sent_size=args.sentence_size).to(device)

    if args.addsep:
        model.resize_embedding('ctx', len(tokenizer))
        model.resize_embedding('typ', len(type_tokenizer))
        print('model embedding resized')

    criterion = get_criterion(args, device)

    _, _, ds_test = get_dataset(args.dataset_name, args.sentence_size, tokenizer, type_tokenizer, args.da,
                                sets=['test'])
    test_dl = DataLoader(dataset=ds_test, shuffle=False, batch_size=args.test_batch_size, num_workers=8)

    if args.wandb:
        wandb.init(project="unsupem", entity="makbn-uofa", tags=[args.dataset_name, args.lm, args.loss],
                   name=args.dataset_name + '-' + args.lm)
        wandb.config = {
            "test_batch_size": args.test_batch_size,
            "sentence_size": args.sentence_size,
            "addsep": args.addsep,
            "loss": args.addsep,
            "lm": args.lm,
            "dataset": args.dataset_name
        }

    test_loss, test_itr, test_pred_labels, _ = do_epoch(phase='test', data_loader=test_dl, model=model, device=device,
                                                        criterion=criterion, epoch_num=0,
                                                        optimizer=None, scheduler=None, amp_scaler=None)

    if args.wandb:
        wandb.log({"loss  test": (test_loss / test_itr),
                   "f1 test": calc_f1(test_pred_labels)[0]})


if __name__ == '__main__':
    main()
