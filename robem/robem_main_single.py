import random
import warnings

import numpy as np
import torch
import tqdm
import wandb
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW

from em.config import parse_args
from em.data.em_data_utils import get_dataset, set_to_device, get_aug_dataset
from em.em_utils import calc_f1, get_criterion
from em.models.em_base_model import BaseModel
from em.models.em_model import EmModel

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
writer = SummaryWriter(log_dir='../tb_run')


def do_epoch(phase, data_loader, model, device, criterion, epoch_num, optimizer=None, scheduler=None, args=None,
             best_ckpt=None, amp_scaler=None, dataset_name=None):
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
    for inputs in pbar:
        for inp in inputs:

            x1, x2, concat, labels = inp
            with torch.set_grad_enabled(phase == 'train'):
                x1 = set_to_device(x1, device)
                x2 = set_to_device(x2, device)

                concat = set_to_device(concat, device)
                labels = labels.to(device)
                with torch.cuda.amp.autocast(enabled=(amp_scaler is not None)):
                    outputs = model(x1, x2, concat)
                    loss = criterion(outputs, labels)

                all_pred += outputs.detach().softmax(dim=1)[:, 1].cpu().numpy().tolist()
                all_labels += labels.cpu().numpy().tolist()
                total_loss += (loss.item() * labels.shape[0])

                if phase == 'train':
                    optimizer.zero_grad()
                    if amp_scaler:
                        amp_scaler.scale(loss).backward()
                        amp_scaler.step(optimizer)
                        amp_scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                total_step += 1
                f1 = calc_f1((all_pred, all_labels))[0] * 100
                pbar.set_description(f'{phase}({epoch_num})\t\tL: {total_loss / total_step: .4f}\t'
                                     f'F1: {f1: .3f}')

    if phase == 'valid' and args and (f1 > model.best_f1):
        model.best_f1 = f1
        model.best_loss = total_loss
        best_ckpt = model.save_to_file(args=args, overwrite=True, dataset_name=dataset_name)

    return total_loss, total_step, (all_pred, all_labels), best_ckpt


def main(dataset_ov=None, rob=False, aug_size=1, alter=None):
    args = parse_args()

    if dataset_ov:
        dataset_name = dataset_ov
    else:
        dataset_name = args.dataset_name

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print('ROBEM:{} for dataset: {} and lm: {} and loss:{} cls:{}'.format(device, dataset_name, args.lm, args.loss,
                                                                          args.deep))
    if args.wandb:
        wandb.init(project="robem", entity="makbn-uofa", tags=[dataset_name, args.lm, args.loss],
                   name=dataset_name + '-' + args.lm)
        wandb.config = {
            "learning_rate": args.lr,
            "train_batch_size": args.train_batch_size,
            "test_batch_size": args.test_batch_size,
            "data_aug": args.da,
            "rob": rob,
            "deep": args.deep,
            "sentence_size": args.sentence_size,
            "fp16": args.fp16,
            "addsep": args.addsep,
            "loss": args.addsep,
            "lm": args.lm,
            "dataset": dataset_name
        }

    scaler = torch.cuda.amp.GradScaler()

    if args.seed != -1:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    tokenizer = BaseModel.get_tokenizers(args.lm, add_special_token=args.addsep)

    if not rob:
        ds_train, ds_val, ds_test = get_dataset(dataset_name, args.sentence_size, tokenizer, args.da,
                                                ditto_aug=args.ditto_aug)
    else:
        ds_train, ds_val, ds_test = get_aug_dataset(dataset_name, args.sentence_size, tokenizer,
                                                    aug_size=aug_size, alter=alter)

    train_dl = DataLoader(dataset=ds_train, shuffle=True, batch_size=args.train_batch_size, num_workers=1)
    val_dl = DataLoader(dataset=ds_val, shuffle=False, batch_size=args.test_batch_size, num_workers=1)
    test_dl = DataLoader(dataset=ds_test, shuffle=False, batch_size=args.test_batch_size, num_workers=1)

    model = EmModel(pretrained_lm=args.lm, sent_size=args.sentence_size, freeze=False,
                    deep_classifier=args.deep).to(device)

    if args.wandb:
        wandb.watch(model)

    if args.addsep:
        model.resize_embedding('ctx', len(tokenizer))
        print('model embedding resized')

    criterion = get_criterion(args, device)

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.wd)
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    g_train_f1 = 0
    g_train_loss = 0

    g_val_f1 = 0
    g_val_loss = 0
    best_ckpt = None
    for epoch in range(args.num_epochs):
        train_loss, train_itr, train_pred_labels, _ = do_epoch(phase='train', data_loader=train_dl, model=model,
                                                               args=args, optimizer=optimizer, device=device,
                                                               epoch_num=epoch, criterion=criterion,
                                                               scheduler=step_lr_scheduler, best_ckpt=best_ckpt,
                                                               amp_scaler=scaler)

        val_loss, val_itr, val_pred_labels, best_ckpt = do_epoch(phase='valid', data_loader=val_dl, model=model,
                                                                 device=device, optimizer=optimizer, epoch_num=epoch,
                                                                 scheduler=step_lr_scheduler, args=args,
                                                                 criterion=criterion, best_ckpt=best_ckpt,
                                                                 amp_scaler=scaler, dataset_name=dataset_name)

        if epoch % 5 == 0:
            test_loss, test_itr, test_pred_labels, _ = do_epoch(phase='test', data_loader=test_dl, model=model,
                                                                device=device,
                                                                criterion=criterion, epoch_num=0,
                                                                optimizer=optimizer, scheduler=None, amp_scaler=scaler)
            wandb.log({"loss  test-mid": (test_loss / test_itr),
                       "f1 test-mid": calc_f1(test_pred_labels)[0]})

        writer.add_scalars('epoch loss', {'train': train_loss / train_itr,
                                          'val': (val_loss / val_itr)}, epoch)

        f1_train, best_th_train = calc_f1(train_pred_labels)
        f1_val, best_th_test = calc_f1(val_pred_labels)

        if args.wandb:
            wandb.log({"loss  val": (val_loss / val_itr),
                       "loss train": (train_loss / train_itr),
                       "f1 val": f1_val,
                       "f1 train": f1_train,
                       "learning rate": step_lr_scheduler.get_last_lr()[0]})

        writer.add_scalars('epoch f1', {'val': f1_val, 'train': f1_train}, epoch)
        writer.add_scalar('learning rate', step_lr_scheduler.get_last_lr()[0], epoch)

        writer.flush()

        g_val_f1 += f1_val
        g_train_f1 += f1_train

        g_train_loss += (val_loss / train_itr)
        g_val_loss += (train_loss / train_itr)

    print(
        f'avg Train(loss:{g_train_loss / args.num_epochs: .4f}\t f1:{(g_train_f1 * 100) / args.num_epochs: .2f}%\t '
        f'Test:{g_val_loss / args.num_epochs: .4f}\t f1:{(g_val_f1 * 100) / args.num_epochs: .2f}%.')

    print(f'loading model from: {best_ckpt}')
    model = torch.load(best_ckpt)

    test_loss, test_itr, test_pred_labels, _ = do_epoch(phase='test', data_loader=test_dl, model=model, device=device,
                                                        criterion=criterion, epoch_num=0,
                                                        optimizer=optimizer, scheduler=None, amp_scaler=scaler)

    if args.wandb:
        wandb.log({"loss  test": (test_loss / test_itr),
                   "f1 test": calc_f1(test_pred_labels)[0],
                   "learning rate": step_lr_scheduler.get_last_lr()[0]})

    return calc_f1(test_pred_labels)[0] * 100


if __name__ == "__main__":
    main()
