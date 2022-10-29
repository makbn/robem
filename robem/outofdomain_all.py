import torch
from torch.utils import data
from em.ditto.ditto_light.ditto_light.dataset import DittoDataset
from em.ditto.ditto_light.ditto_light.knowledge import *
from em.ditto.ditto_light.ditto_light.ditto import evaluate, DittoModel


def get_full_path(logdir, task):
    ckpt_path = os.path.join(logdir, task, 'model.pt')
    return ckpt_path


if __name__ == "__main__":

    datasets = [
        'itunes_amazon',
        'abt_buy',
        'amazon_google',
        'beer_rates',
        'dblp_acm',
        'dblp_scholar',
        'fodors_zagats',
        'walmart_amazon',
        'dirty_itunes_amazon',
        'dirty_dblp_acm',
        'dirty_dblp_scholar',
        'dirty_walmart_amazon',
    ]

    logdir = 'ditto_checkpoints/ditto_da/'
    lm = 'roberta-base'

    results = {}
    mc_test_all = {}
    for train_ds in ['itunes_amazon']:

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = DittoModel(device=device, lm=lm)

        saved_state = torch.load(get_full_path(logdir, train_ds), map_location=lambda storage, loc: storage)

        model.load_state_dict(saved_state['model'])
        model = model.to(device)

        print(f'\n\n\n{train_ds} model loaded!')

        for test_ds in ['itunes_amazon']:

            if test_ds != train_ds:
                continue

            testset = f'../data/augmented/ditto/1/{test_ds}/test.txt'
            trainset = f'../data/augmented/ditto/1/{test_ds}/train.txt'
            validset = f'../data/augmented/ditto/1/{test_ds}/valid.txt'

            # load train/dev/test sets
            train_dataset = DittoDataset(trainset,
                                         lm=lm,
                                         max_len=256,
                                         size=None,
                                         da=None)

            test_dataset = DittoDataset(testset, lm=lm)

            padder = train_dataset.pad

            test_iter = data.DataLoader(dataset=test_dataset,
                                        batch_size=64 * 16,
                                        shuffle=False,
                                        num_workers=0,
                                        collate_fn=padder)

            dev_f1, th, mc_test = evaluate(model, test_iter)
            mc_test_all[(train_ds, test_ds)] = mc_test

            results[f'train-{train_ds}___test-{test_ds}'] = dev_f1

            print(f'{test_ds} best F1: {dev_f1: .4f}')

    #pickle.dump(mc_test_all, open('mc_test_all_ditto_shuff_rob.pckl', 'wb'))
    print(results)



