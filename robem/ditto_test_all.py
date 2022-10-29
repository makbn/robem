import torch
from em.ditto.ditto_light.ditto_light.ditto import evaluate, DittoModel
from em.ditto.ditto_light.ditto_light.dataset import DittoDataset
import os
from torch.utils import data
from em.ditto.ditto_data_alter import DittoItunesAmazonAlter
import tqdm
def get_full_path(logdir, task):
    ckpt_path = os.path.join(logdir, task, 'model.pt')
    return ckpt_path

def do_eval(dataset):
    logdir = 'ditto_checkpoints/ditto_da/'
    lm = 'roberta-base'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DittoModel(device=device, lm=lm)

    saved_state = torch.load(get_full_path(logdir, dataset), map_location=lambda storage, loc: storage)
    # saved_state = torch.load(get_full_path(logdir, ""), map_location=lambda storage, loc: storage)
    model.load_state_dict(saved_state['model'])
    model = model.to(device)
    alter = [None, 'SFF', 'DRP', 'MIS', 'EXT', 'TYP']
    results = {}
    print(f'DITTO: Run alters on dataset: {dataset}')
    for tf in tqdm.tqdm(alter):
        testset = f'../data/augmented/ditto/1/{dataset}/test.txt'
        trainset = f'../data/augmented/ditto/1/{dataset}/train.txt'
        validset = f'../data/augmented/ditto/1/{dataset}/valid.txt'

        # load train/dev/test sets
        train_dataset = DittoDataset(trainset,
                                     lm=lm,
                                     max_len=256,
                                     size=None,
                                     da=None)



        total_f1 = 0
        try_count = 20
        for j in range(try_count):
            if tf:
                test_dataset = DittoDataset(testset, lm=lm, alter=DittoItunesAmazonAlter(mode=tf))
            else:
                test_dataset = DittoDataset(testset, lm=lm)

            padder = train_dataset.pad

            test_iter = data.DataLoader(dataset=test_dataset,
                                        batch_size=64 * 16,
                                        shuffle=False,
                                        num_workers=0,
                                        collate_fn=padder)
            dev_f1, th, mc_test = evaluate(model, test_iter)
            total_f1 += dev_f1

        results[tf] = (total_f1/try_count) * 100
        print(f'\nAlters:{tf} best F1: {results[tf] : .4f}\n')


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

    for ds in ['itunes_amazon']:
            ds_results = do_eval(dataset=ds)


if __name__ == '__main__':
    run_test()