from em.data.abt_buy_dataset import AbtBuyDataset
from em.data.amz_google_dataset import AmazonGoogleDataset
from em.data.beer_rates_dataset import BeerRatesDataset
from em.data.company_dataset import CompanyDataset
from em.data.dblp_acm_dataset import DblpAcmDataset
from em.data.dblp_scholar import DblpScholarDataset
from em.data.em_list_dataset import ListDataset
from em.data.fodors_zagats_dataset import FodorsZagatsDataset
from em.data.itunes_amazon_dataset import ItunesAmazon
from em.data.walmart_amazon_dataset import WalmartAmazonDataset


def get_dataset(name, sentence_size, tokenizer, da, ditto_aug=None, sets=None):
    if sets is None:
        sets = ['train', 'val', 'test']
    dataset_map = {
        'itunes-amazon': (ItunesAmazon, 'itunes_amazon'),
        'dirty-itunes-amazon': (ItunesAmazon, 'dirty_itunes_amazon'),
        'amazon-google': (AmazonGoogleDataset, 'amazon_google'),
        'company': (CompanyDataset, 'company'),
        'abt-buy': (AbtBuyDataset, 'abt_buy'),
        'beer-rates': (BeerRatesDataset, 'beer_rates'),
        'fodors-zagats': (FodorsZagatsDataset, 'fodors_zagats'),
        'dblp-acm': (DblpAcmDataset, 'dblp_acm'),
        'dirty-dblp-acm': (DblpAcmDataset, 'dirty_dblp_acm'),
        'dblp-scholar': (DblpScholarDataset, 'dblp_scholar'),
        'dirty-dblp-scholar': (DblpScholarDataset, 'dirty_dblp_scholar'),
        'walmart-amazon': (WalmartAmazonDataset, 'walmart_amazon'),
        'dirty-walmart-amazon': (WalmartAmazonDataset, 'dirty_walmart_amazon')
    }
    name = name.lower()

    datasets_train = None
    datasets_valid = None
    datasets_test = None

    if name in dataset_map:
        dataset_class, dataset_dir = dataset_map[name]
        if 'train' in sets:
            datasets_train = dataset_class(f'../data/{dataset_dir}', mode='train', transform=[tokenizer],
                                           sentence_size=sentence_size, da=da, ditto_aug=ditto_aug)
        if 'val' in sets or 'validation' in sets or 'valid' in sets:
            datasets_valid = dataset_class(f'../data/{dataset_dir}', mode='valid', transform=[tokenizer],
                                           sentence_size=sentence_size)

        if 'test' in sets:
            datasets_test = dataset_class(f'../data/{dataset_dir}', mode='test', transform=[tokenizer],
                                          sentence_size=sentence_size)
    else:
        raise ValueError('dataset name is wrong!')

    return datasets_train, datasets_valid, datasets_test


def get_aug_dataset(name, sentence_size, tokenizer, sets=None, aug_size=1, alter=None):
    if sets is None:
        sets = ['train', 'val', 'test']
    dataset_map = {
        'itunes-amazon': 'itunes_amazon',
        'dirty-itunes-amazon': 'dirty_itunes_amazon',
        'amazon-google': 'amazon_google',
        'company': 'company',
        'abt-buy': 'abt_buy',
        'beer-rates': 'beer_rates',
        'fodors-zagats': 'fodors_zagats',
        'dblp-acm': 'dblp_acm',
        'dirty-dblp-acm': 'dirty_dblp_acm',
        'dblp-scholar': 'dblp_scholar',
        'dirty-dblp-scholar': 'dirty_dblp_scholar',
        'walmart-amazon': 'walmart_amazon',
        'dirty-walmart-amazon': 'dirty_walmart_amazon'
    }
    name = name.lower()

    datasets_train = None
    datasets_valid = None
    datasets_test = None
    if name in dataset_map:
        if 'train' in sets:
            datasets_train = ListDataset(f'../data/augmented/shuff/{dataset_map[name]}_train_{aug_size}.pkl', mode='train',
                                         transform=[tokenizer], sentence_size=sentence_size)
        if 'test' in sets:
            datasets_test = ListDataset(f'../data/augmented/shuff/{dataset_map[name]}_test_{aug_size}.pkl', mode='test',
                                        transform=[tokenizer], sentence_size=sentence_size, alter=alter)
        if 'val' in sets or 'validation' in sets or 'valid' in sets:
            datasets_valid = ListDataset(f'../data/augmented/shuff/{dataset_map[name]}_valid_{aug_size}.pkl',
                                         mode='valid', transform=[tokenizer], sentence_size=sentence_size)
    else:
        raise ValueError('dataset name is wrong!')

    return datasets_train, datasets_valid, datasets_test


def set_to_device(x, device):
    if 'input_ids' in x:
        x['input_ids'] = x['input_ids'].to(device)
    if 'attention_mask' in x:
        x['attention_mask'] = x['attention_mask'].to(device)
    if 'token_type_ids' in x:
        x['token_type_ids'] = x['token_type_ids'].to(device)
    return x
