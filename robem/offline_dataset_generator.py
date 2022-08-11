import os
import pickle

import tqdm
from em.data.em_base_dataset import EmBaseDataset
from em.data.em_dataaug import RobustAugmenter


class GenDataset(EmBaseDataset):
    def __init__(self, path, mode='train', sentence_size=256, cols = None):
        super().__init__(path, mode, None, None, sentence_size)
        self._read_tabel('tableA.csv', 'left', cols)
        self._read_tabel('tableB.csv', 'right', cols)
        self._read_pairs()

    def _getrawitem(self, idx):
        left = self.pairs[idx][0]
        right = self.pairs[idx][1]
        left_full = f' {self.attr_sep} '.join([str(x) for x in left]).strip()
        right_full = f' {self.attr_sep} '.join([str(x) for x in right]).strip()
        label = self.labels[idx]

        return left_full, right_full, label

    def __getitem__(self, item):
        return self._getrawitem(item)


if __name__ == '__main__':
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
        # 'company'
    ]
    sets = ['train', 'val', 'test']

    dataset_map = {
        'itunes-amazon': (['Song_Name', 'Artist_Name', 'Album_Name', 'Genre',
                                                'Price', 'CopyRight', 'Time', 'Released'], 'itunes_amazon'),
        'dirty-itunes-amazon': (['Song_Name', 'Artist_Name', 'Album_Name', 'Genre',
                           'Price', 'CopyRight', 'Time', 'Released'], 'dirty_itunes_amazon'),
        'amazon-google': (['title', 'manufacturer', 'price'], 'amazon_google'),
        #'company': (['content'], 'company'),
        'abt-buy': (['name', 'description', 'price'], 'abt_buy'),
        'beer-rates': (['Beer_Name', 'Brew_Factory_Name', 'Style', 'ABV'], 'beer_rates'),
        'fodors-zagats': (['name', 'addr', 'city', 'phone', 'type', 'class'], 'fodors_zagats'),
        'dblp-acm': (['title', 'authors', 'venue', 'year'], 'dblp_acm'),
        'dirty-dblp-acm': (['title', 'authors', 'venue', 'year'], 'dirty_dblp_acm'),
        'dblp-scholar': (['title', 'authors', 'venue', 'year'], 'dblp_scholar'),
        'dirty-dblp-scholar': (['title', 'authors', 'venue', 'year'], 'dirty_dblp_scholar'),
        'walmart-amazon': (['title', 'category', 'brand', 'modelno', 'price'], 'walmart_amazon'),
        'dirty-walmart-amazon': (['title', 'category', 'brand', 'modelno', 'price'], 'dirty_walmart_amazon')
    }


    def _generate_ditto_line(x, y):
        new_sen = x[0].split(' [SEP] ')
        new_l = new_sen[0].split(' ATTR ')
        new_r = new_sen[1].split(' ATTR ')
        new_cols = x[1]

        l_str = ""
        r_str = ""
        for l_att, r_att, col in zip(new_l, new_r, new_cols):
            l_str += "COL " + col + " VAL " + l_att.replace('[UNK]', '') + " "
            r_str += "COL " + col + " VAL " + r_att.replace('[UNK]', '') + " "

        f_str = l_str.strip() + "\t" + r_str.strip() + "\t" + str(y)
        return f_str


    for ds_name in datasets:
        cols, ds_dir = dataset_map[ds_name]
        path = f'../data/{ds_dir}'
        rob_aug = RobustAugmenter(size=len(cols), cols=cols)

        for mode in ['train', 'test', 'valid']:
            ds = GenDataset(path=path, mode=mode, cols=cols)
            aug_ds_1p = []
            aug_ds_2p = []
            aug_ds_3p = []

            ditto_augs_1p = []
            ditto_augs_2p = []
            ditto_augs_3p = []

            for l, r, y in tqdm.tqdm(ds,desc=f'{ds_dir} {mode}'):
                # add original data point
                aug_ds_1p.append((l, r, y))
                aug_ds_2p.append((l, r, y))
                aug_ds_3p.append((l, r, y))



                sent = l + ' [SEP] ' + r

                # add original data point
                ditto_augs_1p.append(_generate_ditto_line((sent, cols), y))
                ditto_augs_2p.append(_generate_ditto_line((sent, cols), y))
                ditto_augs_3p.append(_generate_ditto_line((sent, cols), y))
                if mode == 'train':
                    augs = []
                    for i in range(3):
                        if i == 0:
                            n_sen, n_cols, swp = rob_aug.augment_sent(sent, disable_shuffle=False, disable_swap=True)
                        else:
                            n_sen, n_cols, swp = rob_aug.augment_sent(sent, fixed_shuffler=False, disable_swap=True)
                        augs.append((n_sen, n_cols, swp))

                    aug_ds_1p.append(tuple(augs[0][0].split(' [SEP] ')) + (y,))
                    ditto_augs_1p.append(_generate_ditto_line(augs[0], y))

                    for i in range(2):
                        aug_ds_2p.append(tuple(augs[i][0].split(' [SEP] ')) + (y,))
                        ditto_augs_2p.append(_generate_ditto_line(augs[i], y))

                    for i in range(3):
                        aug_ds_3p.append(tuple(augs[i][0].split(' [SEP] ')) + (y,))
                        ditto_augs_3p.append(_generate_ditto_line(augs[i], y))

            pickle.dump(aug_ds_1p, open(f'../data/augmented/shuff/{ds_dir}_{mode}_1.pkl', 'wb'))
            pickle.dump(aug_ds_2p, open(f'../data/augmented/shuff/{ds_dir}_{mode}_2.pkl', 'wb'))
            pickle.dump(aug_ds_3p, open(f'../data/augmented/shuff/{ds_dir}_{mode}_3.pkl', 'wb'))

            ditto_all_augs = [ditto_augs_1p, ditto_augs_2p, ditto_augs_3p]
            for idx in range(3):
                save_path = f'../data/augmented/ditto/shuff/{idx + 1}/{ds_dir}/{mode}.txt'
                try:
                    os.mkdir(f'../data/augmented/ditto/shuff/{idx + 1}/{ds_dir}/')
                except:
                    pass

                with open(save_path, 'w') as out_file:
                    for row in ditto_all_augs[idx]:
                        out_file.write(f'{row}\n')

            print(f'sizes:\toriginal:{len(ds)} 1:{len(aug_ds_1p)}\t\t2:{len(aug_ds_2p)}\t\t3:{len(aug_ds_3p)}')







