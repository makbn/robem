import os

import pickle
import tqdm

def get_load_path(args, ds_train_name):
    name = "{}-{}-{}-{}-{}-model.pt".format(args.lm, ds_train_name, 'deep' if args.deep else 'simple', args.loss
                                            , 'da' if args.da else 'no')
    full_path = os.path.join(args.save_dir, name)
    return full_path


if __name__ == '__main__':

    dataset_map = {
        'itunes-amazon': (['Song_Name', 'Artist_Name', 'Album_Name', 'Genre',
                           'Price', 'CopyRight', 'Time', 'Released'], 'itunes_amazon'),
        'amazon-google': (['title', 'manufacturer', 'price'], 'amazon_google'),
        # 'company': (['content'], 'company'),
        'abt-buy': (['name', 'description', 'price'], 'abt_buy'),
        'beer-rates': (['Beer_Name', 'Brew_Factory_Name', 'Style', 'ABV'], 'beer_rates'),
        'fodors-zagats': (['name', 'addr', 'city', 'phone', 'type', 'class'], 'fodors_zagats'),
        'dblp-acm': (['title', 'authors', 'venue', 'year'], 'dblp_acm'),
        'dblp-scholar': (['title', 'authors', 'venue', 'year'], 'dblp_scholar'),
        'walmart-amazon': (['title', 'category', 'brand', 'modelno', 'price'], 'walmart_amazon')
    }

    aug_size = 1

    pairs = []
    for ds_name, (cols, dir) in dataset_map.items():
        try:
            os.mkdir(f'../../data/augmented/ditto/{aug_size}/{dir}/')
        except:
            print('dir exists!')
        for mode in ['train', 'test', 'valid']:
            data = pickle.load(open(f'../../data/augmented/{dir}_{mode}_{aug_size}.pkl', 'rb'))

            save_path = f'../../data/augmented/ditto/{aug_size}/{dir}/{mode}.txt'
            with open(save_path, 'w') as out_file:
                for inx, row in tqdm.tqdm(enumerate(data)):

                    l_list = row[0].split(' ATTR ')
                    r_list = row[1].split(' ATTR ')
                    label = row[2]
                    l_str = ""
                    r_str = ""

                    for l_att, r_att, col in zip(l_list, r_list, cols):
                        l_str += "COL " + col + " VAL " + l_att.replace('[UNK]', '') + " "
                        r_str += "COL " + col + " VAL " + r_att.replace('[UNK]', '') + " "

                    f_str = l_str.strip() + "\t" + r_str.strip() + "\t" + str(label)
                    out_file.write(f'{f_str}\n')
            print(f'Dataset: {ds_name} mode: {mode}, in: {save_path} saved!')
