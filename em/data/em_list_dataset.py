import pickle

from em.data.em_base_dataset import EmBaseDataset
from em.data.em_data_alter import ItunesAmazonAlter


class ListDataset(EmBaseDataset):

    def __init__(self, path, mode='train', transform=None, device=None, sentence_size=256, alter=None):
        super().__init__(path, mode, transform, device, sentence_size)
        self._read_pairs_pickle(path)
        if alter:
            self._set_alter(alter)

    def _get_alter_handler(self, alter):
        if 'itunes_amazon' in self.path:
            return ItunesAmazonAlter(mode=alter, attr_sep=self.attr_sep)
        else:
            raise NotImplemented('alter is not implemented for this dataset!')

    def _read_pairs_pickle(self, path):
        data = pickle.load(open(path, 'rb'))

        for inx, row in enumerate(data):
            if self.pos_counter >= self.class_num_limit and self.neg_counter >= self.class_num_limit:
                break

            l_list = row[0].split(' ATTR ')
            r_list = row[1].split(' ATTR ')

            if row[2] == 1 and self.pos_counter < self.class_num_limit:

                self.pairs.append((l_list, r_list))
                self.labels.append(1)
                self.pos_counter += 1

            if row[2] == 0 and self.neg_counter < self.class_num_limit:
                self.pairs.append((l_list, r_list))
                self.labels.append(0)
                self.neg_counter += 1

        print(f'Dataset: {path.replace("../data/augmented/", "").replace(".pkl", "")} Label 0:{self.neg_counter}\t1:{self.pos_counter} in {self.mode} set!'
              f'\tMissing docs: 0')

    def _get_mention(self, row):
        return row[0]




