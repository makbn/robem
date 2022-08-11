import os.path
import pickle
from itertools import chain

from sklearn.feature_extraction.text import TfidfVectorizer

from em.data.em_base_dataset import EmBaseDataset


class CompanyDataset(EmBaseDataset):

    def _get_alter_handler(self, alter):
        pass

    def _do_aug(self, label_row):
        left_can = self.tableA[label_row['ltable_id']]
        right_can = self.tableB[label_row['rtable_id']]
        # augment description attr
        left_augs = self.aug_del.augment(left_can[1], num=1)
        left_augs.extend(self.aug_stop.augment(left_can[1], num=1))
        left_augs.extend(self.aug_less.augment(left_can[1], num=1))

        right_augs = self.aug_del.augment(right_can[1], num=1)
        right_augs.extend(self.aug_stop.augment(right_can[1], num=1))
        right_augs.extend(self.aug_less.augment(right_can[1], num=1))

        aug_pairs = []
        for lc in left_augs:
            new_left = (left_can[0], lc.replace('[ UNK ]', '[UNK]'), left_can[2])
            for rc in right_augs:
                new_right = (right_can[0], rc.replace('[ UNK ]', '[UNK]'), right_can[2])
                aug_pairs.append((new_left, new_right))

        aug_pairs.append((right_can, left_can))

        return aug_pairs

    def _get_mention(self, row):
        # this dataset does not contain any attribute that can be considered as a mention, so we need to extract
        # the unique part from content
        unique_part = self.tfidf_vectorizer.get_feature_names_out()[
            self.tfidf_vectorizer.transform([row[0]]).todense().argmax()]
        return unique_part

    def __init__(self, path, mode='train', transform=None, device=None, sentence_size=256, da=False, ditto_aug=None):
        super().__init__(path, mode, transform, device, sentence_size)

        # enable summarization for attr content
        self._enable_summarization(['content'])

        self._read_tabel('tableA.csv', 'left', ['content'])
        self._read_tabel('tableB.csv', 'right', ['content'])

        # we need this to exact mention from content
        #self.tfidf_model_initializer(path)

        if da:
            self._enable_pos_augmentation()

        if ditto_aug:
            self._enable_ditto_augmentation(ditto_aug)

        self._read_pairs()

    def tfidf_model_initializer(self, path):
        if os.path.exists(os.path.join(path, 'cached_tfidf.pickle')):
            print('loading tf-idf vectorizer from cache...')
            self.tfidf_vectorizer = pickle.load(open(os.path.join(path, 'cached_tfidf.pickle'), 'rb'))
        else:
            print('creating tf-idf vectorizer...')
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(2, 3))
            self.tfidf_vectorizer.fit_transform([row[0] for row in chain(self.tableA.values(), self.tableB.values())])
            pickle.dump(self.tfidf_vectorizer, open(os.path.join(path, 'cached_tfidf.pickle'), 'wb'))

        print('tf-idf is ready.')
