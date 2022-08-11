import random
from enum import Enum
import random
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class AugMode(Enum):
    RANDOM_DEL = 0
    REMOVE_STOP_WORD = 1
    REMOVE_CHAR_LESS_THAN = 2
    ROB_ALL = 3
    ROB_SWAP = 4
    ROB_SFL = 5


class RobustAugmenter(object):

    def __init__(self, size, cols, sep_key=' [SEP] '):
        self.shuffle_order = self._generate_shuffle_order(size)
        self.sep_key = sep_key
        self.cols = cols

    @staticmethod
    def _generate_shuffle_order(size):
        from_list = [i for i in range(size)]
        to_list = [i for i in range(size)]
        random.shuffle(to_list)
        return [(i, j) for i, j in zip(from_list, to_list)]

    def _attr_shuffle(self, l_text, r_text, attr_key, fixed_shuffler=True):
        if fixed_shuffler:
            shuffler = self.shuffle_order
        else:
            shuffler = self._generate_shuffle_order(len(self.cols))

        l_atts = l_text.split(attr_key)
        r_atts = r_text.split(attr_key)
        new_l = [""] * len(l_atts)
        new_r = [""] * len(r_atts)

        new_cols = [""] * len(self.cols)
        for (i, j) in shuffler:
            new_l[i] = l_atts[j]
            new_r[i] = r_atts[j]
            new_cols[i] = self.cols[j]

        return attr_key.join(new_l), attr_key.join(new_r), new_cols

    def augment_sent(self, text, op: AugMode = AugMode.ROB_ALL, fixed_shuffler=True, force_swap=False, attr_key=' ATTR '
                     , disable_shuffle=False, disable_swap=False):
        if op not in [AugMode.ROB_ALL, AugMode.ROB_SWAP, AugMode.ROB_SFL]:
            raise ValueError('augmentation operator is not valid')

        left, right = text.split(self.sep_key)
        cols = self.cols

        swap = False
        if not disable_swap and self.sep_key in text and (random.randint(0, 1) == 0 or force_swap):
            right, left = left, right
            swap = True

        if not disable_shuffle:
            left, right, cols = self._attr_shuffle(left, right, attr_key=attr_key, fixed_shuffler=fixed_shuffler)

        return left + self.sep_key + right, cols, swap


class BasicAug:
    def __init__(self, mode: AugMode = AugMode.RANDOM_DEL, **kwargs):
        self.mode = mode
        self.wt = word_tokenize
        if mode == AugMode.REMOVE_STOP_WORD:
            self.stopwords = set(stopwords.words('english'))

        if mode == AugMode.REMOVE_CHAR_LESS_THAN:
            self.size_th = kwargs['min_size']

    def augment(self, text, num=1):
        return self._get_method()(text, num)

    def _random_del_augment(self, text, num):
        words = self.wt(text)
        sentences = []
        for i in range(num):
            new_sentence = []
            for word in words:
                if random.random() <= 0.8:
                    new_sentence.append(word)
            sentences.append(" ".join(new_sentence))
        return sentences

    def _remove_stop_word_augment(self, text, num):
        words = self.wt(text)
        sentences = []
        skip_list = set()
        for i in range(num):
            new_sentence = []
            found = False
            for word in words:
                if word in self.stopwords and word not in skip_list and not found:
                    skip_list.add(word)
                    found = True
                else:
                    new_sentence.append(word)
            sentences.append(" ".join(new_sentence))
        return sentences

    def _remove_less_than_augment(self, text, num):
        words = self.wt(text)
        sentences = []
        skip_list = set()
        for i in range(num):
            new_sentence = []
            found = False
            for word in words:
                if len(word) <= self.size_th and word not in skip_list and not found:
                    skip_list.add(word)
                    found = True
                else:
                    new_sentence.append(word)
            sentences.append(" ".join(new_sentence))
        return sentences

    def _get_method(self):
        if self.mode == AugMode.RANDOM_DEL:
            return self._random_del_augment
        elif self.mode == AugMode.REMOVE_STOP_WORD:
            return self._remove_stop_word_augment
        elif self.mode == AugMode.REMOVE_CHAR_LESS_THAN:
            return self._remove_less_than_augment
        else:
            raise ValueError('wrong augmentation method')


if __name__ == '__main__':
    aug_del = BasicAug(mode=AugMode.RANDOM_DEL)
    aug_stop = BasicAug(mode=AugMode.REMOVE_STOP_WORD)
    aug_less = BasicAug(mode=AugMode.REMOVE_CHAR_LESS_THAN, min_size=3)
    sample = 'whirlpool them duet wfw9200sq white front load washer wfw9200swh 4.0 cu . ft. capacity 6th sense technology ' \
             'quiet wash plus noise reduction built-in water heater add-a-garment feature sanitary cycle 4 temperature ' \
             'selections white finish'

    print([x == sample for x in aug_del.augment(sample, num=5)])
    print([x == sample for x in aug_stop.augment(sample, num=5)])
    print([x == sample for x in aug_less.augment(sample, num=5)])


if __name__ == '__main__':
    ag = RobustAugmenter()
    text = 'first attribute value sentence ATTR second attribute value with some additional token ATTR just a year like 2021' \
           ' [SEP] sigmod conference 2010 papers 2019-12-31 ATTR here the second one ATTR just another year 2012'
    print(text)
    for op in [AugMode.ROB_ALL, AugMode.ROB_ALL, AugMode.ROB_ALL, AugMode.ROB_ALL, AugMode.ROB_ALL, AugMode.ROB_ALL
        , AugMode.ROB_ALL, AugMode.ROB_ALL, AugMode.ROB_ALL, AugMode.ROB_ALL, AugMode.ROB_ALL, AugMode.ROB_ALL]:
        print(op)
        print(ag.augment_sent(text, op=op))

