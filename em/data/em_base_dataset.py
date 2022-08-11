import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from em.data.em_dataaug import BasicAug, AugMode, RobustAugmenter
from em.data.em_summerizer import Summarizer
from em.ditto.augment import Augmenter


class EmBaseDataset(Dataset):
    def __init__(self, path, mode='train', transform=None, device=None, sentence_size=256):
        self.transform = transform
        self.path = path
        self.train = (mode == 'train')
        self.mode = mode
        self.pairs = []
        self.labels = []
        self.tableA = {}
        self.sentence_size = sentence_size
        self.tableB = {}
        self.device = device
        self.pos_counter = 0
        self.neg_counter = 0

        if self.transform and len(self.transform[0].additional_special_tokens) > 0 and self.transform[0]:
            self.attr_sep = self.transform[0].additional_special_tokens[0]
        else:
            self.attr_sep = "ATTR"

        self.class_num_limit = 999999
        self.pos_aug = False
        self.neg_aug = False
        self.ditto_aug = None
        self.do_summarization = False
        self.ditto_augmentor = None
        self.aug_del = BasicAug(mode=AugMode.RANDOM_DEL)
        self.aug_stop = BasicAug(mode=AugMode.REMOVE_STOP_WORD)
        self.aug_less = BasicAug(mode=AugMode.REMOVE_CHAR_LESS_THAN, min_size=3)
        self.alter = None
        self.summarizer = Summarizer()

    def _set_alter(self, alter):
        self.alter = alter
        self.alter_handler = self._get_alter_handler(alter)

    def _set_class_num_limit(self, class_num_limit):
        self.class_num_limit = class_num_limit

    def _enable_pos_augmentation(self):
        self.pos_aug = True

    def _enable_ditto_augmentation(self, type):
        self.ditto_aug = type
        self.ditto_augmentor = Augmenter()

    def _enable_neg_augmentation(self):
        self.pos_aug = True

    def _enable_summarization(self, col_list=None):
        if col_list is None:
            col_list = []
        self.do_summarization = True
        self.summ_candidates = col_list

    def _read_tabel(self, table='tableA.csv', entity='left', cols=None):
        if cols is None:
            cols = []
        if entity is not 'left' and entity is not 'right':
            raise ValueError('entity should be `left` or `right` for em dataset')

        entities = {'left': self.tableA, 'right': self.tableB}
        df = pd.read_csv(os.path.join(self.path, table))

        for inx, row in df.iterrows():
            table_entry = ()
            for col in cols:
                value = "<UNK>" if (pd.isnull(row[col])) else row[col]
                if self.do_summarization and col in self.summ_candidates:
                    value = self.summarizer.summarize(value, self.sentence_size)
                table_entry = table_entry + (value,)

            entities[entity][row['id']] = table_entry

    def _read_pairs(self):
        if self.mode == 'train':
            df = pd.read_csv(os.path.join(self.path, 'train.csv'))
        elif self.mode == 'test':
            df = pd.read_csv(os.path.join(self.path, 'test.csv'))
        else:
            df = pd.read_csv(os.path.join(self.path, 'valid.csv'))

        missing_docs = 0

        for inx, row in df.iterrows():
            if self.pos_counter >= self.class_num_limit and self.neg_counter >= self.class_num_limit:
                break

            if not (row['ltable_id'] in self.tableA and row['rtable_id'] in self.tableB):
                missing_docs += 1
                continue

            if row['label'] == 1 and self.pos_counter < self.class_num_limit:
                self.pairs.append((self.tableA[row['ltable_id']], self.tableB[row['rtable_id']]))
                self.labels.append(1)
                self.pos_counter += 1

                if self.train and self.pos_aug:
                    augmented_rows = self._do_aug(label_row=row)
                    for ar in augmented_rows:
                        self.pairs.append((ar[0], ar[1]))
                        self.labels.append(1)
                        self.pos_counter += 1

            if row['label'] == 0 and self.neg_counter < self.class_num_limit:
                self.pairs.append((self.tableA[row['ltable_id']], self.tableB[row['rtable_id']]))
                self.labels.append(0)
                self.neg_counter += 1

                if self.train and self.neg_aug:
                    augmented_rows = self._do_aug(label_row=row)
                    for ar in augmented_rows:
                        self.pairs.append((ar[0], ar[1]))
                        self.labels.append(0)
                        self.pos_counter += 1

        print(f'Dataset: {self.__class__.__name__} Label 0:{self.neg_counter}\t1:{self.pos_counter} in {self.mode} set!'
              f'\tMissing docs: {missing_docs}')

    def _do_aug(self, label_row):
        """
        get label_row = {ltable_id, rtable_id, label} as input to generate new entry! use ltable_id and rtable_id to
        load data from self.tableA and self.tableB
        :param label_row: series
        :return: list of new tuple (left, right)
        """
        raise NotImplementedError

    def _get_mention(self, row):
        raise NotImplementedError

    def _getrawitem(self, idx):
        left = self.pairs[idx][0]
        right = self.pairs[idx][1]
        left_full = f' {self.attr_sep} '.join([str(x) for x in left]).strip()
        right_full = f' {self.attr_sep} '.join([str(x) for x in right]).strip()

        label = self.labels[idx]

        return [str(self._get_mention(left)), left_full], \
               [str(self._get_mention(right)), right_full], label

    def __getitem__(self, idx):
        left = self.pairs[idx][0]
        right = self.pairs[idx][1]

        left_full = f' {self.attr_sep} '.join([str(x) for x in left]).strip()
        right_full = f' {self.attr_sep} '.join([str(x) for x in right]).strip()

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.alter:
            left, left_full, right, right_full = self.alter_handler.alter(left, left_full, right, right_full)

        concat_orig, left_orig, right_orig = self.input_serializer(left, left_full, right, right_full)

        if self.ditto_aug is not None:
            combined = None
            flipped = False
            for i in range(15):
                combined, flipped = self.ditto_augmentor.augment_sent(left_full + ' [SEP] ' + right_full, self.ditto_aug)
                if len(combined.split(' [SEP] ')) == 2:
                    break

            if combined is None:
                print(f'failed to generate augmentation on idx {idx}!')
                return [(left_orig, right_orig, concat_orig, label)]

            aug_left, aug_right = combined.split(' [SEP] ')
            if flipped:
                aug_concat, aug_left, aug_right = self.input_serializer(right, aug_left, left, aug_right)
            else:
                aug_concat, aug_left, aug_right = self.input_serializer(left, aug_left, right, aug_right)

            return [(left_orig, right_orig, concat_orig, label), (aug_left, aug_right, aug_concat, label)]
        else:
            return [(left_orig, right_orig, concat_orig, label)]

    def input_serializer(self, left, left_full, right, right_full):
        inputs = [
            [str(self._get_mention(left)), left_full],  # left
            [str(self._get_mention(right)), right_full],  # right
        ]
        lef_right = [self.transform[0].encode_plus(text=x_tuple[0], text_pair=x_tuple[1],
                                                   add_special_tokens=True,
                                                   max_length=self.sentence_size,
                                                   truncation_strategy="only_second",
                                                   pad_to_max_length=True,
                                                   return_tensors="pt",
                                                   truncation=True
                                                   ) for x_tuple in inputs]
        # context emb
        concat = self.transform[0].encode_plus(text=left_full, text_pair=right_full,
                                               add_special_tokens=True,
                                               max_length=self.sentence_size,
                                               truncation_strategy="only_second",
                                               pad_to_max_length=True,
                                               return_tensors="pt",
                                               truncation=True
                                               )
        del inputs
        left, right, concat = self._spilt(lef_right, concat)
        return concat, left, right

    @staticmethod
    def _spilt(x, concat):
        left = {
            'input_ids': x[0]['input_ids'][0],
            'attention_mask': x[0]['attention_mask'][0]
        }
        right = {
            'input_ids': x[1]['input_ids'][0],
            'attention_mask': x[1]['attention_mask'][0]
        }
        concat_new = {
            'input_ids': concat['input_ids'][0],
            'attention_mask': concat['attention_mask'][0],
        }

        if 'token_type_ids' in concat:
            concat_new['token_type_ids'] = concat['token_type_ids'][0]

        if 'token_type_ids' in x[0] and 'token_type_ids' in x[1]:
            left['token_type_ids'] = x[0]['token_type_ids'][0]
            right['token_type_ids'] = x[1]['token_type_ids'][0]

        return left, right, concat_new

    def __len__(self):
        return len(self.pairs)

    def _get_alter_handler(self, alter):
        raise NotImplementedError

