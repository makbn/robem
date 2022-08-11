from em.data.em_base_dataset import EmBaseDataset


class FodorsZagatsDataset(EmBaseDataset):

    def _get_alter_handler(self, alter):
        pass

    def _do_aug(self, label_row):
        left_can = self.tableA[label_row['ltable_id']]
        right_can = self.tableB[label_row['rtable_id']]

        left_augs = self.aug_del.augment(left_can[0], num=1)
        left_augs.extend(self.aug_stop.augment(left_can[0], num=1))
        left_augs.extend(self.aug_less.augment(left_can[0], num=1))

        right_augs = self.aug_del.augment(right_can[0], num=1)
        right_augs.extend(self.aug_stop.augment(right_can[0], num=1))
        right_augs.extend(self.aug_less.augment(right_can[0], num=1))

        aug_pairs = []
        for lc in left_augs:
            new_left = (lc.replace('[ UNK ]', '[UNK]'), left_can[1], left_can[2], left_can[3], left_can[4], left_can[5])
            for rc in right_augs:
                new_right = (rc.replace('[ UNK ]', '[UNK]'), right_can[1], right_can[2], right_can[3], right_can[4],
                             right_can[5])
                aug_pairs.append((new_left, new_right))

        aug_pairs.append((right_can, left_can))

        return aug_pairs

    def _get_mention(self, row):
        return row[0]

    def __init__(self, path, mode='train', transform=None, device=None, sentence_size=256, da=False, ditto_aug=None):
        super().__init__(path, mode, transform, device, sentence_size)
        self._read_tabel('tableA.csv', 'left', ['name', 'addr', 'city', 'phone', 'type', 'class'])
        self._read_tabel('tableB.csv', 'right', ['name', 'addr', 'city', 'phone', 'type', 'class'])
        if da:
            self._enable_pos_augmentation()

        if ditto_aug:
            self._enable_ditto_augmentation(ditto_aug)

        self._read_pairs()
