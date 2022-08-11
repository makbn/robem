from em.data.em_base_dataset import EmBaseDataset
from em.data.em_data_alter import ItunesAmazonAlter


class ItunesAmazon(EmBaseDataset):

    def _get_alter_handler(self, alter):
        return ItunesAmazonAlter(mode=alter, attr_sep=self.attr_sep)

    def _do_aug(self, label_row):
        aug_pairs = []
        left_can = self.tableA[label_row['ltable_id']]
        right_can = self.tableB[label_row['rtable_id']]

        aug_pairs.append((right_can, left_can))

        aug_pairs.append(
            (
                (left_can[0], left_can[1], left_can[2], '[UNK]',
                 left_can[4], left_can[5], left_can[6], left_can[7]),

                (right_can[0], right_can[1], right_can[2], right_can[3],
                 right_can[4], right_can[5], right_can[6], right_can[7])
            )
        )
        aug_pairs.append(
            (
                (left_can[0], left_can[1], left_can[2], '[UNK]',
                 left_can[4], left_can[5], left_can[6], left_can[7]),

                (right_can[0], right_can[1], right_can[2], right_can[3],
                 right_can[4], right_can[5], right_can[6], right_can[7])
            )
        )

        return aug_pairs

    def _get_mention(self, row):
        # row[0] is 'Song_Name'
        return row[0]

    def __init__(self, path, mode='train', transform=None, device=None, sentence_size=256, da=False, ditto_aug=None):
        super().__init__(path, mode, transform, device, sentence_size)

        self._read_tabel('tableA.csv', 'left', ['Song_Name', 'Artist_Name', 'Album_Name', 'Genre',
                                                'Price', 'CopyRight', 'Time', 'Released'])
        self._read_tabel('tableB.csv', 'right', ['Song_Name', 'Artist_Name', 'Album_Name', 'Genre',
                                                'Price', 'CopyRight', 'Time', 'Released'])
        if da:
            self._enable_pos_augmentation()

        if ditto_aug:
            self._enable_ditto_augmentation(ditto_aug)

        self._read_pairs()

