import copy

import numpy as np
from paddle.fluid.dataloader.collate import default_collate_fn

from data_utils.dataset import COCODataset
from data_utils.operators import RandomFlip, Resize, Permute, NormalizeImage
from data_utils.operators import Decode, RandomDistort, RandomExpand, RandomCrop
from data_utils.batch_operators import BatchRandomResize, BatchNormalizeImage, BatchPermute, BatchPadGT
from utils.logger import setup_logger

logger = setup_logger(__name__)


class CustomDataset(COCODataset):
    """
    Load dataset with COCO format.

    Args:
        image_dir (str): directory for images.
        anno_path (str): coco annotation file path.
        data_fields (list): key name of data dictionary, at least have 'image'.
        sample_num (int): number of samples to load, -1 means all.
        load_crowd (bool): whether to load crowded ground-truth. False as default
        allow_empty (bool): whether to load empty entry. False as default
        empty_ratio (float): the ratio of empty record number to total 
            record's, if empty_ratio is out of [0. ,1.), do not sample the 
            records and use all the empty entries. 1. as default
    """

    def __init__(self,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 mode='train',
                 sample_num=-1,
                 load_crowd=False,
                 allow_empty=False,
                 empty_ratio=1.,
                 eval_image_size=[640, 640],
                 use_random_distort=True,
                 use_random_expand=True,
                 use_random_crop=True,
                 use_random_flip=True):
        super(CustomDataset, self).__init__(image_dir, anno_path, data_fields,
                                            sample_num, load_crowd, allow_empty, empty_ratio)
        self._curr_iter = 0
        self.mode = mode
        assert self.mode in ['train', 'eval'], "数据处理模式不属于['train', 'eval']"
        # 数据预处理
        self.train_transform = [Decode()]
        self.eval_transform = [Decode(), Resize(target_size=eval_image_size), NormalizeImage(), Permute()]
        # 数据增强
        if use_random_distort:
            self.train_transform.append(RandomDistort())
        if use_random_expand:
            self.train_transform.append(RandomExpand())
        if use_random_crop:
            self.train_transform.append(RandomCrop())
        if use_random_flip:
            self.train_transform.append(RandomFlip())

    def __getitem__(self, idx):
        # data batch
        roidb = copy.deepcopy(self.roidbs[idx])
        roidb['curr_iter'] = self._curr_iter
        self._curr_iter += 1
        if self.mode == 'train':
            for t in self.train_transform:
                roidb = t(roidb)
        else:
            for t in self.eval_transform:
                roidb = t(roidb)
        return roidb


class BatchCompose(object):
    def __init__(self, collate_batch=True):
        self.collate_batch = collate_batch
        # 数据预处理和数据增强
        self.transform = [
            BatchRandomResize(target_size=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768],
                              random_size=True, random_interp=True, keep_ratio=False),
            BatchNormalizeImage(), BatchPermute(), BatchPadGT()]

    def __call__(self, data):
        for t in self.transform:
            try:
                data = t(data)
            except Exception as e:
                logger.warning("fail to map batch transform [{}] "
                               "with error: {}".format(t, e))
                raise e

        # remove keys which is not needed by model
        extra_key = ['h', 'w', 'flipped']
        for k in extra_key:
            for sample in data:
                if k in sample:
                    sample.pop(k)

        # batch data, if user-define batch function needed
        # use user-defined here
        if self.collate_batch:
            batch_data = default_collate_fn(data)
        else:
            batch_data = {}
            for k in data[0].keys():
                tmp_data = []
                for i in range(len(data)):
                    tmp_data.append(data[i][k])
                if not 'gt_' in k and not 'is_crowd' in k and not 'difficult' in k:
                    tmp_data = np.stack(tmp_data, axis=0)
                batch_data[k] = tmp_data
        return batch_data
