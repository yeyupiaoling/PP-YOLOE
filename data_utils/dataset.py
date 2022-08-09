import os

import numpy as np
from paddle.io import Dataset

from utils.logger import setup_logger

logger = setup_logger(__name__)


class COCODataset(Dataset):
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
                 sample_num=-1,
                 load_crowd=False,
                 allow_empty=False,
                 empty_ratio=1.):
        super(COCODataset, self).__init__()
        self.cname2cid = None
        self.catid2clsid = None
        self.load_image_only = False
        self.image_dir = image_dir
        self.anno_path = anno_path
        self.data_fields = data_fields
        self.sample_num = sample_num
        self.load_crowd = load_crowd
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio
        self.parse_dataset()

    def _sample_empty(self, records, num):
        # if empty_ratio is out of [0. ,1.), do not sample the records
        if self.empty_ratio < 0. or self.empty_ratio >= 1.:
            return records
        import random
        sample_num = min(int(num * self.empty_ratio / (1 - self.empty_ratio)), len(records))
        records = random.sample(records, sample_num)
        return records

    def parse_dataset(self):
        assert self.anno_path.endswith('.json'), 'invalid coco annotation file: ' + self.anno_path
        from pycocotools.coco import COCO
        coco = COCO(self.anno_path)
        img_ids = coco.getImgIds()
        img_ids.sort()
        cat_ids = coco.getCatIds()
        records = []
        empty_records = []
        ct = 0

        self.catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
        self.cname2cid = dict({coco.loadCats(catid)[0]['name']: clsid
                               for catid, clsid in self.catid2clsid.items()})

        if 'annotations' not in coco.dataset:
            self.load_image_only = True
            logger.warning('Annotation file: {} does not contains ground truth '
                           'and load image information only.'.format(self.anno_path))

        for img_id in img_ids:
            img_anno = coco.loadImgs([img_id])[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])

            im_path = os.path.join(self.image_dir, im_fname) if self.image_dir else im_fname
            is_empty = False
            if not os.path.exists(im_path):
                logger.warning('Illegal image file: {}, and it will be ignored'.format(im_path))
                continue

            if im_w < 0 or im_h < 0:
                logger.warning('Illegal width: {} or height: {} in annotation, '
                               'and im_id: {} will be ignored'.format(im_w, im_h, img_id))
                continue

            coco_rec = {'im_file': im_path, 'im_id': np.array([img_id]), 'h': im_h,
                        'w': im_w, } if 'image' in self.data_fields else {}

            if not self.load_image_only:
                ins_anno_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None if self.load_crowd else False)
                instances = coco.loadAnns(ins_anno_ids)

                bboxes = []
                is_rbox_anno = False
                for inst in instances:
                    # check gt bbox
                    if inst.get('ignore', False):
                        continue
                    if 'bbox' not in inst.keys():
                        continue
                    else:
                        if not any(np.array(inst['bbox'])):
                            continue

                    # read rbox anno or not
                    is_rbox_anno = True if len(inst['bbox']) == 5 else False
                    if is_rbox_anno:
                        xc, yc, box_w, box_h, angle = inst['bbox']
                        x1 = xc - box_w / 2.0
                        y1 = yc - box_h / 2.0
                        x2 = x1 + box_w
                        y2 = y1 + box_h
                    else:
                        x1, y1, box_w, box_h = inst['bbox']
                        x2 = x1 + box_w
                        y2 = y1 + box_h
                    eps = 1e-5
                    if inst['area'] > 0 and x2 - x1 > eps and y2 - y1 > eps:
                        inst['clean_bbox'] = [round(float(x), 3) for x in [x1, y1, x2, y2]]
                        if is_rbox_anno:
                            inst['clean_rbox'] = [xc, yc, box_w, box_h, angle]
                        bboxes.append(inst)
                    else:
                        logger.warning('Found an invalid bbox in annotations: im_id: {}, '
                                       'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                            img_id, float(inst['area']), x1, y1, x2, y2))

                num_bbox = len(bboxes)
                if num_bbox <= 0 and not self.allow_empty:
                    continue
                elif num_bbox <= 0:
                    is_empty = True

                gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
                if is_rbox_anno:
                    gt_rbox = np.zeros((num_bbox, 5), dtype=np.float32)
                gt_theta = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
                is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_poly = [None] * num_bbox

                has_segmentation = False
                for i, box in enumerate(bboxes):
                    catid = box['category_id']
                    gt_class[i][0] = self.catid2clsid[catid]
                    gt_bbox[i, :] = box['clean_bbox']
                    # xc, yc, w, h, theta
                    if is_rbox_anno:
                        gt_rbox[i, :] = box['clean_rbox']
                    is_crowd[i][0] = box['iscrowd']
                    # check RLE format
                    if 'segmentation' in box and box['iscrowd'] == 1:
                        gt_poly[i] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
                    elif 'segmentation' in box and box['segmentation']:
                        if not np.array(box['segmentation']).size > 0 and not self.allow_empty:
                            bboxes.pop(i)
                            gt_poly.pop(i)
                            np.delete(is_crowd, i)
                            np.delete(gt_class, i)
                            np.delete(gt_bbox, i)
                        else:
                            gt_poly[i] = box['segmentation']
                        has_segmentation = True

                if has_segmentation and not any(gt_poly) and not self.allow_empty:
                    continue

                if is_rbox_anno:
                    gt_rec = {
                        'is_crowd': is_crowd,
                        'gt_class': gt_class,
                        'gt_bbox': gt_bbox,
                        'gt_rbox': gt_rbox,
                        'gt_poly': gt_poly,
                    }
                else:
                    gt_rec = {
                        'is_crowd': is_crowd,
                        'gt_class': gt_class,
                        'gt_bbox': gt_bbox,
                        'gt_poly': gt_poly,
                    }

                for k, v in gt_rec.items():
                    if k in self.data_fields:
                        coco_rec[k] = v

            logger.debug('Load file: {}, im_id: {}, h: {}, w: {}.'.format(im_path, img_id, im_h, im_w))
            if is_empty:
                empty_records.append(coco_rec)
            else:
                records.append(coco_rec)
            ct += 1
            if 0 < self.sample_num <= ct:
                break
        assert ct > 0, f'not found any coco record in {self.anno_path}'
        logger.debug('{} samples in file {}'.format(ct, self.anno_path))
        if self.allow_empty and len(empty_records) > 0:
            empty_records = self._sample_empty(empty_records, len(records))
            records += empty_records
        self.roidbs = records

    def __getitem__(self, idx):
        pass

    def __len__(self, ):
        return len(self.roidbs)

