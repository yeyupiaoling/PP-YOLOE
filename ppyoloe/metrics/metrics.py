import json
import os
import typing

import paddle
from pycocotools.coco import COCO

from ppyoloe.metrics.coco_utils import get_infer_results, cocoapi_eval

__all__ = ['COCOMetric']


class COCOMetric(object):
    def __init__(self, anno_file, **kwargs):
        self.anno_file = anno_file
        coco = COCO(anno_file)
        cats = coco.loadCats(coco.getCatIds())
        self.clsid2catid = {i: cat['id'] for i, cat in enumerate(cats)}
        self.catid2name = {cat['id']: cat['name'] for cat in cats}
        self.bbox_results = []
        self.eval_results = {}
        self.reset()

    def reset(self):
        self.bbox_results = []
        self.eval_results = {}

    def update(self, inputs, outputs):
        outs = {}
        # outputs Tensor -> numpy.ndarray
        for k, v in outputs.items():
            outs[k] = v.numpy() if isinstance(v, paddle.Tensor) else v

        # multi-scale inputs: all inputs have same im_id
        if isinstance(inputs, typing.Sequence):
            im_id = inputs[0]['im_id']
        else:
            im_id = inputs['im_id']
        outs['im_id'] = im_id.numpy() if isinstance(im_id, paddle.Tensor) else im_id

        infer_results = get_infer_results(outs, self.clsid2catid)
        self.bbox_results += infer_results['bbox'] if 'bbox' in infer_results else []

    def accumulate(self):
        output = f"bbox_{paddle.distributed.get_rank()}.json"
        with open(output, 'w') as f:
            json.dump(self.bbox_results, f)

        if len(self.bbox_results) == 0:return [0 for _ in range(12)]

        bbox_stats = cocoapi_eval(jsonfile=output,
                                  style='bbox',
                                  anno_file=self.anno_file)
        os.remove(output)
        return bbox_stats
