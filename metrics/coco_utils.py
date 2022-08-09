import itertools
import sys

import numpy as np
from utils.logger import setup_logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logger = setup_logger(__name__)


def get_det_res(bboxes, bbox_nums, image_id, label_to_cat_id_map, bias=0):
    det_res = []
    k = 0
    for i in range(len(bbox_nums)):
        cur_image_id = int(image_id[i][0])
        det_nums = bbox_nums[i]
        for j in range(det_nums):
            dt = bboxes[k]
            k = k + 1
            num_id, score, xmin, ymin, xmax, ymax = dt.tolist()
            if int(num_id) < 0:
                continue
            category_id = label_to_cat_id_map[int(num_id)]
            w = xmax - xmin + bias
            h = ymax - ymin + bias
            bbox = [xmin, ymin, w, h]
            dt_res = {
                'image_id': cur_image_id,
                'category_id': category_id,
                'bbox': bbox,
                'score': score
            }
            det_res.append(dt_res)
    return det_res


def get_det_poly_res(bboxes, bbox_nums, image_id, label_to_cat_id_map, bias=0):
    det_res = []
    k = 0
    for i in range(len(bbox_nums)):
        cur_image_id = int(image_id[i][0])
        det_nums = bbox_nums[i]
        for j in range(det_nums):
            dt = bboxes[k]
            k = k + 1
            num_id, score, x1, y1, x2, y2, x3, y3, x4, y4 = dt.tolist()
            if int(num_id) < 0:
                continue
            category_id = label_to_cat_id_map[int(num_id)]
            rbox = [x1, y1, x2, y2, x3, y3, x4, y4]
            dt_res = {
                'image_id': cur_image_id,
                'category_id': category_id,
                'bbox': rbox,
                'score': score
            }
            det_res.append(dt_res)
    return det_res


def get_infer_results(outs, catid, bias=0):
    """
    Get result at the stage of inference.
    The output format is dictionary containing bbox or mask result.

    For example, bbox result is a list and each element contains
    image_id, category_id, bbox and score.
    """
    if outs is None or len(outs) == 0:
        raise ValueError(
            'The number of valid detection result if zero. Please use reasonable model and check input data.')

    im_id = outs['im_id']

    infer_res = {}
    if len(outs['bbox']) > 0 and len(outs['bbox'][0]) > 6:
        infer_res['bbox'] = get_det_poly_res(outs['bbox'], outs['bbox_num'], im_id, catid, bias=bias)
    else:
        infer_res['bbox'] = get_det_res(outs['bbox'], outs['bbox_num'], im_id, catid, bias=bias)

    return infer_res


def cocoapi_eval(jsonfile,
                 style,
                 coco_gt=None,
                 anno_file=None,
                 classwise=False):
    """
    Args:
        jsonfile (str): Evaluation json file, eg: bbox.json, mask.json.
        style (str): COCOeval style, can be `bbox` , `segm` , `proposal`, `keypoints` and `keypoints_crowd`.
        coco_gt (str): Whether to load COCOAPI through anno_file,
                 eg: coco_gt = COCO(anno_file)
        anno_file (str): COCO annotations file.
        classwise (bool): Whether per-category AP and draw P-R Curve or not.
    """
    assert coco_gt is not None or anno_file is not None

    if coco_gt is None:
        coco_gt = COCO(anno_file)
    logger.info("Start evaluate...")
    coco_dt = coco_gt.loadRes(jsonfile)
    coco_eval = COCOeval(coco_gt, coco_dt, style)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    if classwise:
        # Compute per-category AP and PR curve
        try:
            from terminaltables import AsciiTable
        except Exception as e:
            logger.error(
                'terminaltables not found, plaese install terminaltables. '
                'for example: `pip install terminaltables`.')
            raise e
        precisions = coco_eval.eval['precision']
        cat_ids = coco_gt.getCatIds()
        # precision: (iou, recall, cls, area range, max dets)
        assert len(cat_ids) == precisions.shape[2]
        results_per_category = []
        for idx, catId in enumerate(cat_ids):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            nm = coco_gt.loadCats(catId)[0]
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            if precision.size:
                ap = np.mean(precision)
            else:
                ap = float('nan')
            results_per_category.append((str(nm["name"]), '{:0.3f}'.format(float(ap))))

        num_columns = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        headers = ['category', 'AP'] * (num_columns // 2)
        results_2d = itertools.zip_longest(
            *[results_flatten[i::num_columns] for i in range(num_columns)])
        table_data = [headers]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data)
        logger.info('Per-category of {} AP: \n{}'.format(style, table.table))
        logger.info("per-category PR curve has output to {} folder.".format(
            style + '_pr_curve'))
    # flush coco evaluation result
    sys.stdout.flush()
    return coco_eval.stats
