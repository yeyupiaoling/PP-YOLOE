import argparse
import functools
import os

import paddle
from paddle.io import DataLoader

from data_utils.reader import CustomDataset
from metrics.metrics import COCOMetric
from model.yolo import PPYOLOE
from utils.logger import setup_logger
from utils.utils import add_arguments, print_arguments

logger = setup_logger(__name__)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('model_type',       str,    'M',                      '所使用PPYOLOE的模型类型', choices=["X", "L", "M", "S"])
add_arg('batch_size',       int,    8,                        '训练的批量大小')
add_arg('num_workers',      int,    4,                        '读取数据的线程数量')
add_arg('num_classes',      int,    80,                        '分类的类别数量')
add_arg('image_dir',        str,    'dataset/',               '图片存放的路径')
add_arg('eval_anno_path',   str,    'dataset/eval.json',      '评估标注信息json文件路径')
add_arg('resume_model',     str,    'output/PPYOLOE_M/best_model',  '恢复训练的模型文件夹，当为None则不使用恢复模型')
args = parser.parse_args()
print_arguments(args)


# 评估模型
def evaluate():
    # 评估数据
    eval_dataset = CustomDataset(image_dir=args.image_dir,
                                 anno_path=args.eval_anno_path,
                                 data_fields=['image'],
                                 mode='eval')
    eval_loader = DataLoader(dataset=eval_dataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)

    if args.model_type == 'X':
        model = PPYOLOE(num_classes=args.num_classes, depth_mult=1.33, width_mult=1.25)
    elif args.model_type == 'L':
        model = PPYOLOE(num_classes=args.num_classes, depth_mult=1.0, width_mult=1.0)
    elif args.model_type == 'M':
        model = PPYOLOE(num_classes=args.num_classes, depth_mult=0.67, width_mult=0.75)
    elif args.model_type == 'S':
        model = PPYOLOE(num_classes=args.num_classes, depth_mult=0.33, width_mult=0.50)
    else:
        raise Exception(f'模型类型不存在，model_type：{args.model_type}')

    metrics = COCOMetric(anno_file=args.eval_anno_path)

    # 加载恢复模型
    if args.resume_model is not None:
        assert os.path.exists(os.path.join(args.resume_model, 'model.pdparams')), "模型参数文件不存在！"
        model.set_state_dict(paddle.load(os.path.join(args.resume_model, 'model.pdparams')))
        logger.info('成功恢复模型参数和优化方法参数：{}'.format(args.resume_model))

    model.eval()
    for batch_id, data in enumerate(eval_loader()):
        outputs = model(data)
        metrics.update(inputs=data, outputs=outputs)
    mAP = metrics.accumulate()[0]
    metrics.reset()
    model.train()
    logger.info('mAP: {:.5f}'.format(mAP))


if __name__ == '__main__':
    evaluate()
