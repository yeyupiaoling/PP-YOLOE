import argparse
import functools
import time

from ppyoloe.trainer import PPYOLOETrainer
from ppyoloe.utils.utils import add_arguments, print_arguments
from ppyoloe.utils.utils import setup_logger

logger = setup_logger(__name__)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('model_type',       str,    'M',                           '所使用PPYOLOE的模型类型', choices=["X", "L", "M", "S"])
add_arg('use_gpu',          bool,   True,                          '是否使用GPU')
add_arg('batch_size',       int,    8,                             '训练的批量大小')
add_arg('num_workers',      int,    4,                             '读取数据的线程数量')
add_arg('num_classes',      int,    80,                            '分类的类别数量')
add_arg('image_size',       str,    '640,640',                     '评估时图像输入大小')
add_arg('image_dir',        str,    'dataset/',                    '图片存放的路径')
add_arg('eval_anno_path',   str,    'dataset/eval.json',           '评估标注信息json文件路径')
add_arg('resume_model',     str,    'models/PPYOLOE_M/best_model', '恢复模型文件夹路径')
args = parser.parse_args()
print_arguments(args)


# 获取训练器
trainer = PPYOLOETrainer(model_type=args.model_type,
                         batch_size=args.batch_size,
                         num_workers=args.num_workers,
                         num_classes=args.num_classes,
                         image_dir=args.image_dir,
                         eval_anno_path=args.eval_anno_path,
                         use_gpu=args.use_gpu)


# 开始评估
start = time.time()
mAP = trainer.evaluate(image_size=args.image_size, resume_model=args.resume_model)[0]
end = time.time()
print('评估消耗时间：{}s，mAP：{:.5f}'.format(int(end - start), mAP))
