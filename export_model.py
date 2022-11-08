import argparse
import functools

from ppyoloe.trainer import PPYOLOETrainer
from ppyoloe.utils.logger import setup_logger
from ppyoloe.utils.utils import add_arguments, print_arguments

logger = setup_logger(__name__)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('model_type',       str,    'M',                            '所使用的模型类型', choices=["X", "L", "M", "S"])
add_arg('use_gpu',          bool,   True,                           '是否使用GPU')
add_arg('num_classes',      int,    80,                             '分类的类别数量')
add_arg('image_shape',      str,    '3,640,640',                    '导出模型图像输入大小')
add_arg('save_model_path',  str,    'models/',                      '导出模型保存的路径')
add_arg('resume_model',     str,    'models/PPYOLOE_M/best_model',  '恢复模型文件夹路径')
args = parser.parse_args()
print_arguments(args)


# 获取训练器
trainer = PPYOLOETrainer(model_type=args.model_type,
                         num_classes=args.num_classes,
                         use_gpu=args.use_gpu)

# 导出预测模型
trainer.export(save_model_path=args.save_model_path,
               image_shape=args.image_shape,
               resume_model=args.resume_model)
