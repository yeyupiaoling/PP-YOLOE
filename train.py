import argparse
import functools

from ppyoloe.trainer import PPYOLOETrainer
from ppyoloe.utils.logger import setup_logger
from ppyoloe.utils.utils import add_arguments, print_arguments

logger = setup_logger(__name__)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('model_type',          str,    'M',                      '所使用的模型类型', choices=["X", "L", "M", "S"])
add_arg('use_gpu',             bool,   True,                     '是否使用GPU')
add_arg('batch_size',          int,    8,                        '训练的批量大小')
add_arg('num_workers',         int,    8,                        '读取数据的线程数量')
add_arg('num_epoch',           int,    80,                       '训练的轮数')
add_arg('num_classes',         int,    80,                       '分类的类别数量')
add_arg('learning_rate',       float,  1.25e-4,                  '初始学习率的大小')
add_arg('log_interval',        int,    100,                      '指定步数打印一次日志')
add_arg('image_dir',           str,    'dataset/',               '图片存放的路径')
add_arg('train_anno_path',     str,    'dataset/train.json',     '训练数据标注信息json文件路径')
add_arg('eval_anno_path',      str,    'dataset/eval.json',      '评估标注信息json文件路径')
add_arg('save_model_path',     str,    'models/',                '模型保存的路径')
add_arg('use_random_distort',  bool,   True,                     '是否使用随机颜色失真数据增强')
add_arg('use_random_expand',   bool,   True,                     '是否使用随机扩张数据增强')
add_arg('use_random_crop',     bool,   True,                     '是否使用随机裁剪数据增强')
add_arg('use_random_flip',     bool,   True,                     '是否使用随机翻转数据增强')
add_arg('resume_model',        str,    None,                     '恢复训练的模型文件夹，当为None则不使用恢复模型')
add_arg('pretrained_model',    str,    None,                     '预训练模型的模型文件，当为None则不使用预训练模型')
args = parser.parse_args()
print_arguments(args)

# 获取训练器
trainer = PPYOLOETrainer(model_type=args.model_type,
                         batch_size=args.batch_size,
                         num_workers=args.num_workers,
                         num_classes=args.num_classes,
                         image_dir=args.image_dir,
                         train_anno_path=args.train_anno_path,
                         eval_anno_path=args.eval_anno_path,
                         use_gpu=args.use_gpu)

trainer.train(num_epoch=args.num_epoch,
              learning_rate=args.learning_rate,
              log_interval=args.log_interval,
              use_random_distort=args.use_random_distort,
              use_random_expand=args.use_random_expand,
              use_random_crop=args.use_random_crop,
              use_random_flip=args.use_random_flip,
              save_model_path=args.save_model_path,
              resume_model=args.resume_model,
              pretrained_model=args.pretrained_model)
