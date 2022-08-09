import argparse
import functools
import os

import paddle

from model.utils import get_infer_cfg_and_input_spec
from model.yolo import PPYOLOE
from utils.logger import setup_logger
from utils.utils import add_arguments, print_arguments, get_coco_model

logger = setup_logger(__name__)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('model_type',       str,    'M',                            '所使用的模型类型', choices=["X", "L", "M", "S"])
add_arg('num_classes',      int,    80,                             '分类的类别数量')
add_arg('image_shape',      str,    '3,640,640',                    '导出模型图像输入大小')
add_arg('save_model_dir',   str,    'output_inference/',            '导出模型保存的路径')
add_arg('resume_model',     str,    'output/PPYOLOE_M/best_model',  '恢复模型文件夹路径')
args = parser.parse_args()
print_arguments(args)


# 导出模型
def export_model():
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

    if args.resume_model is not None:
        assert os.path.exists(os.path.join(args.resume_model, 'model.pdparams')), "模型参数文件不存在！"
        model.set_state_dict(paddle.load(os.path.join(args.resume_model, 'model.pdparams')))
        logger.info('成功恢复模型参数和优化方法参数：{}'.format(args.resume_model))
    else:
        # 加载官方预训练模型
        pretrained_path = get_coco_model(model_type=args.model_type)
        model.set_state_dict(paddle.load(pretrained_path))
        logger.info('成功加载预训练模型：{}'.format(pretrained_path))
    model.eval()

    image_shape = [int(i) for i in args.image_shape.split(',')]
    static_model, pruned_input_spec = get_infer_cfg_and_input_spec(model=model, image_shape=image_shape)
    model_path = os.path.join(args.save_model_dir, f'PPYOLOE_{args.model_type.upper()}')
    paddle.jit.save(static_model, os.path.join(model_path, 'model'), input_spec=pruned_input_spec)
    logger.info(f'导出模型保存在：{model_path}')


if __name__ == '__main__':
    export_model()
