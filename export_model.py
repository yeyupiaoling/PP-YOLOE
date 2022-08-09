import argparse
import functools
import os

import paddle
from paddle.static import InputSpec

from model.yolo import PPYOLOE
from utils.logger import setup_logger
from utils.utils import add_arguments, print_arguments

logger = setup_logger(__name__)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('model_type',       str,    'S',                            '所使用的模型类型', choices=["X", "L", "M", "S"])
add_arg('num_classes',      int,    1,                              '分类的类别数量')
add_arg('image_shape',      str,    '3,640,640',                    '导出模型图像输入大小')
add_arg('save_model_dir',   str,    'output_inference/',            '导出模型保存的路径')
add_arg('resume_model',     str,    'output/PPYOLOE_S/best_model',  '恢复训练的模型文件')
args = parser.parse_args()
print_arguments(args)


def _prune_input_spec(input_spec, program, targets):
    # try to prune static program to figure out pruned input spec
    # so we perform following operations in static mode
    device = paddle.get_device()
    paddle.enable_static()
    paddle.set_device(device)
    pruned_input_spec = [{}]
    program = program.clone()
    program = program._prune(targets=targets)
    global_block = program.global_block()
    for name, spec in input_spec[0].items():
        try:
            v = global_block.var(name)
            pruned_input_spec[0][name] = spec
        except Exception:
            pass
    paddle.disable_static(place=device)
    return pruned_input_spec


def _get_infer_cfg_and_input_spec(model):
    image_shape = [int(i) for i in args.image_shape.split(',')]
    image_shape = [None] + image_shape
    im_shape = [None, 2]
    scale_factor = [None, 2]

    for layer in model.sublayers():
        if hasattr(layer, 'convert_to_deploy'):
            layer.convert_to_deploy()

    model.fuse_norm = False
    input_spec = [{"image": InputSpec(shape=image_shape, name='image'),
                   "im_shape": InputSpec(shape=im_shape, name='im_shape'),
                   "scale_factor": InputSpec(shape=scale_factor, name='scale_factor')}]
    static_model = paddle.jit.to_static(model, input_spec=input_spec)
    # NOTE: dy2st do not pruned program, but jit.save will prune program
    # input spec, prune input spec here and save with pruned input spec
    pruned_input_spec = _prune_input_spec(input_spec, static_model.forward.main_program,
                                          static_model.forward.outputs)

    return static_model, pruned_input_spec


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

    assert os.path.exists(os.path.join(args.resume_model, 'model.pdparams')), "模型参数文件不存在！"
    model.set_state_dict(paddle.load(os.path.join(args.resume_model, 'model.pdparams')))
    logger.info('成功恢复模型参数和优化方法参数：{}'.format(args.resume_model))
    model.eval()

    static_model, pruned_input_spec = _get_infer_cfg_and_input_spec(model=model)
    model_path = os.path.join(args.save_model_dir, f'PPYOLOE_{args.model_type.upper()}')
    paddle.jit.save(static_model, os.path.join(model_path, 'model'), input_spec=pruned_input_spec)


if __name__ == '__main__':
    export_model()
