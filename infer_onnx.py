import argparse
import functools
import cv2
import numpy as np
from PIL import ImageFont

from ppyoloe.onnx_predict import PPYOLOEONNXPredictor
from ppyoloe.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_gpu',          bool,   True,                           '是否使用GPU进行预测')
add_arg('use_tensorrt',     bool,   False,                          '是否使用TensorRT加速')
add_arg('image_path',       str,    'dataset/test.jpg',             '预测的图片路径')
add_arg('image_shape',      str,    '640,640',                      '导出模型图像输入大小')
add_arg('onnx_model',       str,    'models/PPYOLOE_M/model.onnx',  'ONNX模型文件夹路径')
add_arg('labels_list_path', str,    'dataset/label_list.txt',       '数据标签列表文件路径')
args = parser.parse_args()
print_arguments(args)


def main():
    image_shape = [int(s) for s in args.image_shape.split(',')]
    # 字体的格式
    font_style = ImageFont.truetype("utils/simsun.ttc", 14, encoding="utf-8")
    predictor = PPYOLOEONNXPredictor(model_path=args.onnx_model,
                                     labels_list_path=args.labels_list_path,
                                     use_gpu=args.use_gpu,
                                     use_tensorrt=args.use_tensorrt,
                                     height=image_shape[0],
                                     width=image_shape[1])
    image = cv2.imdecode(np.fromfile(args.image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    result = predictor.infer(image)
    print('识别结果：', result)
    img = predictor.draw_box(image, results=result, font_style=font_style)
    # 显示图像
    cv2.imshow('result', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
