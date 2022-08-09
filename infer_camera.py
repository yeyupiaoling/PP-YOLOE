import argparse
import functools
import cv2
from PIL import ImageFont

from utils.predictor import DetectionPredictor
from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('device_id',        int,    0,                              '摄像头ID或者视频路径')
add_arg('use_gpu',          bool,   True,                           '是否使用GPU进行预测')
add_arg('use_tensorrt',     bool,   False,                          '是否使用TensorRT加速')
add_arg('image_shape',      str,    '640,640',                      '导出模型图像输入大小')
add_arg('model_dir',        str,    'output_inference/PPYOLOE_M',   '预测模型文件夹路径')
add_arg('labels_list_path', str,    'dataset/label_list.txt',       '数据标签列表文件路径')
args = parser.parse_args()
print_arguments(args)


def main():
    image_shape = [int(s) for s in args.image_shape.split(',')]
    # 打开摄像头并设置摄像头
    cap = cv2.VideoCapture(args.device_id)
    # 字体的格式
    font_style = ImageFont.truetype("utils/simsun.ttc", 14, encoding="utf-8")
    predictor = DetectionPredictor(model_dir=args.model_dir,
                                   labels_list_path=args.labels_list_path,
                                   use_gpu=args.use_gpu,
                                   use_tensorrt=args.use_tensorrt,
                                   height=image_shape[0],
                                   width=image_shape[1])
    while True:
        # 从摄像头读取图片
        success, image = cap.read()
        if success:
            result = predictor.infer([image])[0]
            print('识别结果：', result)
            img = predictor.draw_box(image, results=result, font_style=font_style)
            # 显示图像
            cv2.imshow('result', img)
            cv2.waitKey(1)


if __name__ == "__main__":
    main()
