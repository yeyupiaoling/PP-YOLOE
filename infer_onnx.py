import cv2
import numpy as np
from PIL import ImageFont

from utils.onnx_predictor import ONNXPredictor


def main():
    image_path = 'dataset/test.jpg'
    # 字体的格式
    font_style = ImageFont.truetype("utils/simsun.ttc", 14, encoding="utf-8")
    predictor = ONNXPredictor(model_path='output_inference/model.onnx', labels_list_path='dataset/label_list.txt')
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    result = predictor.infer(image)
    print('识别结果：', result)
    img = predictor.draw_box(image, results=result, font_style=font_style)
    # 显示图像
    cv2.imshow('result', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
