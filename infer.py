import cv2
import numpy as np
from PIL import ImageFont

from utils.predictor import DetectionPredictor


def main():
    image_path = 'dataset/images2/xcf_105866722_1.jpg'
    # 字体的格式
    font_style = ImageFont.truetype("utils/simsun.ttc", 18, encoding="utf-8")
    predictor = DetectionPredictor(model_dir='output_inference/PPYOLOE_S', labels_list_path='dataset/label_list.txt')
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    result = predictor.infer([image])[0]
    print('识别结果：', result)
    img = predictor.draw_box(image, results=result, font_style=font_style)
    # 显示图像
    cv2.imshow('result', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
