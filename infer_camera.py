import cv2
from PIL import ImageFont

from utils.predictor import DetectionPredictor


def main():
    # 打开摄像头并设置摄像头
    cap = cv2.VideoCapture(0)
    # 字体的格式
    font_style = ImageFont.truetype("utils/simsun.ttc", 18, encoding="utf-8")
    predictor = DetectionPredictor(model_dir='output_inference/PPYOLOE_S', labels_list_path='dataset/label_list.txt')
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
