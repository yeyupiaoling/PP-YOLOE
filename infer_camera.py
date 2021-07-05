import cv2
import numpy as np
import paddle.inference as paddle_infer
from PIL import Image, ImageFont, ImageDraw

# 创建 config
config = paddle_infer.Config('output_inference/ppyolo_mbv3_large_qat/model.pdmodel',
                             'output_inference/ppyolo_mbv3_large_qat/model.pdiparams')
config.enable_use_gpu(1000, 0)
config.enable_memory_optim()

# 根据 config 创建 predictor
predictor = paddle_infer.create_predictor(config)

# 获取输入层
image_handle = predictor.get_input_handle('image')
im_shape_handle = predictor.get_input_handle('im_shape')
scale_factor_handle = predictor.get_input_handle('scale_factor')
image_handle.reshape([1, 3, 320, 320])
im_shape_handle.reshape([1, 2])
scale_factor_handle.reshape([1, 2])

# 获取输出的名称
output_names = predictor.get_output_names()

# 打开摄像头并设置摄像头
cap = cv2.VideoCapture(0)
label_file = 'dataset/label_list.txt'

with open(label_file, 'r', encoding='utf-8') as f:
    names = f.readlines()


def load_image(img):
    img = cv2.resize(img, (320, 320))
    mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis, :]
    std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis, :]
    img = (img / 255.0 - mean) / std
    img = np.transpose(img, axes=[2, 0, 1])
    img = img.astype(np.float32, copy=False)
    img = img[np.newaxis, :]
    return img


def infer(im, threshold=0.5):
    im_shape = np.array([[im.shape[0], im.shape[1]]]).astype(np.float32)
    scale_factor = np.array([[1., 1.]], dtype=np.float32)
    image = load_image(im)

    # 设置输入
    image_handle.copy_from_cpu(image)
    im_shape_handle.copy_from_cpu(im_shape)
    scale_factor_handle.copy_from_cpu(scale_factor)
    # 运行predictor
    predictor.run()
    # 获取输出
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu()
    expect_boxes = (output_data[:, 1] > threshold) & (output_data[:, 0] > -1)
    output_data = output_data[expect_boxes, :]
    return output_data


# 对图像进行画框
def draw_box(img, results):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for i in range(len(results)):
        bbox = results[i, 2:]
        label = int(results[i, 0])
        score = results[i, 1]
        xmin, ymin, xmax, ymax = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        # 画人脸框
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(0, 0, 255), width=2)
        # 字体的格式
        font_style = ImageFont.truetype("simsun.ttc", 18, encoding="utf-8")
        # 绘制文本
        draw.text((xmin, ymin), '%s, %0.2f' % (names[label], score), (0, 255, 0), font=font_style)
    cv2.imshow('result', cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))


def main():
    while True:
        # 从摄像头读取图片
        success, image = cap.read()
        if success:
            result = infer(image)
            print(result)
            draw_box(image, result)
            cv2.waitKey(1)


if __name__ == '__main__':
    main()
