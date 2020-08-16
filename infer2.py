import os
import time
import cv2
import numpy as np
import paddle.fluid as fluid
from PIL import Image, ImageFont, ImageDraw

input_size = [608, 608]
# image mean
img_mean = [0.485, 0.456, 0.406]
# image std.
img_std = [0.229, 0.224, 0.225]
# data label/
label_file = '../dataset/gesture/label_list.txt'
# threshold value
score_threshold = 0.5
# infer model path
infer_model_path = 'output/yolov3_mobilenet_v3/'
# Whether use GPU to train.
use_gpu = True

# 打开摄像头并设置摄像头
imageWidth = 1280
imageHeight = 720

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, imageWidth * 2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imageHeight)


def get_image():
    while True:
        # 从摄像头读取图片
        sucess, image = cap.read()
        if sucess:
            image = image[:, 0:imageWidth, :]
            img = np.array((image,)).astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (input_size[0], input_size[1]))
            img = (img - img_mean) * img_std
            img = img.transpose()
            img = np.expand_dims(img, axis=0).astype(np.float32)
            return img, image


if not os.path.exists(infer_model_path):
    raise ValueError("The model path [%s] does not exist." % infer_model_path)

place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

[infer_program,
 feeded_var_names,
 target_var] = fluid.io.load_inference_model(dirname=infer_model_path,
                                             executor=exe,
                                             model_filename='__model__',
                                             params_filename='__params__')

print(target_var)

with open(label_file, 'r', encoding='utf-8') as f:
    names = f.readlines()


# 预测图像
def infer():
    img, image = get_image()
    im_size = np.array([[image.shape[0], image.shape[1]]])
    
    nmsed_out_v = exe.run(infer_program,
                          feed={feeded_var_names[0]: img, feeded_var_names[1]: im_size},
                          fetch_list=target_var,
                          return_numpy=False)
    nmsed_out_v = np.array(nmsed_out_v[0])
    print(nmsed_out_v)
    results = []
    try:
        for dt in nmsed_out_v:
            if dt[1] < score_threshold:
                continue
            results.append(dt)
        draw_image(image, results)
    except:
        pass


# 限制数据的结果，保证在合理范围内
def clip_bbox(bbox):
    xmin = max(min(bbox[0], 1.), 0.)
    ymin = max(min(bbox[1], 1.), 0.)
    xmax = max(min(bbox[2], 1.), 0.)
    ymax = max(min(bbox[3], 1.), 0.)
    return xmin, ymin, xmax, ymax


# 对图像进行画框
def draw_image(img, results):
    w, h, c = img.shape
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for result in results:
        xmin, ymin, xmax, ymax = clip_bbox(result[2:])
        xmin, ymin, xmax, ymax = int(xmin * h), int(ymin * w), int(xmax * h), int(ymax * w)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(0, 0, 255), width=4)
        # 字体的格式
        font_style = ImageFont.truetype("font/simfang.ttf", 18, encoding="utf-8")
        # 绘制文本
        draw.text((xmin, ymin), '%s, %0.2f' % (names[int(result[0])], result[1]), (0, 255, 0),
                  font=font_style)
    # 显示图像
    cv2.imshow('result image', cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


if __name__ == '__main__':
    while True:
        infer()
