import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


class ONNXPredictor:
    def __init__(self, model_path, labels_list_path, use_gpu=True, width=640, height=640, threshold=0.5, use_tensorrt=False):
        self.width = width
        self.height = height
        self.threshold = threshold
        # 预测器
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession(model_path, so)
        if use_tensorrt and use_gpu:
            self.net.set_providers(['TensorrtExecutionProvider'])
        elif use_gpu:
            self.net.set_providers(['CUDAExecutionProvider'])
        else:
            self.net.set_providers(['CPUExecutionProvider'])

        with open(labels_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.labels = [line.replace('\n', '') for line in lines]

    def load_image(self, img):
        img = cv2.resize(img, (self.width, self.height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mean = np.array([0.485, 0.456, 0.406]).astype('float32')
        std = np.array([0.229, 0.224, 0.225]).astype('float32')
        img = (img.astype('float32') * 1.0 / 255.0 - mean) / std
        img = np.transpose(img, axes=[2, 0, 1])
        img = img.astype(np.float32, copy=False)
        return img

    # 预测图片
    def infer(self, img):
        im_shape = np.array([img.shape[0], img.shape[1]]).astype(np.float32)
        scale_factor = np.array([self.width, self.height], dtype=np.float32) / im_shape
        image = self.load_image(img)
        scale_factor = np.expand_dims(scale_factor, axis=0).astype(np.float32)
        image = np.expand_dims(image, axis=0).astype(np.float32)

        outputs = self.net.run(None, {
            self.net.get_inputs()[0].name: image,
            self.net.get_inputs()[1].name: scale_factor})

        # 获取输出
        outs = np.array(outputs[0])
        expect_boxes = (outs[:, 1] > self.threshold) & (outs[:, 0] > -1)
        np_boxes = outs[expect_boxes, :]
        result = []
        for o in np_boxes:
            result.append([self.labels[int(o[0])], o[1], int(o[2]), int(o[3]), int(o[4]), int(o[5])])
        return result

    # 对图像进行画框
    @staticmethod
    def draw_box(img, results, font_style):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for result in results:
            bbox = result[2:]
            label = result[0]
            score = result[1]
            xmin, ymin, xmax, ymax = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            # 画人脸框
            draw.rectangle([xmin, ymin, xmax, ymax], outline=(0, 0, 255), width=2)
            # 绘制文本
            draw.text((xmin, ymin), '%s, %0.2f' % (label, score), (0, 255, 0), font=font_style)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

