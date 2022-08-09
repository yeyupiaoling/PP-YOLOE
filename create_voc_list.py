import os
import re

from tqdm import tqdm


def create_list(images_dir, annotations_dir):
    num = 0
    eval_list = []
    train_list = []
    labels_set = set()
    images = []
    for root, dirs, files in os.walk(images_dir):
        for name in files:
            img_path = os.path.join(root, name).replace('\\', '/')
            images.append(img_path)
    for image in tqdm(images):
        path = image[len(images_dir)+1:].replace('jpg', 'xml')
        ann_path = os.path.join(annotations_dir, path).replace('\\', '/')
        if os.path.exists(ann_path):
            num += 1
            # 安装1%划分评估集
            if num % 100 == 0:
                eval_list.append('%s %s\n' % (image[8:], ann_path[8:]))
            else:
                train_list.append('%s %s\n' % (image[8:], ann_path[8:]))
            # 获取训练集的标签
            with open(ann_path, 'r', encoding='utf-8') as f1:
                ret = re.findall(r'<name>\S+</name>', f1.read())
            for r in ret:
                labels_set.add(r[6:-7])

    with open('dataset/train.txt', 'w', encoding='utf-8') as f:
        for line in train_list:
            f.write('%s' % line)
    with open('dataset/eval.txt', 'w', encoding='utf-8') as f:
        for line in eval_list:
            f.write('%s' % line)
    with open('dataset/label_list.txt', 'w', encoding='utf-8') as f:
        for line in labels_set:
            f.write('%s\n' % line)


if __name__ == '__main__':
    create_list(images_dir='dataset/images', annotations_dir='dataset/annotations')
