import os
import xml.etree.ElementTree
from tqdm import tqdm


# 生成图像列表
def create(images_dir, annotations_dir, train_list_path, test_list_path, label_file):
    f_train = open(train_list_path, 'w', encoding='utf-8')
    f_test = open(test_list_path, 'w', encoding='utf-8')
    f_label = open(label_file, 'w', encoding='utf-8')
    label = set()
    images = os.listdir(images_dir)
    i = 0
    for image in tqdm(images):
        i += 1
        annotation_path = os.path.join(annotations_dir, image[:-3] + 'xml').replace('\\', '/')
        image_path = os.path.join(images_dir, image).replace('\\', '/')
        if not os.path.exists(annotation_path):
            continue
        root = xml.etree.ElementTree.parse(annotation_path).getroot()
        for object in root.findall('object'):
            label.add(object.find('name').text)
        if i % 20 == 0:
            f_test.write("%s %s\n" % (image_path[image_path.find('/') + 1:], annotation_path[annotation_path.find('/') + 1:]))
        else:
            f_train.write("%s %s\n" % (image_path[image_path.find('/') + 1:], annotation_path[annotation_path.find('/') + 1:]))
    for l in label:
        f_label.write("%s\n" % l)
    f_train.close()
    f_test.close()
    f_label.close()
    print('create data list done!')


if __name__ == '__main__':
    create('dataset/images', 'dataset/annotation', 'dataset/trainval.txt', 'dataset/test.txt', 'dataset/label_list.txt')
