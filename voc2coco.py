import argparse
import json
import os
import xml.etree.ElementTree as ET

from tqdm import tqdm


def voc_get_label_anno(dataset_path, voc_anno_list, labels_path, use_mainbody=True):
    if use_mainbody:
        labels_str = ['foreground']
        with open(labels_path, 'w', encoding='utf-8') as f:
            f.write('foreground')
    else:
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels_str = f.read().split()
    labels_ids = list(range(1, len(labels_str) + 1))

    with open(voc_anno_list, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    ann_img_paths = []
    for line in lines:
        i_path, a_path = line.strip('\n').split(' ')
        ann_img_paths.append([os.path.join(dataset_path, a_path), i_path])
    return dict(zip(labels_str, labels_ids)), ann_img_paths


def voc_get_image_info(annotation_root, im_id, filename):
    size = annotation_root.find('size')
    width = float(size.findtext('width'))
    height = float(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': im_id
    }
    return image_info


def voc_get_coco_annotation(obj, label2id, use_mainbody=True):
    label = obj.findtext('name')
    if not use_mainbody:
        assert label in label2id, f"{label} is not in label2id."
        category_id = label2id[label]
    else:
        category_id = 1
    bndbox = obj.find('bndbox')
    xmin = float(bndbox.findtext('xmin'))
    ymin = float(bndbox.findtext('ymin'))
    xmax = float(bndbox.findtext('xmax'))
    ymax = float(bndbox.findtext('ymax'))
    assert xmax > xmin and ymax > ymin, f"Box size error, {xmin}, {ymin}, {xmax}, {ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    anno = {'area': o_width * o_height,
            'iscrowd': 0,
            'bbox': [xmin, ymin, o_width, o_height],
            'category_id': category_id,
            'ignore': 0,
            }
    return anno


def voc_xmls_to_cocojson(annotation_paths, label2id, voc_anno_list, use_mainbody=True):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # bounding box start id
    im_id = 0
    print('Start converting !')
    for a_path, i_path in tqdm(annotation_paths):
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()
        try:
            for obj in ann_root.findall('object'):
                ann = voc_get_coco_annotation(obj=obj, label2id=label2id, use_mainbody=use_mainbody)
                if ann == '':continue
                ann.update({'image_id': im_id, 'id': bnd_id})
                output_json_dict['annotations'].append(ann)
                bnd_id = bnd_id + 1
        except Exception as e:
            print(a_path, e)
            continue

        img_info = voc_get_image_info(ann_root, im_id, i_path)
        output_json_dict['images'].append(img_info)

        im_id += 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)
    with open(voc_anno_list.replace('txt', 'json'), 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_path',
                        type=str,
                        default='dataset/',
                        help='VOC的数据保存路径')
    parser.add_argument('--voc_label_list',
                        type=str,
                        default='dataset/label_list.txt',
                        help='VOC数据的类别标签')
    parser.add_argument('--use_mainbody',
                        type=bool,
                        default=False,
                        help='是否导出主体检测数据，如果是则全部类别变成foreground')
    args = parser.parse_args()

    # 转换训练数据
    args.voc_anno_list = 'dataset/train.txt'
    label2id, ann_paths = voc_get_label_anno(args.dataset_path, args.voc_anno_list, args.voc_label_list,
                                             use_mainbody=args.use_mainbody)
    voc_xmls_to_cocojson(annotation_paths=ann_paths,
                         label2id=label2id,
                         voc_anno_list=args.voc_anno_list,
                         use_mainbody=args.use_mainbody)

    # 转换评估数据
    args.voc_anno_list = 'dataset/eval.txt'
    label2id, ann_paths = voc_get_label_anno(args.dataset_path, args.voc_anno_list, args.voc_label_list,
                                             use_mainbody=args.use_mainbody)
    voc_xmls_to_cocojson(annotation_paths=ann_paths,
                         label2id=label2id,
                         voc_anno_list=args.voc_anno_list,
                         use_mainbody=args.use_mainbody)


if __name__ == '__main__':
    main()
