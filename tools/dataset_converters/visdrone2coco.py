# Copyright (c) OpenMMLab. All rights reserved.
"""Convert VisDrone-DET txt annotations to COCO format.
The output COCO json is IDENTICAL in structure to labelme2coco.py.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from mmengine import track_iter_progress
from mmyolo.utils.misc import IMG_EXTENSIONS


VISDRONE_ID_TO_NAME = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str, help='Dataset image directory')
    parser.add_argument('--labels-dir', type=str, help='VisDrone txt label directory')
    parser.add_argument('--out', type=str, help='COCO label json output path')
    parser.add_argument('--class-id-txt', default=None, type=str, help='All class id txt path')
    return parser.parse_args()


def format_coco_annotations(points, image_id, ann_id, category_id):
    ann = dict()
    ann['iscrowd'] = 0
    ann['category_id'] = category_id
    ann['id'] = ann_id
    ann['image_id'] = image_id

    x1, y1 = points[0]
    x2, y2 = points[1]
    ann['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    ann['area'] = ann['bbox'][2] * ann['bbox'][3]

    seg = np.asarray(points).copy()
    seg[1, :] = np.asarray(points)[2, :]
    seg[2, :] = np.asarray(points)[1, :]
    ann['segmentation'] = [list(seg.flatten())]
    return ann


def parse_visdrone_to_coco(img_dir, labels_dir, all_classes_id=None):
    coco = {'images': [], 'categories': [], 'annotations': []}
    image_id, ann_id = 0, 0

    if all_classes_id is None:
        category_to_id = {}
        categories_labels = []
    else:
        category_to_id = all_classes_id
        categories_labels = list(all_classes_id.keys())
        for k, v in category_to_id.items():
            coco['categories'].append({'id': v, 'name': k})

    img_files = [
        p for p in Path(img_dir).iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    ]

    for img_path in track_iter_progress(sorted(img_files)):
        label_path = Path(labels_dir) / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        from PIL import Image
        w, h = Image.open(img_path).size

        image_id += 1
        coco['images'].append({
            'id': image_id,
            'file_name': img_path.name,
            'width': w,
            'height': h
        })

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue

                x, y, bw, bh = map(float, parts[:4])
                score = int(parts[4])
                cid = int(parts[5])

                if score == 0 or cid == 0:
                    continue

                class_name = VISDRONE_ID_TO_NAME[cid]

                if all_classes_id is None and class_name not in categories_labels:
                    categories_labels.append(class_name)
                    cid_new = len(categories_labels)
                    category_to_id[class_name] = cid_new
                    coco['categories'].append({'id': cid_new, 'name': class_name})
                elif all_classes_id is not None and class_name not in category_to_id:
                    raise ValueError(f'Unexpected class {class_name}')

                ann_id += 1
                x1, y1 = x, y
                x2, y2 = x + bw, y + bh
                points = [[x1, y1], [x2, y2], [x1, y2], [x2, y1]]

                ann = format_coco_annotations(
                    points, image_id, ann_id, category_to_id[class_name])
                coco['annotations'].append(ann)

    return coco, category_to_id


def convert_visdrone_to_coco(img_dir, labels_dir, out_path, class_id_txt=None):
    assert Path(out_path).suffix == '.json'

    if class_id_txt:
        all_classes_id = {}
        for line in Path(class_id_txt).read_text().splitlines():
            v, k = line.split(' ')
            all_classes_id[k] = int(v)
    else:
        all_classes_id = None

    coco, category_to_id = parse_visdrone_to_coco(
        img_dir, labels_dir, all_classes_id)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(coco, open(out_path, 'w'), indent=2)

    if class_id_txt is None:
        txt_path = Path(out_path).with_name('class_with_id.txt')
        with open(txt_path, 'w') as f:
            for k, v in category_to_id.items():
                f.write(f'{v} {k}\n')


def main():
    args = parse_args()
    convert_visdrone_to_coco(args.img_dir, args.labels_dir, args.out, args.class_id_txt)
    print('All done!')


if __name__ == '__main__':
    main()
