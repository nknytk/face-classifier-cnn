# coding: utf-8

import json
import random
import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(__file__))
import imgutil


def main(config_file):
    with open(config_file) as fp:
        config = json.load(fp)
    img2dataset(
        config['orig_data_dir'],
        config.get('img_size', 100),
        config.get('test_ratio', 0.2),
        config.get('num_class')
    )


def img2dataset(orig_data_dir, img_size, test_ratio, num_class):
    orig_datasets = [d for d in os.listdir(orig_data_dir) if os.path.isdir(os.path.join(orig_data_dir, d))]
    if num_class is not None:
        random.shuffle(orig_datasets)
        target_labels = [orig_labels.pop() for i in range(num_class)]
    else:
        target_labels = sorted(orig_datasets)
        num_class = len(target_labels)

    dataset_label_table = {}
    dataset_dir = os.path.join('dataset', '{0}x{0}_{1}'.format(img_size, num_class))

    for i, orig_label in enumerate(target_labels):
        dataset_label_table[str(i)] = orig_label

        orig_label_dir = os.path.join(orig_data_dir, orig_label)
        for f in os.listdir(orig_label_dir):
            img = cv2.imread(os.path.join(orig_label_dir, f))
            resized_img = cv2.resize(img, (img_size, img_size))
            if random.random() > test_ratio:
                dir_to_save = os.path.join(dataset_dir, 'train', str(i))
            else:
                dir_to_save = os.path.join(dataset_dir, 'test', str(i))
            save_img(dir_to_save, f, resized_img)

    with open(os.path.join(dataset_dir, 'label_table.tsv'), mode='w') as fp:
        for label in sorted(dataset_label_table.keys()):
            fp.write('{}\t{}\n'.format(label, dataset_label_table[label]))


def save_img(dir_to_save, file_name, img):
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)
    cv2.imwrite(os.path.join(dir_to_save, file_name), img)


if __name__ == '__main__':
    main(sys.argv[1])
