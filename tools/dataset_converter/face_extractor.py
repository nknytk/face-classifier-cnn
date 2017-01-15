# coding: utf-8

import os
import sys
import random
import traceback
import cv2

sys.path.append(os.path.dirname(__file__))
import imgutil


def raw_faces(parent_img_dir):
    name_table = {}
    for label in os.listdir(parent_img_dir):
        orig_img_dir = os.path.join(parent_img_dir, label)
        face_img_dir = os.path.join(parent_img_dir + '_raw_faces', label)
        if not os.path.exists(face_img_dir):
            os.makedirs(face_img_dir)

        for img_file in os.listdir(orig_img_dir):
            if img_file.endswith('.gif'):
                continue
            try:
                orig_img = imgutil.load_img(os.path.join(orig_img_dir, img_file))
                face_imgs = imgutil.crip_face(orig_img)
                for i, face in enumerate(face_imgs):
                    cv2.imwrite(os.path.join(face_img_dir, '{}_{}.jpg'.format(img_file.split('.')[0], i)), face)
                sys.stdout.write('Processed {}.\n'.format(os.path.join(orig_img_dir, img_file)))
            except:
                sys.stdout.write('Error occured for {}.\n'.format(os.path.join(orig_img_dir, img_file)))
                sys.stdout.write(traceback.format_exc())


if __name__ == '__main__':
    data_dir = sys.argv[1]
    raw_faces(data_dir)
