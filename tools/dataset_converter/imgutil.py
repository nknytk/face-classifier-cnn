# coding: utf-8

import os
import re
import sys
from urllib.request import urlopen
import numpy
import cv2


MIME_JPG_PTN = re.compile('^.*(jpeg|jpg|jpe).*$', re.IGNORECASE)
MIME_PNG_PTN = re.compile('^.*png.*$', re.IGNORECASE)
FACE_CLASSIFIER = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')


def load_img(file_path):
    try:
        if os.path.exists(file_path):
            return cv2.imread(file_path)

        elif file_path.startswith('http'):
            with urlopen(file_path) as fp:
                img_bin = numpy.fromstring(fp.read(), dtype=numpy.uint8)
                mime = fp.getheader('Content-Type', '')
                print(mime)
            if MIME_JPG_PTN.match(mime):
                return cv2.imdecode(img_bin, cv2.IMREAD_UNCHANGED)
            elif MIME_PNG_PTN.match(mime):
                return cv2.imdecode(img_bin, cv2.IMREAD_UNCHANDED)
            else:
                sys.stderr.write('Unacceptable mime type {}.\n'.format(mime))

        else:
            sys.stderr.write('{} is not found.\n'.format(file_path))

    except Exception as e:
        sys.stderr.write('Failed to load {} by {}\n'.format(file_path, e))

    return None

def crip_face(orig_img):
    img_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
    face_rects = FACE_CLASSIFIER.detectMultiScale(img_blurred, scaleFactor=1.1, minNeighbors=1, minSize=(10, 10))
    face_imgs = []
    for r in face_rects:
        face_imgs.append(orig_img[r[1]: r[1] + r[3], r[0]: r[0] + r[2]])
    return face_imgs


def read_label_table(file_name):
    label_table = {}

    with open(file_name) as fp:
        for line in fp:
            if line.startswith('#'):
                continue
            fields = line.strip('\n\r-').split('\t')
            if len(fields) != 2:
                continue

            label_table[fields[0]] = fields[1]

    return label_table
