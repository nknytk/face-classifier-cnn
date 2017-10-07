# coding: utf-8

import os
import sys
import numpy
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers
from PIL import Image

"""
Convolution:
  out_size = (in_size + pad * 2 - ksize) / stride + 1
Inception:
  out_channels = out1 + out3 + out33 + proj_pool(default: in_channels)
  out_size = Convolution2d(ksize=3, pad=1, stride=stride) -> (in_size + 1 * 2 - 3) / stride + 1
"""


class FaceClassifier100x100I2(chainer.Chain):
    """ 100x100の顔画像を分類する、Inceptionの実装その2 """

    def __init__(self, n_classes, n_base_units=8):
        super().__init__(
            # conv1: 100 -> (100 - 4) / 3 + 1 = 33
            conv1 = L.Convolution2D(3, n_base_units, 4, stride=3),
            # pool1: 33 -> (33 + 1*2 - 3) / 2 + 1 = 17
            # inc1: 17 -> (17 + 1*2 - 3) / 2 + 1 = 9
            inc1 = L.InceptionBN(
                in_channels=n_base_units,
                out1=0,
                proj3=n_base_units,
                out3=n_base_units,
                proj33=n_base_units,
                out33=n_base_units,
                pooltype='max',
                stride=2
            ),
            # inc2: 9 -> (9 + 1*2 - 1) / 2 + 1 = 9
            inc2 = L.InceptionBN(
                in_channels=n_base_units * 3,
                out1=n_base_units,
                proj3=n_base_units,
                out3=n_base_units,
                proj33=n_base_units,
                out33=n_base_units,
                pooltype='avg',
                stride=1,
                proj_pool = n_base_units
            ),
            # inc3: 9 -> (9 + 1*2 - 3) / 2 + 1 = 5
            inc3 = L.InceptionBN(
                in_channels=n_base_units * 4,
                out1=0,
                proj3=n_base_units,
                out3=n_base_units,
                proj33=n_base_units,
                out33=n_base_units,
                pooltype='max',
                stride=2,
                proj_pool=n_base_units
            ),
            fc1=L.Linear(5 * 5 * n_base_units * 3, 128),
            fc2=L.Linear(128, n_classes)
        )
        self.n_units = 5 * 5 * n_base_units * 3

    def __call__(self, x, t):
        h = self.predict(x)
        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss

    def predict(self, x):
        h = self.reduct(x)
        h = F.dropout(F.relu(self.fc1(h)))
        h = self.fc2(h)
        return h

    def reduct(self, x):
        h = F.max_pooling_2d(
            F.relu(F.local_response_normalization(self.conv1(x))),
            ksize=3,
            stride=2,
            pad=1
        )
        h = F.relu(self.inc1(h))
        h = F.relu(self.inc2(h))
        h = F.relu(self.inc3(h))
        return h


class FaceClassifier100x100A(chainer.Chain):
    """ 100x100の顔画像を分類する、AlexNetを参考にした実装 """
    def __init__(self, n_classes, n_base_units=16):
        super().__init__(
            # conv1: 100 -> (100 - 4) / 2 + 1 = 49
            conv1 = L.Convolution2D(3, n_base_units, 4, stride=2),
            # pool1: 49 -> (49 - 3) / 2 + 1 = 24
            # conv2: 24 -> (24 + 1*2 - 3) / 1 + 1 = 24
            conv2 = L.Convolution2D(n_base_units, n_base_units * 2, 3, stride=1, pad=1),
            # pool2: 24 -> (24 - 4) / 2 + 1 = 11
            # conv3: 11 -> (11 + 1*2 - 3) / 2 + 1 = 6
            conv3 = L.Convolution2D(n_base_units * 2, n_base_units * 2, 3, stride=2, pad=1),
            fc1=L.Linear(6 * 6 * n_base_units * 2, 128),
            fc2=L.Linear(128, n_classes)
        )
        self.n_units = 6 * 6 * n_base_units * 2

    def __call__(self, x, t):
        h = self.predict(x)
        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss

    def predict(self, x):
        h = self.reduct(x)
        h = F.dropout(F.relu(self.fc1(h)))
        h = self.fc2(h)
        return h

    def reduct(self, x):
        h = F.max_pooling_2d(
            F.relu(F.local_response_normalization(self.conv1(x))),
            ksize=3,
            stride=2
        )
        h = F.max_pooling_2d(
            F.relu(F.local_response_normalization(self.conv2(h))),
            ksize=4,
            stride=2
        )
        h = F.relu(self.conv3(h))
        return h


class FaceClassifier100x100_Relearn(chainer.Chain):
    """ 学習済み顔分類モデルを読み込み、結合層だけを再学習する """

    def __init__(self, n_classes, n_base_units, learned_model_file, orig_model_class, orig_n_classes):
        self.learned_model = orig_model_class(orig_n_classes, n_base_units)
        chainer.serializers.load_npz(learned_model_file, self.learned_model)

        super().__init__(
            fc1=L.Linear(self.learned_model.n_units, 128),
            fc2=L.Linear(128, n_classes)
        )

    def __call__(self, x, t):
        h = self.predict(x)
        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss

    def predict(self, x):
        h = self.learned_model.reduct(x)
        h = F.dropout(F.relu(self.fc1(h)))
        h = self.fc2(h)
        return h

    def classify(self, img, resize=True, bgr2rgb=True):
        if resize:
            img = Image.fromarray(img, mode='RGB').resize((100, 100))
            img = numpy.asarray(img)
        t_img = img.transpose(2, 0, 1)  # width, height, channel -> channel, width, height
        if bgr2rgb:
            t_img = numpy.array([t_img[2], t_img[1], t_img[0]])  # BGR -> RGB
        x = chainer.Variable(numpy.asarray([t_img], dtype=numpy.float32), volatile="on")
        label = numpy.argmax(self.predict(x).data[0])
        return label
