# coding: utf-8

import chainer
from chainer import links as L
from chainer import functions as F
import numpy
from PIL import Image


class MLPClassifier(chainer.Chain):

    def __init__(self, n_classes, feature_extractor):
        self.fe = feature_extractor
        super().__init__(
            fc1=L.Linear(self.fe.n_units, 50),
            fc2=L.Linear(50, n_classes)
        )

    def __call__(self, x, t):
        h = self.predict(x)
        loss = F.softmax_cross_entropy(h, t)
        acc = F.accuracy(h, t)
        chainer.report({'loss': loss, 'accuracy': acc}, self)
        return loss

    def predict(self, x):
        h = F.relu(self.fc1(F.dropout(x, 0.4)))
        h = self.fc2(h)
        return h

    def image_to_feature(self, image_np):
        """
        画像をRGB, (3, 100, 100)のnumpy.arrayで受け取り、特徴ベクトルを抽出して返す。
        """
        _train = chainer.config.train
        chainer.config.train = False

        x = chainer.Variable(numpy.array([image_np], dtype=numpy.float32))
        feature_vector = F.flatten(self.fe.reduct(x)).data

        chainer.config.train = _train
        return feature_vector

    def classify(self, img, resize=True, bgr2rgb=True):
        """ OpenCVで読み込んだ画像を受け取り、ラベルを返す """
        if resize:
            img = Image.fromarray(img, mode='RGB').resize((100, 100))
            img = numpy.asarray(img)
        t_img = img.transpose(2, 0, 1)  # width, height, channel -> channel, width, height
        if bgr2rgb:
            t_img = numpy.array([t_img[2], t_img[1], t_img[0]])  # BGR -> RGB

        img_feature = self.image_to_feature(t_img)
        x = chainer.Variable(numpy.array([img_feature], dtype=numpy.float32))
        label = numpy.argmax(self.predict(x).data[0])
        return label
