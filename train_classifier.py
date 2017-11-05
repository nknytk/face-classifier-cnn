# coding: utf-8

import json
import os
import sys
import chainer
from chainer.training import extensions
import numpy
from PIL import Image

sys.path.append(os.path.dirname(__file__))
import cnn_feature_extractors
import classifiers


def main(config_file):
    with open(config_file) as fp:
        conf = json.load(fp)
    fe_conf = conf['feature_extractor']
    cl_conf = conf['classifier']

    fe_class = getattr(cnn_feature_extractors, fe_conf['model'])
    feature_extractor = fe_class(n_classes=fe_conf['n_classes'], n_base_units=fe_conf['n_base_units'])
    chainer.serializers.load_npz(fe_conf['out_file'], feature_extractor)

    model = classifiers.MLPClassifier(cl_conf['n_classes'], feature_extractor)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    device = cl_conf.get('device', -1)
    train_dataset = feature_dataset(os.path.join(cl_conf['dataset_path'], 'train'), model)
    train_iter = chainer.iterators.SerialIterator(train_dataset, conf.get('batch_size', 1))
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = chainer.training.Trainer(updater, (cl_conf['epoch'], 'epoch'), out='out_re')

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar(update_interval=10))

    test_dataset_path = os.path.join(cl_conf['dataset_path'], 'test')
    if os.path.exists(test_dataset_path):
        test_dataset = feature_dataset(test_dataset_path, model)
        test_iter = chainer.iterators.SerialIterator(test_dataset, 10, repeat=False, shuffle=False)
        trainer.extend(extensions.Evaluator(test_iter, model, device=device))
        trainer.extend(extensions.PrintReport([
            'epoch', 'main/loss', 'validation/main/loss',
            'main/accuracy', 'validation/main/accuracy'
        ]))
    else:
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy']))

    trainer.run()

    chainer.serializers.save_npz(cl_conf['out_file'], model)


def feature_dataset(root_dir, model):
    features = []
    labels = []
    for label_dir in os.listdir(root_dir):
        label = numpy.int32(label_dir)
        for fname in os.listdir(os.path.join(root_dir, label_dir)):
            labels.append(label)
            img = Image.open(os.path.join(root_dir, label_dir, fname))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_np = numpy.asarray(img, dtype=numpy.float32).transpose(2, 0, 1)
            feature = model.image_to_feature(img_np)
            features.append(feature)
    return chainer.datasets.TupleDataset(features, labels)


if __name__ == '__main__':
    main(sys.argv[1])
