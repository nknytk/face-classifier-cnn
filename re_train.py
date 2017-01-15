# coding: utf-8

import json
import pickle
import os
import sys
import cv2
import chainer
from chainer.training import extensions

sys.path.append(os.path.dirname(__file__))
import cnn_models


def main(config_file):
    with open(config_file) as fp:
        conf = json.load(fp)

    model_class = getattr(cnn_models, conf['model'])
    model = cnn_models.FaceClassifier100x100_Relearn(
        n_classes=conf['re_train']['n_classes'],
        n_base_units=conf['n_base_units'],
        learned_model_file=conf['pre_train']['out_file'],
        orig_model_class=model_class,
        orig_n_classes=conf['pre_train']['n_classes']
    )
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_dataset = create_dataset(os.path.join(conf['re_train']['dataset_path'], 'train'))
    train_iter = chainer.iterators.SerialIterator(train_dataset, conf.get('batch_size', 10))
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=-1)
    trainer = chainer.training.Trainer(updater, (conf['re_train']['epoch'], 'epoch'), out='out')

    eval_model = model.copy()
    eval_model.tain = False

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar(update_interval=10))

    test_dataset_path = os.path.join(conf['re_train']['dataset_path'], 'test')
    if os.path.exists(test_dataset_path):
        test_dataset = create_dataset(test_dataset_path)
        test_iter = chainer.iterators.SerialIterator(test_dataset, 1, repeat=False, shuffle=False)
        trainer.extend(extensions.Evaluator(test_iter, eval_model, device=-1))
        trainer.extend(extensions.PrintReport([
            'epoch', 'main/loss', 'validation/main/loss',
            'main/accuracy', 'validation/main/accuracy'
        ]))
    else:
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy']))

    trainer.run()

    with open(conf['re_train']['out_file'], mode='wb') as fp:
        pickle.dump(model, fp)

def create_dataset(root_dir):
    data_pairs = []
    for label_dir in os.listdir(root_dir):
        label = int(label_dir)
        files = os.listdir(os.path.join(root_dir, label_dir))
        data_pairs += [(os.path.join(label_dir, f), label) for f in files]
    return chainer.datasets.image_dataset.LabeledImageDataset(pairs=data_pairs, root=root_dir)


if __name__ == '__main__':
    main(sys.argv[1])
