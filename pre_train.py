# coding: utf-8

import json
import os
import sys
import chainer
from chainer.training import extensions

sys.path.append(os.path.dirname(__file__))
import cnn_models


def main(config_file):
    with open(config_file) as fp:
        conf = json.load(fp)

    model_class = getattr(cnn_models, conf['model'])
    model = model_class(conf['pre_train']['n_classes'], conf['n_base_units'])

    resume_file = conf['pre_train']['out_file'] + '.to_resume'
    if os.path.exists(resume_file):
        chainer.serializers.load_npz(resume_file, model)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    device = conf['pre_train'].get('device', -1)
    train_dataset = create_dataset(os.path.join(conf['pre_train']['dataset_path'], 'train'))
    train_iter = chainer.iterators.SerialIterator(train_dataset, conf.get('batch_size', 10))
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = chainer.training.Trainer(updater, (conf['pre_train']['epoch'], 'epoch'), out='out')

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar(update_interval=10))

    test_dataset_path = os.path.join(conf['pre_train']['dataset_path'], 'test')
    if os.path.exists(test_dataset_path):
        test_dataset = create_dataset(test_dataset_path)
        test_iter = chainer.iterators.SerialIterator(test_dataset, 20, repeat=False, shuffle=False)
        trainer.extend(extensions.Evaluator(test_iter, model, device=device))
        trainer.extend(extensions.PrintReport([
            'epoch', 'main/loss', 'validation/main/loss',
            'main/accuracy', 'validation/main/accuracy'
        ]))
    else:
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy']))

    trainer.run()

    chainer.serializers.save_npz(conf['pre_train']['out_file'], model)

def create_dataset(root_dir):
    data_pairs = []
    for label_dir in os.listdir(root_dir):
        label = int(label_dir)
        files = os.listdir(os.path.join(root_dir, label_dir))
        data_pairs += [(os.path.join(label_dir, f), label) for f in files]
    return chainer.datasets.image_dataset.LabeledImageDataset(pairs=data_pairs, root=root_dir)


if __name__ == '__main__':
    main(sys.argv[1])
