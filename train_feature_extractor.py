# coding: utf-8

import json
import os
import sys
from random import random, choice
import chainer
from chainer.training import extensions, triggers
from chainer.datasets.image_dataset import LabeledImageDataset
import numpy
from PIL import Image, ImageOps

sys.path.append(os.path.dirname(__file__))
import cnn_feature_extractors


class LabeledImageDatasetWithAugmentation(LabeledImageDataset):

    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards)
        self.aug_functions = [
            lambda pil_img: pil_img,
            lambda pil_img: ImageOps.mirror(pil_img),
            self._random_crop,
            lambda pil_img: self._random_crop(ImageOps.mirror(pil_img))
        ]
        self.max_crop_offset = 0.05

    def _read_image_as_array(self, path, dtype):
        aug_func = choice(self.aug_functions)
        f = Image.open(path)
        try:
            image = numpy.asarray(aug_func(f), dtype=dtype)
        finally:
            if hasattr(f, 'close'):
                f.close()
        return image

    def _random_crop(self, pil_img):
        orig_x, orig_y = pil_img.size
        tmp_x = int(orig_x * (1 + self.max_crop_offset))
        tmp_y = int(orig_y * (1 + self.max_crop_offset))
        tmp_img = pil_img.resize((tmp_x, tmp_y), Image.LANCZOS)
        offset_x = int((tmp_x - orig_x) * random())
        offset_y = int((tmp_y - orig_y) * random())
        cropped_img = tmp_img.crop((offset_x, offset_y, orig_x + offset_x, orig_y + offset_y))
        return cropped_img

    def get_example(self, i):
        path, int_label = self._pairs[i]
        full_path = os.path.join(self._root, path)
        image = self._read_image_as_array(full_path, self._dtype)

        if image.ndim == 2:
            # image is greyscale
            image = image[:, :, numpy.newaxis]
        label = numpy.array(int_label, dtype=self._label_dtype)
        return image.transpose(2, 0, 1), label


def main(config_file):
    with open(config_file) as fp:
        conf = json.load(fp)['feature_extractor']
    paths = conf['out_file'].rsplit('/', 1)
    if len(paths) == 2:
        out_dir, result_file = paths
    else:
        out_dir = '.'
        result_file = conf['out_file']

    model_class = getattr(cnn_feature_extractors, conf['model'])
    model = model_class(conf['n_classes'], conf['n_base_units'])

    resume_file = conf['out_file'] + '.to_resume'
    if os.path.exists(resume_file):
        chainer.serializers.load_npz(resume_file, model)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    device = conf.get('device', -1)
    train_dataset = create_dataset(os.path.join(conf['dataset_path'], 'train'), True)
    train_iter = chainer.iterators.SerialIterator(train_dataset, conf.get('batch_size', 10))
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = chainer.training.Trainer(updater, (conf['epoch'], 'epoch'), out=out_dir)

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar(update_interval=10))

    test_dataset_path = os.path.join(conf['dataset_path'], 'test')
    if os.path.exists(test_dataset_path):
        test_dataset = create_dataset(test_dataset_path)
        test_iter = chainer.iterators.SerialIterator(test_dataset, 20, repeat=False, shuffle=False)
        trainer.extend(extensions.Evaluator(test_iter, model, device=device))
        trainer.extend(extensions.PrintReport([
            'epoch', 'main/loss', 'validation/main/loss',
            'main/accuracy', 'validation/main/accuracy'
        ]))
        trainer.extend(
            extensions.snapshot_object(model, result_file),
            trigger=triggers.MaxValueTrigger(key='validation/main/accuracy', trigger=(1, 'epoch'))
        )
        trainer.run()
    else:
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy']))
        trainer.run()
        chainer.serializers.save_npz(conf['out_file'], model.to_cpu())



def create_dataset(root_dir, augumentation=False):
    data_pairs = []
    for label_dir in os.listdir(root_dir):
        label = int(label_dir)
        files = os.listdir(os.path.join(root_dir, label_dir))
        data_pairs += [(os.path.join(label_dir, f), label) for f in files]
    if augumentation:
        return LabeledImageDatasetWithAugmentation(pairs=data_pairs, root=root_dir)
    else:
        return LabeledImageDataset(pairs=data_pairs, root=root_dir)

if __name__ == '__main__':
    main(sys.argv[1])
