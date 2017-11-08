# 顔分類CNNツール v2

顔写真から個人を識別するCNNモデルの実装と、学習用データセット作成補助ツール。  

学習部分も含め、Raspberry Pi Type B (初代) で個人の識別ができることを目標として実装された。  

1. 事前に集めた大量の顔画像を使い、サーバでCNNの分類器を学習させる。  
2. 学習済み分類器をRaspberry Piに置き、画像特徴量抽出器とする
3. 実際に分類したい画像の特徴量を取り出し、多層パーセプトロンで分類する

ことで、Raspberry Pi上での処理の軽減と分類力の向上を目指す。

## Requirements

* 必須
    - OS: Ubuntu 16.04 または Raspbian Jessie
    - Python 3
* データセット作成補助ツールに必要
    - BeatifulSoup 4
    - OpenCV 3.1.0 with Python 3 binding
* 画像分類モデルの使用に必要
    - chainer 3.0.0
    - Pillow 4

## ドキュメント

* [依存ライブラリのインストール方法](docs/install.md)
* [学習用人物画像の収集](docs/collect-imgs.md)
* [画像から学習用データを作成する方法](docs/create-dataset.md)
* [学習](docs/train.md)
* [学習済み特徴量抽出器](docs/pre_trained_models.md)
* [識別性能の検証結果](docs/performance.md)

参考

* [識別性能の検証結果(v1)](docs/performance_v1.md)
* [v2のv1の違い](docs/comparison_v1_v2.md)

## License

MIT License (see `LICENSE` file).
