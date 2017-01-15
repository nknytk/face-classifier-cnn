# 顔分類CNNツール

顔写真から個人を識別するCNNモデルの実装と、学習用データセット作成補助ツール。  

学習部分も含め、Raspberry Pi Type B (初代、以降RP) で個人の識別ができることを目標として実装された。  

1. 事前に集めた大量の顔画像を使い、サーバでCNNの分類器を学習させる。  
2. 学習済み分類器をRPに置き、実際に分類したい画像で結合部分だけを再学習させる

ことで、RP上での処理の軽減と分類力の向上を狙う。

## Requirements

* OS: Ubuntu 16.04 または Raspbian Jessie
* Python 3
* OpenCV 3.1.0 with Python 3 binding
* chainer 1.18
* Pillow 4
* BeatifulSoup 4

## ドキュメント

* [依存ライブラリのインストール方法](docs/install.md)
* [人物画像の集め方](docs/collect-imgs.md)
* [画像から学習用データを作成する方法](docs/create-dataset.md)
* [学習のさせ方](未記載)
* [事前学習済みモデル](未記載)
* [転移学習の成果について](未記載)
* [モデルに与えるパラメータと識別性能について](未記載)
