# 学習

1. 大量データを用いて、顔の分類に一般的に有効な特徴量の抽出方法を学習させる
2. 実際に分類したい人物の画像を上記学習済みモデルで特徴ベクトルに変換し、特徴ベクトルの分類方法を学習させる

の2段階の学習を行う。


## 設定ファイル

JSON形式の設定ファイルを作成する。

### 設定例

```JSON
{
    "feature_extractor": {
        "model": "FaceClassifier100x100V2",
        "n_base_units": 16,
        "dataset_path": "dataset/100x100_120",
        "epoch": 10,
        "n_classes": 120,
        "out_file": "pre_trained_models/V2_16.npz",
        "device": 0
    },
    "classifier": {
        "dataset_path": "dataset/100x100_8",
        "epoch": 30,
        "n_classes": 8,
        "out_file": "cl.npz"
    }
}
```

* feature_extractor
    - model: CNNのクラス名
    - n_base_units: フィルタの数を調整するためのパラメータ
    - dataset_path: 特徴量抽出学習用データセットのパス
    - epoch: 学習させるepoch数
    - n_classes: 特徴量抽出学習用データのクラス数
    - out_file: 学習済み特徴量抽出器の出力先ファイル名。`chainer.serializers.save_npz()`によりnpz形式で出力される。
    - device: GPUデバイス番号。GPUを使用する場合のみ指定する。指定がなければCPUで学習する。
* classifier
    - dataset_path: 分類器学習用データセット(=分類したい人物のデータセット)のパス
    - epoch: 学習させるepoch数
    - n_classes: 分類対象データのクラス数
    - out_file: 学習済み分類器の出力先ファイル名。`chainer.serializers.save_npz()`によりnpz形式で出力される。

## 特徴量抽出器の学習

クラス数、1クラスあたりの画像数が共にに多い教師データを与えてCNN分類器を学習させる。  
学習させたCNNの全結合層の手前の層の出力を、顔から個人を識別するために一般的に有効な特徴量として扱う。  
教師データが実際に分類したい人物の顔の画像を含んでいる必要はなく、教師画像の数が多いことが重要である。  

```
python train_feature_extractor.py <設定ファイル>
```

本リポジトリに付属の[特徴量抽出学習済みモデル](./pre_trained_models.md)を利用する場合、特徴量抽出器の学習は省略できる。  


## 分類器の学習

実際に分類したい人物の顔の画像を教師データとして与えて学習させる。  
学習済み特徴量抽出器により画像をベクトルに変換し、ベクトルを分類する多層パーセプトロンを学習させる。

```
python train_classifier.py <設定ファイル>
```
