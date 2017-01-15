# 学習

大量データを用いて有効な特徴量の抽出方法を学習させるための事前学習、  
実際に分類したい画像の分類方法を学習させるための再学習の2段階の学習を行う。


## 設定ファイル

JSON形式の設定ファイルを作成する。

### 設定例

```JSON
{
    "model": "FaceClassifier100x100A",
    "n_base_units": 16,
    "pre_train": {
        "dataset_path": "dataset/100x100_120",
        "epoch": 200,
        "n_classes": 120,
        "out_file": "pre_trained_models/A_16.npz"
    },
    "re_train": {
        "dataset_path": "dataset/100x100_8",
        "epoch": 30,
        "n_classes": 8,
        "out_file": "dataset/100x100_8/face_A_16.pickle"
    }
}
```

* model: CNNのクラス名
* n_base_units: フィルタの数を調整するためのパラメータ
* pre_train
    - dataset_path: 事前学習用データセットのパス
    - epoch: 学習させるepoch数
    - n_classes: 事前学習用データのクラス数
    - out_file: 事前学習済みモデルの出力先ファイル名。`chainer.serializers.save_npz()`によりnpz形式で出力される。
* re_train
    - dataset_path: 再学習用データセット(=分類したい人物のデータセット)のパス
    - epoch: 学習させるepoch数
    - n_classes: 再学習用データのクラス数
    - out_file: 再学習済みモデルの出力先ファイル名。`pickle.dump()`によりバイナリで出力される。

## 事前学習

クラス数、1クラスあたりの画像数がともに多い教師データを与えて学習を行い、  
顔から個人を識別するために一般的に有効な特徴量の取り出し方を学習させる。  
教師データが実際に分類したい人物の顔の画像を含んでいる必要はない。  

```
python pre_train.py <設定ファイル>
```

本リポジトリに付属の[事前学習済みモデル](./pre_trained_models.md)を利用する場合、事前学習は省略できる。  


## 再学習

実際に分類したい人物の顔の画像を教師データとして与えて学習させる。  
畳み込み・プーリング層の重み(=特徴量抽出)は事前学習済みモデルのものを引き継いで固定し、結合層の重みだけを学習する。

```
python re_train.py <設定ファイル>
```
