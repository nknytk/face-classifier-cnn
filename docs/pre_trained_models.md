# 事前学習済みモデル

120人16576枚の画像を使い200epoch学習させたモデルが、`pre_trained_models/`以下に掲載されている。  
これらのモデルを元に結合層の再学習だけを行うことで、事前学習を省略できる。

### モデルの選択

ファイル名は以下の規約で命名されている。

```
${クラス識別子}_${n_base_unit}.npz
```

### クラス識別子

* A: [FaceClassifier100x100A](../cnn_models.py#L177)
* I: [FaceClassifier100x100I](../cnn_models.py#L106)
* I2: [FaceClassifier100x100I2](../cnn_models.py#L21)

クラスごとの識別能力は I2 > A >>> I  
ただし、I2とAの識別能力は`n_base_unit`の設定次第で逆転する。

### n_base_unit

フィルタの数を調整するための変数。小さいほど処理が軽く、大きいほど識別性能が高い傾向がある。
