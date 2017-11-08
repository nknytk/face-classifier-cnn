# 特徴量抽出器の学習済みモデル

120人16576枚の画像を使い200epoch学習させた特徴量抽出器が、`pre_trained_models/`以下に掲載されている。  
これらを使用することで特徴量抽出器の学習を省略し、分類器の学習だけで済ませることができる。

### モデルの選択

ファイル名は以下の規約で命名されている。

```
${クラス識別子}_${n_base_unit}.npz
```

### クラス識別子

* V: [FaceClassifier100x100V](../cnn_feature_extractors.py#L85)
* V2: [FaceClassifier100x100V2](../cnn_feature_extractors.py#L18)

V2のほうが識別能力が高く、処理が重い。

### n_base_unit

フィルタの数を調整するための変数。小さいほど処理が軽く、大きいほど識別性能が高い傾向がある。
