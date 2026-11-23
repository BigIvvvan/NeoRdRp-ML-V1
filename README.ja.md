# NeoRdRp-ML V1

[English](README.md) | [モデルとデータ](https://huggingface.co/BigIvvvan/NeoRdRp-ML-V1)

NeoRdRp-ML V1 は、RNA 依存性 RNA ポリメラーゼ（RdRp）の検出、ウイルス科の
推定、および RdRp 領域の開始・終了位置の予測を行う研究用モデルです。この
リポジトリには、学習コード、推論コード、再現スクリプト、テスト、および 1 件の
蛋白質を対象とした同梱デモが含まれます。

モデルは、アミノ酸トークンに対する双方向 Mamba エンコーダと、ESM-C および
ProTrek のプール表現を組み合わせます。両蛋白質言語モデルは入力文脈と学習時の
蒸留教師の両方に使われます。V1 モデルのパラメータ数は 17,090,558 です。

## 出力

- RdRp 確率と校正済みの二値判定
- 最も確率の高いウイルス科、その確率、回答保留フラグ
- 1 始まり・両端を含む `Qstart` / `Qend` と校正済み区間

配列が 4,096 残基を超える場合は、中央の評価窓と座標を併記します。

## インストール

Python 3.11 以降が必要です。

```bash
python -m pip install neordrp-ml-v1
```

モデルリポジトリは非公開です。初回実行前に Hugging Face のアクセストークンを
設定するか、`--checkpoint` でローカルの V1 チェックポイントを指定してください。

```bash
export HF_TOKEN=your_token
```

## 1 蛋白質デモ

```bash
neordrp-ml-v1 demo
```

同梱デモでは、`E1:L0/Q2N379` が RdRp 陽性、`Caliciviridae`、`Qstart=6`、
`Qend=481` 付近と予測されます。結果を JSON に保存する場合は次を使います。

```bash
neordrp-ml-v1 demo --output demo_prediction.json
```

## 任意の FASTA を予測

```bash
python -m pip install "neordrp-ml-v1[embed]"
neordrp-ml-v1 embed --fasta proteins.faa --output proteins_features.npz --device cuda
neordrp-ml-v1 predict --fasta proteins.faa --features proteins_features.npz --output predictions.json
```

特徴抽出には `esmc_600m` と `westlake-repl/ProTrek_650M` が必要です。これらの
上流モデルの重みは V1 リリースには含まれません。

## 学習の再現

学習には CUDA GPU が必要です。次のコマンドは、前処理済み学習データを
Hugging Face から取得して SHA-256 を確認した後、V1 の 2 段階学習を実行します。

```bash
bash scripts/reproduce_v1.sh
```

第 1 段階では検出、65 科分類、境界予測、特徴蒸留を学習します。第 2 段階では
第 1 段階の重みから初期化し、境界予測を重点的に微調整します。

## 科学的な位置付け

V1 は研究用であり、臨床診断には使用できません。固定 RefSeq 評価では AUROC
0.7621、60,000 陰性中の偽陽性 60 件以内で 1,235 陽性中 181 件を検出しました。
一方、境界位置予測は未完成であり、41 件中、開始端が 10 残基以内だったのは
1 件、終了端は 3 件、両端は 0 件でした。V1 の位置予測を成功した機能として
表現してはいけません。

詳細は [学習](docs/training.md)、[データカード](docs/data_card.md)、
[モデルカード](docs/model_card.md)、
[再現手順](docs/reproducibility.md) を参照してください。
