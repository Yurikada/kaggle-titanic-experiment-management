# Kaggle Titanic: 実験管理ケーススタディ / Experiment Management Case Study

[日本語](#日本語) | [English](#english)

## 日本語

### 概要

このリポジトリは、KaggleのTitanicコンペを題材に、機械学習の実験をどのように設計・比較・検証したかをまとめた学習成果物です。

Leaderboard上位を目指した高性能モデルではありません。採用担当者・技術面接担当者に、次の取り組み方を確認していただくことを目的としています。

- 仮説を立ててから特徴量を変更する
- validation splitの偶然性を複数分割で確認する
- モデル比較時の固定条件と変更条件を明示する
- 全体精度だけでなく、属性別の誤判定を分析する
- スコアが悪化した実験も残し、原因を分解する
- 同じ条件でsubmissionを再生成できるようにする

コンペ: <https://www.kaggle.com/competitions/titanic>

### このケースで示したこと

単純な決定木を出発点として、男性生存者の見逃しに着目し、複数のvalidation split、`FamilySize`特徴量、木の深さの比較を行いました。

特に重要だった学びは、妥当に見える仮説でも、複数条件を同時に変更したり、特徴量を集約して情報を失ったりすると、性能が下がりうることです。そのため、良い実験とはスコアが上がった実験だけではなく、仮説、操作、結果、解釈を追跡できる実験だと考えています。

### 主な結果

#### ベースライン

```text
model: DecisionTreeClassifier(max_depth=3, random_state=42)
features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, CabinKnown
validation: 80:20 stratified split
validation accuracy: 0.7933
Kaggle public score: 0.77990
```

#### 誤判定分析

最初のvalidation splitでは、生存した男性24人を全員「非生存」と誤判定していました。ただし、分割を変えると結果は次のように変化しました。

```text
random_state  male_survivors  missed  captured  miss_rate
42            24              24      0         1.0000
7             23              18      5         0.7826
2026          25              20      5         0.8000
```

ここから、次の2つを分けて解釈しました。

- 「男性生存者を全員見逃した」は、特定splitに依存する
- 「男性生存者を見逃しやすい」は、複数splitでも残る傾向である

#### FamilySize仮説

「家族のいる男性と、単独の男性では行動が異なるのではないか」という仮説から、`FamilySize = SibSp + Parch + 1`を導入しました。

その後、`SibSp`と`Parch`を`FamilySize`へ置き換え、同時に`max_depth=5`へ変更した提出では、Public Scoreが`0.74162`まで低下しました。

この結果は性能改善ではありません。一方で、次の問題を確認できた失敗実験として記録しています。

- 特徴量の集約によって、個別の情報を失う可能性がある
- 木を深くしても汎化性能が上がるとは限らない
- 特徴量とモデル深度を同時に変えると、悪化原因を一つに絞れない

### 実験上の学び

#### 比較条件を固定する

`max_depth`を比較するときは、特徴量、validation split、前処理、評価指標を固定します。特徴量を比較するときは、モデル深度とvalidation方法を固定します。

#### 単一splitを過信しない

データ数が少ない場合、1回のholdout評価は分割の影響を強く受けます。複数splitは、スコアを水増しするためではなく、観察した現象が特定分割だけのものかを確認するために使いました。

#### 失敗実験も残す

仮説どおりに改善しなかった場合も削除せず、何を変え、何が悪化し、どの条件が交絡したかを記録しました。

### データについて

Kaggleのコンペデータは、このリポジトリには含めていません。Kaggleから取得し、次のように配置してください。

```text
data/
  train.csv
  test.csv
  gender_submission.csv
```

Kaggle CLIを使う場合:

```powershell
kaggle competitions download -c titanic -p data
```

ダウンロード後、ZIPファイルを`data/`へ展開してください。`data/`はGit管理対象外です。

### ディレクトリ構成

```text
.
├── 01_eda.py
├── 02_baseline.py
├── 03_depth4_error_analysis.py
├── 04_depth4_age_split_error_analysis.py
├── 05_multi_split_validation.py
├── 06_family_size_multi_split_validation.py
├── 07_depth_5_6_7_multi_split_validation.py
├── 08_depth5_family_size_replaced_submission.py
├── error_analysis/
├── submissions/
├── experiment_summary.md
├── requirements.txt
└── README.md
```

### 再現方法

依存ライブラリをインストールします。

```powershell
pip install -r requirements.txt
```

Kaggleデータ未取得の環境でコードの疎通だけ確認する場合は、合成の小さなスモークテスト用データを生成できます。

```powershell
python scripts/generate_smoke_data.py
python 02_baseline.py
```

この合成データはKaggleスコアや分析結果の再現には使いません。スクリプトが期待する列、前処理、submission生成が動くことだけを確認するためのものです。

ベースラインを実行します。

```powershell
python 02_baseline.py
```

`FamilySize`置換版を実行します。

```powershell
python 08_depth5_family_size_replaced_submission.py
```

submissionは`submissions/`へ出力されます。

### 原理から学ぶNotebook

`titanic_first_principles_learning_journal.ipynb` は、精度よりも次を優先する学習記録です。

- 可視化を見る前の予想
- 特徴量を「現実の不完全な測定」として読む姿勢
- 分母、欠損、95% Wilson区間、分布の確認
- 多数派予測・単純ルール・Logistic Regressionの比較
- 混同行列と誤分類行から作る次の問い
- AIが代筆しない、実行者本人の反省・違和感・次の仮説の記入欄

生成と全セル検証:

```powershell
python build_learning_notebook.py
python validate_learning_notebook.py
```

公式データで検証した提出物は`submissions/submission_first_principles.csv`へ出力されます。
公式サイズでない入力では、誤提出を避けるためファイル名に`smoke`が付きます。

### 不確実性・欠損・誤りの構造を見るNotebook

`uncertainty_publish/titanic_uncertainty_and_error_structure.ipynb` は、
「棒グラフの高さが違うとき、どこからを差と呼べるか」を出発点にした日英併記の公開用Notebookです。
可視化する対象を次の3つに絞っています。

- 比率の不確実性: 群ごとのWilson信頼区間と、2群の差のブートストラップ分布
- 欠損の構造: 欠損の共起、`Age`欠損行を落としたときに残る乗客の構成変化、欠損自体と生存率の関係
- 誤りの偏り: 層別のOOF誤分類率、誤りの向き、決定木の予測確率の較正

モデルと特徴量は`02_baseline.py`のdepth 3ベースラインをそのまま引き継ぎ、
評価だけを単一splitから層化5-foldのOut-of-Fold予測へ変更しています。
新しいモデル選択は行っていません。

生成と全セル検証:

```powershell
python build_uncertainty_notebook.py
python validate_uncertainty_notebook.py
```

ローカル実行の提出物は`uncertainty_publish/outputs/`へ出力され、`submissions/`とは分離しています。
公開手順とチェックリストは`NOTEBOOK_PUBLISHING.md`にあります。

### 判定の基準を先に決めるNotebook

`preregistration_publish/titanic_preregistered_comparisons.ipynb` は、
**採否の基準を結果を見る前に登録してから**4つの比較を実行した記録です。

| 比較 | 変えた条件 | 結果 |
|---|---|---|
| `Cabin` の表現 | 落とす / 有無フラグ / デッキ | 現行維持 |
| 家族による情報の漏れ | 通常分割 / グループ分割 | 上振れは検出、比較は歪んでいない |
| `Age` の欠損処理 | 落とす / 全体中央値 / 群別中央値 / 中央値+フラグ | 現行維持 |
| 木の深さ | 1〜10と制限なし | 深さ3 |

4つとも「変更に足る根拠なし」で終わり、提出スコアも既存と同じ `0.77990` でした。
その過程で、判定ルールの形が結論を反転させたこと、継続を決めた推定値が独立な観測で
半減したこと、分解能以下の差に付けた順位が入れ替わったことが観測できています。

対応するスクリプトは `09_cabin_representation_comparison.py` から
`13_depth_selection.py` です。

生成と全セル検証:

```powershell
python build_preregistration_notebook.py
python validate_preregistration_notebook.py
```

検証スクリプトは、Notebookが出力する提出物が
`submissions/decision_tree_depth3_baseline.csv` と同一であることも確認します。
公開手順は `NOTEBOOK_PUBLISHING.md` にあります。

### 位置づけ

このリポジトリは、公開Notebookを模倣して高スコアを主張するものではありません。面接で、仮説、変更条件、検証結果、誤判定、失敗からの学びを自分の言葉で説明できることを重視したケーススタディです。

---

## English

### Overview

This repository is a learning-focused case study based on the Kaggle Titanic competition. Its purpose is not to present a top-ranking solution, but to demonstrate a traceable machine-learning experiment process:

- hypothesis-driven feature changes
- validation split checks
- controlled model comparison
- subgroup error analysis
- failed-experiment review
- reproducible submission generation

Competition: <https://www.kaggle.com/competitions/titanic>

### Baseline

```text
model: DecisionTreeClassifier(max_depth=3, random_state=42)
features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, CabinKnown
validation: 80:20 stratified split
validation accuracy: 0.7933
Kaggle public score: 0.77990
```

The first split missed all 24 male survivors. Repeating the split showed that the all-missed result was split-dependent, while the broader difficulty in capturing male survivors remained.

### Failed Experiment

The project tested the hypothesis that men with and without family might behave differently by introducing `FamilySize = SibSp + Parch + 1`.

A later submission replaced `SibSp` and `Parch` with `FamilySize` and changed the tree to `max_depth=5`. Its Kaggle public score fell to `0.74162`.

This was not a performance improvement. It was retained because it demonstrated that:

- feature aggregation can discard useful information
- a deeper tree does not guarantee better generalization
- changing features and model depth together makes causal interpretation difficult

### Reproduce

Kaggle data is not included. Place `train.csv`, `test.csv`, and `gender_submission.csv` under `data/`, then run:

```powershell
pip install -r requirements.txt
python 02_baseline.py
```

This repository intentionally emphasizes explainable experiment management over leaderboard optimization.
