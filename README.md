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
