from pathlib import Path

import nbformat as nbf


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_DIR / "titanic_explainable_bilingual_workflow.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


md(
    r"""
# Titanic: スコア悪化の原因を1回1変更で切り分ける
## Separating feature effects from model complexity | JP / EN

過去の提出では、決定木の深さと `FamilySize` の導入を同時に変更した。その結果、Public Score は
**0.77990 から 0.74162へ低下した**。しかし、この比較からは、特徴量とモデル複雑度のどちらが
悪化に寄与したのか分からない。

そこで本Notebookでは、同じLogistic Regressionを用いて特徴表現だけを変更し、反復交差検証で
差を確認する。高スコアモデルの提示ではなく、**比較条件を固定し、失敗から何を言えるかを
限定するためのケーススタディ**である。

### English abstract

An earlier submission changed both tree depth and `FamilySize`, so its lower public score could not identify the cause.
This notebook holds logistic regression fixed, changes only the feature representation, and evaluates the difference
with repeated cross-validation. It is an experiment audit rather than a claim of leaderboard superiority.

**構成 / Contents:** 問題設定 → データ確認 → 観察 → 統制比較 → 誤分類分析 → 提出検査 → 結論と限界
"""
)

md(
    r"""
## 1. 問題設定 / Question

確認する問いは次の1点である。

> `FamilySize`を含む家族特徴量は、モデルを固定した条件でも未知データへの精度を改善するか。

### 比較条件 / Experiment contract

| Item | Decision |
|---|---|
| Target / 目的変数 | `Survived` (0 or 1) |
| Primary metric / 主指標 | Accuracy（Kaggle と同じ） |
| Secondary metrics / 補助指標 | ROC AUC, F1 |
| Validation / 検証 | Repeated stratified K-fold |
| Controlled ablation / 比較 | Same logistic model: base vs. family features |
| Flexible model / 柔軟なモデル | Random forest on enhanced features |
| Seed / 乱数 | 42 |

可視化から確認できるのは観察上の関連であり、性別や客室等が生存の原因であるとは断定しない。
The visual patterns are associations, not causal estimates.
"""
)

md(
    r"""
## 2. 実行環境 / Setup

Kaggleでは `/kaggle/input/titanic/`、ローカルでは `data/` を参照する。公式データと
スモークデータを形状で区別し、非公式データの結果を公開用の数値として扱わない。
"""
)

code(
    r"""
from pathlib import Path
import platform
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from IPython.display import display, Markdown
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
PALETTE = {
    "blue": "#2F6B9A",
    "blue_light": "#A9C9E2",
    "gold": "#D39B2A",
    "gold_light": "#F2D79C",
    "ink": "#24323F",
    "muted": "#687784",
    "grid": "#DDE3E8",
    "paper": "#FAFBFC",
}

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.facecolor": PALETTE["paper"],
    "axes.facecolor": PALETTE["paper"],
    "axes.edgecolor": PALETTE["ink"],
    "axes.labelcolor": PALETTE["ink"],
    "text.color": PALETTE["ink"],
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.8,
    "axes.titleweight": "bold",
    "figure.dpi": 120,
})

print(f"Python {platform.python_version()} | pandas {pd.__version__} | scikit-learn {sklearn.__version__}")
"""
)

code(
    r"""
KAGGLE_DATA_DIR = Path("/kaggle/input/titanic")
LOCAL_DATA_DIR = Path("data")

if (KAGGLE_DATA_DIR / "train.csv").exists():
    DATA_DIR = KAGGLE_DATA_DIR
    OUTPUT_DIR = Path("/kaggle/working")
    ENVIRONMENT = "Kaggle"
elif (LOCAL_DATA_DIR / "train.csv").exists():
    DATA_DIR = LOCAL_DATA_DIR
    OUTPUT_DIR = Path("submissions")
    ENVIRONMENT = "Local"
else:
    raise FileNotFoundError(
        "train.csv and test.csv were not found. "
        "Kaggle: attach the Titanic competition dataset. "
        "Local: place files under data/."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

is_official_shape = train.shape == (891, 12) and test.shape == (418, 11)
data_mode = "official Titanic competition data" if is_official_shape else "non-official/smoke data"

display(Markdown(
    f"**Environment / 実行環境:** `{ENVIRONMENT}`  \n"
    f"**Data mode / データ判定:** `{data_mode}`  \n"
    f"**Shapes / 形状:** train `{train.shape}`, test `{test.shape}`"
))

if not is_official_shape:
    display(Markdown(
        "> ⚠️ **Smoke-test mode / スモークテスト:** "
        "The code path can be checked, but the displayed statistics are not Kaggle Titanic results. "
        "Attach the official dataset before publishing.  \n"
        "> コード動作確認用データです。投稿前に公式データで再実行してください。"
    ))
"""
)

md(
    r"""
### 2.1 入力を検査する / Validate inputs

主キー、目的変数、列の一致を分析前に検査する。入力条件が崩れていれば、後段の図表も比較結果も
信用できないためである。
"""
)

code(
    r"""
expected_train_columns = {
    "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
}
expected_test_columns = expected_train_columns - {"Survived"}

assert set(train.columns) == expected_train_columns, "Unexpected train schema"
assert set(test.columns) == expected_test_columns, "Unexpected test schema"
assert train["PassengerId"].is_unique, "Duplicate PassengerId in train"
assert test["PassengerId"].is_unique, "Duplicate PassengerId in test"
assert set(train["Survived"].dropna().unique()).issubset({0, 1}), "Target must be binary"
assert train["Survived"].notna().all(), "Target contains missing values"

quality_checks = pd.DataFrame({
    "check": [
        "train rows", "test rows", "duplicate train IDs",
        "duplicate test IDs", "missing targets", "positive target rate"
    ],
    "value": [
        len(train), len(test), train["PassengerId"].duplicated().sum(),
        test["PassengerId"].duplicated().sum(), train["Survived"].isna().sum(),
        f"{train['Survived'].mean():.1%}",
    ],
})
display(quality_checks)
"""
)

md(
    r"""
## 3. データを確認する / Data

### 3.1 主要列 / Compact data dictionary

| Column | 日本語 | English |
|---|---|---|
| `Pclass` | 客室等級 (1–3) | Passenger class |
| `Sex` | 性別 | Recorded sex |
| `Age` | 年齢 | Age in years |
| `SibSp` | 同乗した兄弟・配偶者数 | Siblings/spouses aboard |
| `Parch` | 同乗した親・子ども数 | Parents/children aboard |
| `Fare` | 運賃 | Ticket fare |
| `Cabin` | 客室番号 | Cabin identifier |
| `Embarked` | 乗船港 | Port of embarkation |
| `Survived` | 生存 (1) / 非生存 (0) | Outcome |

Source: [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/data)
テストデータの正解ラベルは非公開であるため、モデル比較には `train.csv` のみを用いる。
"""
)

code(
    r"""
preview = train.head(5).copy()
display(preview)

missing_table = (
    train.isna()
    .agg(["sum", "mean"])
    .T
    .rename(columns={"sum": "missing_count", "mean": "missing_rate"})
    .sort_values("missing_rate", ascending=False)
)
missing_table["missing_count"] = missing_table["missing_count"].astype(int)
missing_display = missing_table.head(8).copy()
missing_display["missing_rate"] = missing_display["missing_rate"].map("{:.1%}".format)
display(missing_display)
"""
)

md(
    r"""
## 4. 観察から仮説を作る / Observation

可視化では比率だけでなく分母を併記する。小標本の高い比率を、安定した傾向と誤認しないためである。
"""
)

code(
    r"""
missing_plot = missing_table.query("missing_count > 0").sort_values("missing_rate")

fig, ax = plt.subplots(figsize=(8, 3.8))
bars = ax.barh(
    missing_plot.index,
    missing_plot["missing_rate"],
    color=PALETTE["blue"],
    edgecolor=PALETTE["ink"],
    linewidth=0.8,
)
ax.bar_label(bars, labels=[f"{v:.1%}" for v in missing_plot["missing_rate"]], padding=4)
ax.set(
    title="Missing values by feature",
    xlabel="Missing share of training rows",
    ylabel="",
    xlim=(0, max(1.0, missing_plot["missing_rate"].max() * 1.18)),
)
sns.despine(ax=ax)
plt.tight_layout()
plt.show()
"""
)

md(
    r"""
欠損率が高い列を単純に削除せず、欠損そのものが情報を持つ可能性を確認する。`Cabin`は
文字列を直接使わず、まず`CabinKnown`（記録の有無）へ圧縮する。`Age`は学習fold内の
中央値で補完し、検証データの情報が前処理へ混入しないようにする。

*Missingness may carry information. Imputation is learned inside each training fold.*
"""
)

code(
    r"""
def rate_table(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    result = (
        frame.groupby(group_col, dropna=False)["Survived"]
        .agg(passengers="size", survivors="sum", survival_rate="mean")
        .reset_index()
    )
    return result

sex_rates = rate_table(train, "Sex").sort_values("survival_rate", ascending=False)
class_rates = rate_table(train, "Pclass").sort_values("Pclass")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

for ax, table, category, title in [
    (axes[0], sex_rates, "Sex", "Survival rate by recorded sex"),
    (axes[1], class_rates, "Pclass", "Survival rate by passenger class"),
]:
    bars = ax.bar(
        table[category].astype(str),
        table["survival_rate"],
        color=[PALETTE["blue"], PALETTE["gold"], PALETTE["blue_light"]][:len(table)],
        edgecolor=PALETTE["ink"],
        linewidth=0.8,
    )
    labels = [
        f"{rate:.1%}\n(n={count})"
        for rate, count in zip(table["survival_rate"], table["passengers"])
    ]
    ax.bar_label(bars, labels=labels, padding=4, fontsize=9)
    ax.set(title=title, xlabel="", ylabel="Survival rate", ylim=(0, 1.12))
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    sns.despine(ax=ax)

fig.suptitle("Observed survival patterns (training data)", y=1.04, fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

sex_rates_display = sex_rates.copy()
class_rates_display = class_rates.copy()
sex_rates_display["survival_rate"] = sex_rates_display["survival_rate"].map("{:.1%}".format)
class_rates_display["survival_rate"] = class_rates_display["survival_rate"].map("{:.1%}".format)
display(sex_rates_display)
display(class_rates_display)
"""
)

md(
    r"""
棒の高さだけでなく`n=`を確認する。大きな率の差が見られても、救命方針、客室位置、
家族構成などが交絡しているため、この図だけから因果関係は判断できない。
"""
)

code(
    r"""
eda = train.copy()
eda["FamilySize"] = eda["SibSp"] + eda["Parch"] + 1
eda["IsAlone"] = np.where(eda["FamilySize"].eq(1), "Alone", "With family")

family_rates = (
    eda.groupby("FamilySize")["Survived"]
    .agg(passengers="size", survival_rate="mean")
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

sns.histplot(
    data=eda,
    x="Age",
    hue="Survived",
    bins=18,
    stat="density",
    common_norm=False,
    element="step",
    fill=False,
    palette={0: PALETTE["gold"], 1: PALETTE["blue"]},
    linewidth=2,
    ax=axes[0],
)
axes[0].set(title="Age distribution by outcome", xlabel="Age (years)", ylabel="Density")

axes[1].scatter(
    family_rates["FamilySize"],
    family_rates["survival_rate"],
    s=np.clip(family_rates["passengers"] * 8, 35, 500),
    color=PALETTE["blue"],
    edgecolor=PALETTE["ink"],
    linewidth=0.8,
)
for row in family_rates.itertuples():
    axes[1].annotate(
        f"n={row.passengers}",
        (row.FamilySize, row.survival_rate),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=8,
    )
axes[1].set(
    title="Survival rate by family size",
    xlabel="Family size (SibSp + Parch + 1)",
    ylabel="Survival rate",
    ylim=(-0.05, 1.08),
)
axes[1].yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")

for ax in axes:
    sns.despine(ax=ax)
plt.tight_layout()
plt.show()
"""
)

md(
    r"""
### 4.1 仮説 / Hypothesis

家族人数と生存率の間には非線形な関係が見られる。しかし、学習データ上でパターンが見えたことと、
未知データへの予測精度が改善することは別である。

> **H1:** `FamilySize`, `IsAlone`, `Title`, `CabinKnown` add stable predictive signal.
> **H0:** The apparent pattern is redundant, unstable, or too sparse to improve generalization.

そこで、同じLogistic Regressionのまま特徴表現だけを変更する。ここで差が再現すれば、
少なくとも「モデル複雑度を変えなくても家族特徴量が追加情報を持つ」という判断材料になる。
"""
)

md(
    r"""
## 5. 条件を固定して比較する / Controlled experiment

### 5.1 目的変数を使わない特徴量設計 / Feature engineering

- `Title`: 氏名から敬称を抽出し、低頻度カテゴリを `Rare` に統合
- `FamilySize`: `SibSp + Parch + 1`
- `IsAlone`: `FamilySize == 1`
- `CabinKnown`: 客室番号が記録されているか

すべて`Survived`を参照せずに作成する。補完・標準化・one-hot encodingはPipeline内に置き、
各foldの学習データだけから推定する。
"""
)

code(
    r"""
COMMON_TITLES = {"Mr", "Miss", "Mrs", "Master"}

def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.copy()
    extracted_title = features["Name"].str.extract(r",\s*([^.]*)\.", expand=False).str.strip()
    features["Title"] = extracted_title.where(extracted_title.isin(COMMON_TITLES), "Rare")
    features["FamilySize"] = features["SibSp"] + features["Parch"] + 1
    features["IsAlone"] = features["FamilySize"].eq(1).astype(int)
    features["CabinKnown"] = features["Cabin"].notna().astype(int)
    return features

train_features = add_features(train)
test_features = add_features(test)

base_numeric = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
base_categorical = ["Sex", "Embarked"]
family_numeric = base_numeric + ["FamilySize", "IsAlone"]
family_categorical = base_categorical
enhanced_numeric = family_numeric + ["CabinKnown"]
enhanced_categorical = base_categorical + ["Title"]

display(
    train_features[
        ["Name", "Title", "FamilySize", "IsAlone", "CabinKnown"]
    ].head(8)
)
"""
)

code(
    r"""
def make_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def make_pipeline(model, numeric_features, categorical_features):
    return Pipeline([
        ("preprocess", make_preprocessor(numeric_features, categorical_features)),
        ("model", model),
    ])


model_specs = {
    "Dummy baseline": (
        DummyClassifier(strategy="most_frequent"),
        base_numeric,
        base_categorical,
    ),
    "Logistic · base": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        base_numeric,
        base_categorical,
    ),
    "Logistic · family": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        family_numeric,
        family_categorical,
    ),
    "Logistic · enhanced": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        enhanced_numeric,
        enhanced_categorical,
    ),
    "Random forest · enhanced": (
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        enhanced_numeric,
        enhanced_categorical,
    ),
}

X_all = train_features.drop(columns="Survived")
y = train_features["Survived"].astype(int)

minority_count = int(y.value_counts().min())
n_splits = min(5, minority_count)
if n_splits < 2:
    raise ValueError("At least two observations are required in each target class.")

repeated_cv = RepeatedStratifiedKFold(
    n_splits=n_splits,
    n_repeats=3,
    random_state=RANDOM_STATE,
)

print(f"Repeated stratified CV: {n_splits} folds × 3 repeats = {n_splits * 3} validation scores/model")
"""
)

md(
    r"""
### 5.2 反復交差検証 / Repeated cross-validation

平均値だけでなく、分割間の分布を確認する。小規模データでは、1回のholdoutが結論を
大きく変えるためである。
"""
)

code(
    r"""
scoring = {"accuracy": "accuracy", "roc_auc": "roc_auc", "f1": "f1"}
cv_rows = []
fitted_pipelines = {}

for model_name, (estimator, numeric_features, categorical_features) in model_specs.items():
    pipeline = make_pipeline(estimator, numeric_features, categorical_features)
    fitted_pipelines[model_name] = pipeline
    scores = cross_validate(
        pipeline,
        X_all,
        y,
        cv=repeated_cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
    )
    for fold_index in range(len(scores["test_accuracy"])):
        cv_rows.append({
            "model": model_name,
            "split": fold_index + 1,
            "accuracy": scores["test_accuracy"][fold_index],
            "roc_auc": scores["test_roc_auc"][fold_index],
            "f1": scores["test_f1"][fold_index],
        })

cv_results = pd.DataFrame(cv_rows)
cv_summary = (
    cv_results.groupby("model", sort=False)
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        accuracy_min=("accuracy", "min"),
        accuracy_max=("accuracy", "max"),
        roc_auc_mean=("roc_auc", "mean"),
        f1_mean=("f1", "mean"),
    )
    .reset_index()
    .sort_values("accuracy_mean", ascending=False)
)

cv_summary_display = cv_summary.copy()
for metric_column in [
    "accuracy_mean", "accuracy_std", "accuracy_min",
    "accuracy_max", "roc_auc_mean", "f1_mean",
]:
    cv_summary_display[metric_column] = cv_summary_display[metric_column].map("{:.3f}".format)
display(cv_summary_display)
"""
)

code(
    r"""
model_order = cv_summary["model"].tolist()

fig, ax = plt.subplots(figsize=(9.5, 4.8))
sns.boxplot(
    data=cv_results,
    x="accuracy",
    y="model",
    order=model_order,
    color=PALETTE["blue_light"],
    linecolor=PALETTE["ink"],
    whis=(0, 100),
    ax=ax,
)
sns.stripplot(
    data=cv_results,
    x="accuracy",
    y="model",
    order=model_order,
    color=PALETTE["blue"],
    size=4,
    alpha=0.7,
    ax=ax,
)
ax.set(
    title="Repeated cross-validation accuracy",
    xlabel=f"Accuracy across {n_splits * 3} validation folds per model",
    ylabel="",
    xlim=(0, 1.01),
)
ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
sns.despine(ax=ax)
plt.tight_layout()
plt.show()
"""
)

code(
    r"""
summary_lookup = cv_summary.set_index("model")
base_mean = summary_lookup.loc["Logistic · base", "accuracy_mean"]
family_mean = summary_lookup.loc["Logistic · family", "accuracy_mean"]
feature_delta = family_mean - base_mean

best_model_name = (
    cv_summary.loc[cv_summary["model"] != "Dummy baseline"]
    .sort_values(["accuracy_mean", "accuracy_std"], ascending=[False, True])
    .iloc[0]["model"]
)

direction = "higher" if feature_delta > 0 else "lower or equal"
direction_ja = "高かった" if feature_delta > 0 else "同等以下だった"

display(Markdown(
    f"### 特徴量比較の結果 / Controlled ablation result\n"
    f"- Logistic base mean accuracy: **{base_mean:.3f}**\n"
    f"- Logistic + family mean accuracy: **{family_mean:.3f}**\n"
    f"- Difference (family − base): **{feature_delta:+.3f}**\n\n"
    f"同じLogistic Regressionでは、家族特徴量を加えた平均CV Accuracyの方が**{direction_ja}**。"
    f"ただし、この平均差だけでは安定した改善とは断定しない。  \n"
    f"Under the same logistic model, family features produced a **{direction}** mean CV accuracy. "
    f"The mean difference alone does not establish a stable improvement.\n\n"
    f"誤分類診断に用いるモデル / Model used for diagnostics: **{best_model_name}**"
))
"""
)

md(
    r"""
### 5.3 過去の失敗をどう読み直すか

過去の試行では`FamilySize`導入と木の深さ変更を同時に行い、Public Scoreが
**0.77990 → 0.74162**に低下した。その結果だけでは、特徴量と深さのどちらが
悪化に寄与したのか分離できない。

| Tracked submission | Public score | Interpretation |
|---|---:|---|
| Depth-3 baseline | 0.77990 | Reference |
| Depth-5 + FamilySize replacement | 0.74162 | Failed experiment; two factors changed |

本Notebookでは、同じモデルで特徴表現だけを変えた。したがって、ここで得られる差は
過去のPublic Score低下を直接説明するものではないが、`FamilySize`だけを原因とみなす根拠が
弱いことは確認できる。

*The leaderboard scores are historical records. The controlled ablation isolates feature representation, but it does not reconstruct the old tree experiment.*
"""
)

md(
    r"""
## 6. 全体精度の外側を見る / Error analysis

モデル選択と性能推定に同じCVを用いると、選択後の数値は楽観的になる。ここでは最終性能を
断定するためではなく、**どのように誤るかを診断するため**にOOF predictionを用いる。
選択後性能を不偏に推定するにはnested CVが必要である。
"""
)

code(
    r"""
diagnostic_cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=RANDOM_STATE,
)
best_pipeline = clone(fitted_pipelines[best_model_name])

oof_pred = cross_val_predict(
    best_pipeline,
    X_all,
    y,
    cv=diagnostic_cv,
    method="predict",
    n_jobs=1,
)
oof_proba = cross_val_predict(
    best_pipeline,
    X_all,
    y,
    cv=diagnostic_cv,
    method="predict_proba",
    n_jobs=1,
)[:, 1]

oof_accuracy = accuracy_score(y, oof_pred)
oof_auc = roc_auc_score(y, oof_proba)
oof_f1 = f1_score(y, oof_pred)

fig, ax = plt.subplots(figsize=(5.2, 4.2))
ConfusionMatrixDisplay.from_predictions(
    y,
    oof_pred,
    display_labels=["Not survived", "Survived"],
    cmap="Blues",
    colorbar=False,
    ax=ax,
)
ax.set_title(f"OOF confusion matrix · {best_model_name}\nAccuracy={oof_accuracy:.3f}, AUC={oof_auc:.3f}, F1={oof_f1:.3f}")
plt.tight_layout()
plt.show()
"""
)

code(
    r"""
diagnostic = train_features[["PassengerId", "Sex", "Pclass", "Survived"]].copy()
diagnostic["Predicted"] = oof_pred
diagnostic["Correct"] = diagnostic["Survived"].eq(diagnostic["Predicted"])

subgroup_rows = []
for dimension in ["Sex", "Pclass"]:
    for group_value, group in diagnostic.groupby(dimension):
        actual_survivors = group["Survived"].eq(1)
        subgroup_rows.append({
            "dimension": dimension,
            "group": str(group_value),
            "passengers": len(group),
            "survivors": int(actual_survivors.sum()),
            "accuracy": accuracy_score(group["Survived"], group["Predicted"]),
            "survivor_recall": (
                recall_score(group["Survived"], group["Predicted"], pos_label=1)
                if actual_survivors.any() else np.nan
            ),
        })

subgroup_metrics = pd.DataFrame(subgroup_rows)
subgroup_display = subgroup_metrics.copy()
subgroup_display["accuracy"] = subgroup_display["accuracy"].map("{:.1%}".format)
subgroup_display["survivor_recall"] = subgroup_display["survivor_recall"].map(
    lambda value: "N/A" if pd.isna(value) else f"{value:.1%}"
)
display(subgroup_display)

plot_metrics = subgroup_metrics.melt(
    id_vars=["dimension", "group", "passengers", "survivors"],
    value_vars=["accuracy", "survivor_recall"],
    var_name="metric",
    value_name="value",
)
plot_metrics["label"] = plot_metrics["dimension"] + ": " + plot_metrics["group"]

fig, ax = plt.subplots(figsize=(9.5, 4.8))
sns.barplot(
    data=plot_metrics,
    x="value",
    y="label",
    hue="metric",
    palette=[PALETTE["blue"], PALETTE["gold"]],
    edgecolor=PALETTE["ink"],
    ax=ax,
)
ax.set(
    title="OOF diagnostic metrics by subgroup",
    xlabel="Rate (check subgroup sample sizes in the table)",
    ylabel="",
    xlim=(0, 1.02),
)
ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
ax.legend(
    title="",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
    frameon=False,
)
sns.despine(ax=ax)
plt.tight_layout()
plt.show()
"""
)

md(
    r"""
`survivor_recall`は、実際に生存した人のうち生存と予測できた割合である。全体Accuracyが
同じでも、グループごとに見逃し方は異なり得る。小標本の率は不安定であるため、
必ず`survivors`列と併せて確認する。

*This is a failure-mode diagnostic, not a fairness certification.*
"""
)

md(
    r"""
### 6.1 置換重要度 / Permutation importance

別holdout上で1列ずつランダムに崩し、Accuracyの低下を測る。結果はモデルと分割に依存し、
負値も有限標本の揺れや特徴量の冗長性によって生じる。因果効果としては解釈しない。
"""
)

code(
    r"""
_, best_numeric, best_categorical = model_specs[best_model_name]
best_feature_columns = best_numeric + best_categorical
X_for_importance = X_all[best_feature_columns]

X_train, X_valid, y_train, y_valid = train_test_split(
    X_for_importance,
    y,
    test_size=0.25,
    stratify=y,
    random_state=RANDOM_STATE,
)
importance_pipeline = clone(fitted_pipelines[best_model_name])
importance_pipeline.fit(X_train, y_train)

importance = permutation_importance(
    importance_pipeline,
    X_valid,
    y_valid,
    scoring="accuracy",
    n_repeats=15,
    random_state=RANDOM_STATE,
    n_jobs=1,
)
importance_table = (
    pd.DataFrame({
        "feature": X_valid.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    })
    .sort_values("importance_mean", ascending=False)
    .head(10)
    .sort_values("importance_mean")
)

fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.barh(
    importance_table["feature"],
    importance_table["importance_mean"],
    xerr=importance_table["importance_std"],
    color=PALETTE["blue"],
    edgecolor=PALETTE["ink"],
    ecolor=PALETTE["muted"],
    capsize=3,
)
ax.axvline(0, color=PALETTE["ink"], linewidth=1)
ax.set(
    title=f"Permutation importance on one validation split · {best_model_name}",
    xlabel="Mean decrease in validation accuracy (15 permutations)",
    ylabel="",
)
sns.despine(ax=ax)
plt.tight_layout()
plt.show()
"""
)

md(
    r"""
## 7. 提出ファイルを検査する / Submission checks

最終モデルを全学習データでfitし、`PassengerId`と0/1予測だけを保存する。保存前に行数、
ID順序、重複、欠損、予測値の範囲を検査する。
"""
)

code(
    r"""
final_pipeline = clone(fitted_pipelines[best_model_name])
final_pipeline.fit(X_all, y)
test_pred = final_pipeline.predict(test_features.drop(columns="Survived", errors="ignore")).astype(int)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_pred,
})

assert len(submission) == len(test), "Submission row count mismatch"
assert submission["PassengerId"].equals(test["PassengerId"]), "PassengerId order changed"
assert submission["PassengerId"].is_unique, "Duplicate PassengerId"
assert set(submission["Survived"].unique()).issubset({0, 1}), "Predictions must be 0/1"
assert submission.isna().sum().sum() == 0, "Submission contains missing values"

submission_filename = "submission.csv" if is_official_shape else "submission_smoke.csv"
submission_path = OUTPUT_DIR / submission_filename
submission.to_csv(submission_path, index=False)

display(submission.head())
display(Markdown(
    f"✅ Saved / 保存先: `{submission_path}`  \n"
    f"Rows / 行数: **{len(submission)}** · Predicted survival rate / 予測生存率: **{submission['Survived'].mean():.1%}**"
))
"""
)

md(
    r"""
## 8. ここまでで確認できたこと / Conclusions

以下の数値は実行結果から生成する。特徴量やモデルを変更した場合でも、本文の結論と出力値が
ずれにくいようにするためである。
"""
)

code(
    r"""
winner = cv_summary.iloc[0]
spread = winner["accuracy_max"] - winner["accuracy_min"]
ablation_message_ja = (
    "家族特徴量を加えた平均値の方が高かったものの、この差だけでは安定した改善とは断定できません。"
    if feature_delta > 0
    else "家族特徴量を加えた平均値は同等以下でした。"
)
ablation_message_en = (
    "Family features had a higher mean, but this difference alone does not establish a stable improvement."
    if feature_delta > 0
    else "Family features had an equal or lower mean under the same logistic model."
)

official_note = (
    "公式 Titanic データで実行済みです。"
    if is_official_shape
    else "現在はスモークデータです。投稿前に公式 Titanic データで再実行してください。"
)
official_note_en = (
    "Executed on the official Titanic data."
    if is_official_shape
    else "Currently executed on smoke data; rerun on the official Titanic data before publishing."
)

display(Markdown(
    f"### 日本語\n"
    f"1. 反復 CV の最高平均 Accuracy は **{winner['accuracy_mean']:.3f}** "
    f"（**{winner['model']}**）でした。\n"
    f"2. 最高モデルでも split 間の最小–最大幅は **{spread:.3f}**。単一 split の結論は不安定です。\n"
    f"3. Base → Family Logistic の差は **{feature_delta:+.3f}**。{ablation_message_ja}\n"
    f"4. OOF 診断は Accuracy **{oof_accuracy:.3f}**, ROC AUC **{oof_auc:.3f}**, F1 **{oof_f1:.3f}**。\n"
    f"5. {official_note}\n\n"
    f"### English\n"
    f"1. Best repeated-CV mean accuracy: **{winner['accuracy_mean']:.3f}** "
    f"(**{winner['model']}**).\n"
    f"2. Its min–max split spread is **{spread:.3f}**, so one split is not enough.\n"
    f"3. Base → family logistic delta: **{feature_delta:+.3f}**. {ablation_message_en}\n"
    f"4. OOF diagnostic: accuracy **{oof_accuracy:.3f}**, ROC AUC **{oof_auc:.3f}**, F1 **{oof_f1:.3f}**.\n"
    f"5. {official_note_en}"
))
"""
)

md(
    r"""
### 8.1 ここから先は言えない / Limitations

- CVはleaderboardの完全な代替ではない。
- モデル選択後のOOF指標は診断用であり、不偏な最終性能推定ではない。
- サブグループ率は小標本で不安定である。
- 置換重要度は因果効果を示さない。
- Titanicは歴史的・倫理的にセンシティブな実データであり、属性差を人物評価へ一般化しない。

### 8.2 次に行うなら / Next experiments

1. `Title`だけ、`FamilySize`だけを1変更ずつablationする。
2. Nested CVでモデル選択バイアスを分離する。
3. Calibration curveとthreshold sensitivityを追加する。
4. 同一CV splitを固定し、実験表に平均・標準偏差・差分を記録する。
"""
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3",
    },
}

nbf.write(nb, OUTPUT_PATH)
print(f"wrote {OUTPUT_PATH}")
