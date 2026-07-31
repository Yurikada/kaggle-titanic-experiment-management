# %% Imports
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# %% Registered contract (fixed before running)
# 問い   : 決定木を超える本質的な表現力が他の族にあるか
# 特徴量 : Experiment -01 と -03 で確定した8列。族をまたいで固定
# 族     : 決定木 / ロジスティック回帰 / ランダムフォレストの3本
# 前処理 : 族に付随するものとして扱う。線形のみ数値列を標準化する
# 探索   : 各外側学習foldの内側3-foldでハイパーパラメータを選ぶ。
#          gridは事前に明示したものであり、最適値の証明ではない
# 分割   : 層化5-fold x 20 seed = 100 外側fold。全族が同一のfoldを使う
# 指標   : accuracy
# 判定   : ペア差版の1-SEルール。最良族との平均差 <= 1SE の族のうち、
#          事前宣言した単純さ順位が最も高いものを選ぶ
# 単純さ順位（事前宣言）:
#          決定木 < ロジスティック回帰 < ランダムフォレスト
#          根拠: 決定木はIF-THEN規則として完全に可視化でき前処理も不要。
#          線形は係数で影響度を読めるが標準化を要する。
#          森は重要度は出せても個々の予測根拠を追えない。
# 解釈可能性の宣言:
#          決定木を捨てるのは、解釈性を犠牲にしてでも取る価値のある
#          明確な精度向上が実測された場合のみとする。
#          1SE以内の差であれば決定木を維持する。
# 採用後の扱い:
#          決定木族が選ばれた場合は現行構成（深さ3、Experiment -04で決定済み）を維持し、
#          新しい提出物は作らない。他の族が選ばれた場合はその族が新しい構成となる。
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

TARGET_COLUMN = "Survived"
SEEDS = list(range(20))
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 3
RANDOM_STATE = 42
ONE_SE_MULTIPLIER = 1.0

FEATURES = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "CabinKnown",
]
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare", "CabinKnown"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]

# 単純さ順位。小さいほど単純で、同点のとき優先される。
FAMILY_PRIORITY = {"decision tree": 0, "logistic regression": 1, "random forest": 2}

assert len(SEEDS) == 20


# %% Load
train = pd.read_csv(DATA_DIR / "train.csv")
assert train.shape == (891, 12)
prepared = train.assign(CabinKnown=train["Cabin"].notna().astype(int))
X = prepared[FEATURES]
y = prepared[TARGET_COLUMN]


# %% Pipelines: preprocessing belongs to the family
def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        # One-Hot後のダミー列は標準化しない。正則化がカテゴリ側へ偏るため。
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            CATEGORICAL_FEATURES,
        ),
    ])


FAMILIES = {
    "decision tree": {
        "scale_numeric": False,
        "model": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "grid": {"model__max_depth": [3, 4, 5, 6]},
    },
    "logistic regression": {
        "scale_numeric": True,
        # l1 を扱えるソルバーが要る。既定の lbfgs は l2 のみ。
        "model": LogisticRegression(
            solver="liblinear", max_iter=5000, random_state=RANDOM_STATE
        ),
        "grid": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__penalty": ["l1", "l2"],
        },
    },
    "random forest": {
        "scale_numeric": False,
        "model": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "grid": {
            "model__max_depth": [3, 4, 5],
            "model__max_features": ["sqrt", "log2"],
        },
    },
}


def build_pipeline(family_name: str) -> Pipeline:
    specification = FAMILIES[family_name]
    return Pipeline([
        ("preprocess", build_preprocessor(specification["scale_numeric"])),
        ("model", specification["model"]),
    ])


# %% Nested cross-validation
# 入れ子CVは14分かかる。集計側の不具合で計算をやり直さずに済むよう、
# fold単位の結果を先にCSVへ保存し、以降の実行では読み直す。
EXPERIMENT_DIR = PROJECT_DIR / "experiments"
EXPERIMENT_DIR.mkdir(exist_ok=True)
FOLD_RESULTS_PATH = EXPERIMENT_DIR / "model_family_nested_cv_folds.csv"

records = []
for seed in [] if FOLD_RESULTS_PATH.exists() else SEEDS:
    outer = StratifiedKFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=seed)
    for fold_number, (train_index, valid_index) in enumerate(outer.split(X, y), start=1):
        inner = StratifiedKFold(
            n_splits=N_INNER_SPLITS,
            shuffle=True,
            random_state=10_000 + seed * 10 + fold_number,
        )
        for family_name in FAMILIES:
            search = GridSearchCV(
                estimator=build_pipeline(family_name),
                param_grid=FAMILIES[family_name]["grid"],
                scoring="accuracy",
                cv=inner,
                n_jobs=1,
                refit=True,
                error_score="raise",
            )
            search.fit(X.iloc[train_index], y.iloc[train_index])
            predicted = search.predict(X.iloc[valid_index])
            record = {
                "seed": seed,
                "fold": fold_number,
                "family": family_name,
                "accuracy": (predicted == y.iloc[valid_index].to_numpy()).mean(),
            }
            record.update(
                {key: str(value) for key, value in search.best_params_.items()}
            )
            records.append(record)

EXPECTED_FOLDS = len(SEEDS) * N_OUTER_SPLITS
if FOLD_RESULTS_PATH.exists():
    results = pd.read_csv(FOLD_RESULTS_PATH)
    print(f"既存の結果を読み込み: {FOLD_RESULTS_PATH}")
else:
    results = pd.DataFrame(records)
    results.to_csv(FOLD_RESULTS_PATH, index=False)
    print(f"結果を保存: {FOLD_RESULTS_PATH}")
assert len(results) == EXPECTED_FOLDS * len(FAMILIES)


# %% Outer results
by_fold = results.pivot(
    index=["seed", "fold"], columns="family", values="accuracy"
).reindex(columns=list(FAMILIES))
assert len(by_fold) == EXPECTED_FOLDS

summary = pd.DataFrame({
    "mean_accuracy": by_fold.mean(),
    "std": by_fold.std(),
    "priority": pd.Series(FAMILY_PRIORITY),
})
print("== 族ごとの外側fold平均（100 fold）==")
print(summary.round(4).to_string())


# %% Which hyperparameters the inner CV chose, and whether the grid saturated
print()
print("== 内側CVが選んだパラメータの分布 ==")
for family_name in FAMILIES:
    subset = results[results["family"] == family_name]
    # 他の族のパラメータ列も results に混在するので、その族のgridにある列だけを見る
    parameter_columns = [
        column for column in FAMILIES[family_name]["grid"] if column in subset.columns
    ]
    counts = (
        subset.groupby(parameter_columns, dropna=False)
        .size()
        .rename("folds")
        .reset_index()
        .sort_values("folds", ascending=False)
    )
    print(f"-- {family_name}")
    print(counts.to_string(index=False))
    for column in parameter_columns:
        grid_values = FAMILIES[family_name]["grid"][column]
        numeric_grid = [v for v in grid_values if isinstance(v, (int, float))]
        if not numeric_grid:
            continue
        top = str(max(numeric_grid))
        share = (subset[column] == top).mean()
        if share >= 0.5:
            print(
                f"   注意: {column} が grid 上端 {top} を {share:.0%} の fold で選択。"
                "最適値が探索範囲の外にある可能性がある。"
            )


# %% Paired 1-SE rule with the declared family priority
best_family = summary["mean_accuracy"].idxmax()
rows = []
for family_name in FAMILIES:
    delta = (by_fold[best_family] - by_fold[family_name]).to_numpy()
    assert delta.size == EXPECTED_FOLDS
    deviation = delta.std(ddof=1)
    standard_error = deviation / np.sqrt(delta.size) if deviation > 0 else 0.0
    rows.append({
        "family": family_name,
        "deficit_from_best": delta.mean(),
        "se": standard_error,
        "within_1se": delta.mean() <= ONE_SE_MULTIPLIER * standard_error,
        "priority": FAMILY_PRIORITY[family_name],
    })

comparison = pd.DataFrame(rows)
print()
print(f"最良族: {best_family}（{summary.loc[best_family, 'mean_accuracy']:.4f}）")
print(comparison.round(5).to_string(index=False))

selected = (
    comparison[comparison["within_1se"]].sort_values("priority").iloc[0]
)
print()
print("== 選択 ==")
print(f"  1SE以内: {', '.join(comparison[comparison['within_1se']]['family'])}")
print(f"  選択: {selected['family']}")
if selected["family"] == "decision tree":
    print("  -> 現行構成（深さ3）を維持。新しい提出物は作らない。")
else:
    print("  -> 族を変更する。この族が新しい構成となる。")
