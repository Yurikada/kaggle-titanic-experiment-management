# %% Imports
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# %% Registered contract (fixed before running)
# 問い     : 決定木の深さをいくつに確定するか
# 構成     : Experiment 2026-07-31-01 と -03 で確定した構成
#            Pclass, Sex, Age(全体中央値補完), SibSp, Parch, Fare, Embarked, CabinKnown
# 候補     : 深さ 1〜10 と制限なしの11通り
# 分割     : 層化5-fold x 20 seed = 100 fold。全候補が同一のfoldを使う。
# 指標     : accuracy
# 判定     : ペア差版の1-SEルール
#            各foldで最良候補とのペア差を取り、平均 <= 1.0 * SE を満たす候補を
#            「最良と区別がつかない」とする。その中で最も単純なものを選ぶ。
# 単純さ   : 平均の葉の数が少ないほど単純（今回の補足登録）。
#            列数は全候補で同じなので、従来の第1優先は機能しない。
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

TARGET_COLUMN = "Survived"
DEPTH_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]
DEPTH_LABELS = ["unlimited" if depth is None else str(depth) for depth in DEPTH_GRID]
SEEDS = list(range(20))
N_SPLITS = 5
TREE_RANDOM_STATE = 42
ONE_SE_MULTIPLIER = 1.0
EXPECTED_FOLDS = len(SEEDS) * N_SPLITS

assert len(SEEDS) == 20


# %% Load and build the settled feature set
train = pd.read_csv(DATA_DIR / "train.csv")
assert train.shape == (891, 12)
y = train[TARGET_COLUMN]

features = train.assign(CabinKnown=train["Cabin"].notna().astype(int))[
    ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "CabinKnown"]
]


def build_pipeline(max_depth) -> Pipeline:
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()
    preprocess = ColumnTransformer([
        (
            "numeric",
            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
            numeric_columns,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_columns,
        ),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        (
            "model",
            DecisionTreeClassifier(max_depth=max_depth, random_state=TREE_RANDOM_STATE),
        ),
    ])


# %% Run: identical folds for every candidate depth
records = []
for seed in SEEDS:
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold_number, (train_index, valid_index) in enumerate(
        splitter.split(features, y),
        start=1,
    ):
        for depth, label in zip(DEPTH_GRID, DEPTH_LABELS):
            pipeline = build_pipeline(depth)
            pipeline.fit(features.iloc[train_index], y.iloc[train_index])
            records.append({
                "seed": seed,
                "fold": fold_number,
                "depth_label": label,
                "train_accuracy": (
                    pipeline.predict(features.iloc[train_index])
                    == y.iloc[train_index].to_numpy()
                ).mean(),
                "validation_accuracy": (
                    pipeline.predict(features.iloc[valid_index])
                    == y.iloc[valid_index].to_numpy()
                ).mean(),
                "leaves": int(pipeline.named_steps["model"].get_n_leaves()),
            })

results = pd.DataFrame(records)
assert len(results) == EXPECTED_FOLDS * len(DEPTH_GRID)

by_fold = results.pivot(
    index=["seed", "fold"],
    columns="depth_label",
    values="validation_accuracy",
).reindex(columns=DEPTH_LABELS)
assert len(by_fold) == EXPECTED_FOLDS

summary = (
    results.groupby("depth_label")
    .agg(
        validation_mean=("validation_accuracy", "mean"),
        validation_std=("validation_accuracy", "std"),
        train_mean=("train_accuracy", "mean"),
        leaves_mean=("leaves", "mean"),
    )
    .reindex(DEPTH_LABELS)
)
summary["train_minus_validation"] = summary["train_mean"] - summary["validation_mean"]

print("== 候補ごとの結果（100 fold）==")
print(summary.round(4).to_string())


# %% Paired 1-SE rule, complexity measured by mean leaves
best_label = summary["validation_mean"].idxmax()
print()
print(
    f"最良候補: 深さ {best_label}"
    f"（平均 {summary.loc[best_label, 'validation_mean']:.4f}、"
    f"葉 {summary.loc[best_label, 'leaves_mean']:.1f}）"
)

rows = []
for label in DEPTH_LABELS:
    delta = (by_fold[best_label] - by_fold[label]).to_numpy()
    assert delta.size == EXPECTED_FOLDS
    mean_delta = delta.mean()
    standard_deviation = delta.std(ddof=1)
    se_delta = standard_deviation / np.sqrt(delta.size) if standard_deviation > 0 else 0.0
    rows.append({
        "depth": label,
        "mean_delta": mean_delta,
        "se_delta": se_delta,
        "threshold_1se": ONE_SE_MULTIPLIER * se_delta,
        "within_1se": mean_delta <= ONE_SE_MULTIPLIER * se_delta,
        "leaves_mean": summary.loc[label, "leaves_mean"],
    })

comparison = pd.DataFrame(rows)
print()
print("== 最良候補とのペア差 ==")
print(comparison.round(5).to_string(index=False))

candidates = comparison[comparison["within_1se"]].sort_values("leaves_mean")
selected = candidates.iloc[0]

print()
print("== 選択 ==")
print(f"  1SE以内の候補: {', '.join(candidates['depth'])}")
print(
    f"  選択: 深さ {selected['depth']}"
    f"（平均の葉 {selected['leaves_mean']:.1f}、"
    f"検証正解率 {summary.loc[selected['depth'], 'validation_mean']:.4f}）"
)
print(
    f"  最良候補との差 {selected['mean_delta']:+.5f}、"
    f"1SE閾値 {selected['threshold_1se']:.5f}"
)
