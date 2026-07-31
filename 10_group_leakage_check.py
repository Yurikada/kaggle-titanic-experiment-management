# %% Imports
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier


# %% Contract
# 問い     : 同じ家族が学習側と検証側へ分かれることで、検証正解率がどれだけ上振れするか
# グループ : Ticket と Surname を辺とみなした連結成分
# 設計A    : 分割方法を変える（StratifiedKFold と StratifiedGroupKFold）
# 設計B    : 同じ分割の中で、検証行を「学習側に仲間あり / なし」に分けて比べる
# 深さ     : 3（現行ベースライン。漏洩は深い木ほど大きいはずなので、これは下限の推定）
# 腕       : 落とす / 有無フラグ。上振れが腕によって違うかを見るため2つ測る
# 指標     : accuracy
# 解釈基準 : 0.007未満は分解能以下、0.007-0.015は絶対値を本番推定に使わない、
#            0.015超は過去の比較を読み直す
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

TARGET_COLUMN = "Survived"
BASE_FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
DEPTH = 3
SEEDS = [0, 1, 2, 3, 42]
N_SPLITS = 5
TREE_RANDOM_STATE = 42


# %% Load and build the connected components of Ticket and Surname
train = pd.read_csv(DATA_DIR / "train.csv")
assert train.shape == (891, 12)
train["Surname"] = train["Name"].str.split(",").str[0].str.strip()

parent = {index: index for index in train.index}


def find(node: int) -> int:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def union(left: int, right: int) -> None:
    root_left, root_right = find(left), find(right)
    if root_left != root_right:
        parent[root_right] = root_left


for column in ["Ticket", "Surname"]:
    for _, indices in train.groupby(column).groups.items():
        members = list(indices)
        for other in members[1:]:
            union(members[0], other)

groups = pd.Series([find(index) for index in train.index], index=train.index)
y = train[TARGET_COLUMN]
print(f"グループ数 {groups.nunique()} / 乗客 {len(train)}")


# %% Arms
arms = {
    "落とす": train[BASE_FEATURES].copy(),
    "有無フラグ": train[BASE_FEATURES].assign(
        CabinKnown=train["Cabin"].notna().astype(int)
    ),
}


def build_pipeline(features: pd.DataFrame) -> Pipeline:
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
            DecisionTreeClassifier(max_depth=DEPTH, random_state=TREE_RANDOM_STATE),
        ),
    ])


# %% Out-of-fold predictions under both split schemes
def run_scheme(features: pd.DataFrame, splitter, use_groups: bool) -> pd.DataFrame:
    rows = []
    split_arguments = (train, y, groups) if use_groups else (train, y)
    for fold_number, (train_index, valid_index) in enumerate(
        splitter.split(*split_arguments),
        start=1,
    ):
        pipeline = build_pipeline(features)
        pipeline.fit(features.iloc[train_index], y.iloc[train_index])
        prediction = pipeline.predict(features.iloc[valid_index])

        training_groups = set(groups.iloc[train_index])
        rows.append(pd.DataFrame({
            "fold": fold_number,
            "correct": (prediction == y.iloc[valid_index].to_numpy()).astype(int),
            "has_relative_in_train": groups.iloc[valid_index]
            .isin(training_groups)
            .to_numpy()
            .astype(int),
            "survived": y.iloc[valid_index].to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True)


records = []
design_b_records = []
for arm_name, features in arms.items():
    for seed in SEEDS:
        stratified = run_scheme(
            features,
            StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed),
            use_groups=False,
        )
        grouped = run_scheme(
            features,
            StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed),
            use_groups=True,
        )
        assert len(stratified) == len(grouped) == len(train)

        records.append({
            "arm": arm_name,
            "seed": seed,
            "stratified_accuracy": stratified["correct"].mean(),
            "grouped_accuracy": grouped["correct"].mean(),
            "grouped_exposure": grouped["has_relative_in_train"].mean(),
            "stratified_exposure": stratified["has_relative_in_train"].mean(),
        })

        with_relative = stratified[stratified["has_relative_in_train"] == 1]
        without_relative = stratified[stratified["has_relative_in_train"] == 0]
        design_b_records.append({
            "arm": arm_name,
            "seed": seed,
            "n_with": len(with_relative),
            "n_without": len(without_relative),
            "accuracy_with": with_relative["correct"].mean(),
            "accuracy_without": without_relative["correct"].mean(),
            "survival_with": with_relative["survived"].mean(),
            "survival_without": without_relative["survived"].mean(),
        })

design_a = pd.DataFrame(records)
design_a["inflation"] = design_a["stratified_accuracy"] - design_a["grouped_accuracy"]
design_b = pd.DataFrame(design_b_records)
design_b["gap"] = design_b["accuracy_with"] - design_b["accuracy_without"]


# %% Design A
print()
print("== 設計A: 分割方法を変える（seedごと、891行のOOF正解率）==")
print(
    design_a[
        [
            "arm",
            "seed",
            "stratified_accuracy",
            "grouped_accuracy",
            "inflation",
            "stratified_exposure",
            "grouped_exposure",
        ]
    ]
    .round(4)
    .to_string(index=False)
)
print()
print("腕ごとの上振れ:")
print(
    design_a.groupby("arm")["inflation"]
    .agg(["mean", "min", "max"])
    .round(4)
    .to_string()
)


# %% Design B
print()
print("== 設計B: 同じ分割の中で、仲間あり / なしを比べる ==")
print(design_b.round(4).to_string(index=False))
print()
print("腕ごとの差（仲間あり − 仲間なし）:")
print(design_b.groupby("arm")["gap"].agg(["mean", "min", "max"]).round(4).to_string())


# %% Do the two designs agree, and does inflation differ between arms
print()
print("== まとめ ==")
for arm_name in arms:
    a_value = design_a.loc[design_a["arm"] == arm_name, "inflation"].mean()
    b_value = design_b.loc[design_b["arm"] == arm_name, "gap"].mean()
    print(f"{arm_name:6s} 設計A {a_value:+.4f}   設計B {b_value:+.4f}")

arm_names = list(arms)
difference_between_arms = (
    design_a.loc[design_a["arm"] == arm_names[0], "inflation"].mean()
    - design_a.loc[design_a["arm"] == arm_names[1], "inflation"].mean()
)
print()
print(
    f"上振れの腕間の差（{arm_names[0]} − {arm_names[1]}）: "
    f"{difference_between_arms:+.4f}"
)
print("これが腕どうしの比較を歪めていた分。0に近ければ、引き算で消えている。")
