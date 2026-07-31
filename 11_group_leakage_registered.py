# %% Imports
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# %% Registered contract (fixed before running)
# 目的:
#   設計A（分割方法を変える）で、腕「落とす」の上振れについて
#   平均 - 2SE > 0 が成り立つかを確定させる。
# Seed:
#   探索で使った 0, 1, 2, 3, 42 は判定に用いない。
#   追加 seed は 5 から 29 の25個。番号は結果を見る前に固定した。
# 判定:
#   追加25 seedのみで判定する。30 seed全体は参考値として併記する。
#   25 seed完走前の途中確認による早期停止は行わない。
#   平均 - 2SE > 0 なら「漏洩による上振れを検出」と認定する。
#   平均 - 2SE <= 0 なら「この設計の検出下限を超える上振れは検出されなかった」と記録する。
#   効果が0であることの証明とはしない。片側の判定であり、負側は検出対象に含めない。
# 判定対象:
#   腕「落とす」の設計Aのみ。腕「有無フラグ」と設計Bは記述のみで判定しない。
# 事前に見積もった検出下限:
#   探索5 seedのSEは 0.0039 だったので、25 seedでは 2SE = 0.0035 前後と予想した。
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

TARGET_COLUMN = "Survived"
BASE_FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
DEPTH = 3
EXPLORATORY_SEEDS = [0, 1, 2, 3, 42]
REGISTERED_SEEDS = list(range(5, 30))
N_SPLITS = 5
TREE_RANDOM_STATE = 42
JUDGED_ARM = "落とす"

assert len(REGISTERED_SEEDS) == 25
assert not set(REGISTERED_SEEDS) & set(EXPLORATORY_SEEDS)


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


def run_scheme(features: pd.DataFrame, splitter, use_groups: bool) -> pd.DataFrame:
    rows = []
    split_arguments = (train, y, groups) if use_groups else (train, y)
    for _, (train_index, valid_index) in enumerate(splitter.split(*split_arguments)):
        pipeline = build_pipeline(features)
        pipeline.fit(features.iloc[train_index], y.iloc[train_index])
        prediction = pipeline.predict(features.iloc[valid_index])
        training_groups = set(groups.iloc[train_index])
        rows.append(pd.DataFrame({
            "correct": (prediction == y.iloc[valid_index].to_numpy()).astype(int),
            "has_relative_in_train": groups.iloc[valid_index]
            .isin(training_groups)
            .to_numpy()
            .astype(int),
        }))
    return pd.concat(rows, ignore_index=True)


# %% Run all 30 seeds, tagged by whether they belong to the registered set
records = []
for arm_name, features in arms.items():
    for seed in EXPLORATORY_SEEDS + REGISTERED_SEEDS:
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
        assert grouped["has_relative_in_train"].sum() == 0

        with_relative = stratified[stratified["has_relative_in_train"] == 1]
        without_relative = stratified[stratified["has_relative_in_train"] == 0]

        records.append({
            "arm": arm_name,
            "seed": seed,
            "registered": seed in REGISTERED_SEEDS,
            "design_a_inflation": stratified["correct"].mean() - grouped["correct"].mean(),
            "design_b_gap": with_relative["correct"].mean() - without_relative["correct"].mean(),
        })

results = pd.DataFrame(records)
assert len(results) == len(arms) * (len(EXPLORATORY_SEEDS) + len(REGISTERED_SEEDS))


# %% Registered judgement
def summarise(values: np.ndarray) -> dict:
    mean = values.mean()
    standard_error = values.std(ddof=1) / np.sqrt(values.size)
    return {
        "n_seeds": values.size,
        "mean": mean,
        "se": standard_error,
        "mean_minus_2se": mean - 2 * standard_error,
        "detection_limit_2se": 2 * standard_error,
        "positive_seeds": int((values > 0).sum()),
    }


judged = results[(results["arm"] == JUDGED_ARM) & results["registered"]]
judgement = summarise(judged["design_a_inflation"].to_numpy())

print("== 登録された判定: 設計A / 腕「落とす」/ 追加25 seed のみ ==")
for key, value in judgement.items():
    print(f"  {key:22s} {value:.5f}" if isinstance(value, float) else f"  {key:22s} {value}")

detected = judgement["mean_minus_2se"] > 0
print()
if detected:
    print("  判定: 平均 - 2SE > 0 -> 漏洩による上振れを検出")
else:
    print("  判定: 平均 - 2SE <= 0 -> この設計の検出下限を超える上振れは検出されなかった")
    print(f"        検出下限は 2SE = {judgement['detection_limit_2se']:.5f}。")
    print("        これ未満の効果の有無は判定していない。効果が0であることの証明ではない。")


# %% Reference values, not part of the judgement
print()
print("== 参考: 30 seed全体（判定には使わない）==")
reference = summarise(
    results[results["arm"] == JUDGED_ARM]["design_a_inflation"].to_numpy()
)
print(
    f"  n={reference['n_seeds']} 平均 {reference['mean']:+.5f} "
    f"SE {reference['se']:.5f} 平均-2SE {reference['mean_minus_2se']:+.5f}"
)

print()
print("== 記述のみ: 判定しない値 ==")
for arm_name in arms:
    for design, column in [("設計A", "design_a_inflation"), ("設計B", "design_b_gap")]:
        subset = results[(results["arm"] == arm_name) & results["registered"]]
        stats = summarise(subset[column].to_numpy())
        mark = "  <- 登録判定対象" if (arm_name == JUDGED_ARM and design == "設計A") else ""
        print(
            f"  {arm_name:6s} {design} 平均 {stats['mean']:+.5f} "
            f"SE {stats['se']:.5f} 平均-2SE {stats['mean_minus_2se']:+.5f} "
            f"正の符号 {stats['positive_seeds']}/25{mark}"
        )

arm_names = list(arms)
registered_only = results[results["registered"]]
arm_gap = (
    registered_only[registered_only["arm"] == arm_names[0]]["design_a_inflation"].mean()
    - registered_only[registered_only["arm"] == arm_names[1]]["design_a_inflation"].mean()
)
print()
print(f"  上振れの腕間の差（{arm_names[0]} − {arm_names[1]}、設計A、25 seed）: {arm_gap:+.5f}")
print("  腕どうしの比較で引き算しても残る歪みの大きさ。")
