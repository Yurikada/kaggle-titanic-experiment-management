# %% Imports
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# %% Comparison contract
# 目的     : 同じ船の未知乗客に対する性能
# 固定条件 : 決定木、層化5-fold、accuracy、下記のベース7列
# 変える条件: Cabin の表現のみ（落とす / 有無フラグ / デッキ）
# 深さ     : 頑健性の軸。選択には使わない
# 採用条件 : 現行(有無フラグ)との対応のあるペア差が
#            95%区間で0を含まず、かつ平均 +0.015 以上。これを6深さ中5以上で満たす
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

TARGET_COLUMN = "Survived"
BASE_FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
DEPTHS = [3, 4, 5, 6, 7, 8]
SEEDS = [0, 1, 2, 3, 42]
N_SPLITS = 5
TREE_RANDOM_STATE = 42
REFERENCE_ARM = "有無フラグ"
MINIMUM_GAIN = 0.015
REQUIRED_DEPTHS = 5
N_BOOTSTRAP = 4000


# %% Load
train = pd.read_csv(DATA_DIR / "train.csv")
assert train.shape == (891, 12)
y = train[TARGET_COLUMN]


# %% Three representations of Cabin
def make_arms(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    base = frame[BASE_FEATURES].copy()

    with_flag = base.copy()
    with_flag["CabinKnown"] = frame["Cabin"].notna().astype(int)

    with_deck = base.copy()
    # 欠損は捨てずに "Missing" という水準として残す。
    # この水準が有無の情報を含むため、CabinKnown は入れない。
    with_deck["Deck"] = frame["Cabin"].str[0].fillna("Missing")

    return {"落とす": base, "有無フラグ": with_flag, "デッキ": with_deck}


arms = make_arms(train)
for name, features in arms.items():
    print(f"{name:6s}: {features.shape[1]}列 {list(features.columns)}")


# %% Pipeline: preprocessing is fitted inside each training fold
def build_pipeline(features: pd.DataFrame, max_depth: int) -> Pipeline:
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
            DecisionTreeClassifier(
                max_depth=max_depth,
                random_state=TREE_RANDOM_STATE,
            ),
        ),
    ])


# %% Run: the same folds are reused by every arm and depth
records = []
for seed in SEEDS:
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold_number, (train_index, valid_index) in enumerate(
        splitter.split(train, y),
        start=1,
    ):
        y_fold_train = y.iloc[train_index]
        y_fold_valid = y.iloc[valid_index]

        for arm_name, features in arms.items():
            X_fold_train = features.iloc[train_index]
            X_fold_valid = features.iloc[valid_index]

            for depth in DEPTHS:
                pipeline = build_pipeline(features, depth)
                pipeline.fit(X_fold_train, y_fold_train)

                valid_prediction = pipeline.predict(X_fold_valid)
                true_negative, false_positive, false_negative, true_positive = (
                    confusion_matrix(y_fold_valid, valid_prediction, labels=[0, 1]).ravel()
                )

                records.append({
                    "seed": seed,
                    "fold": fold_number,
                    "arm": arm_name,
                    "depth": depth,
                    "train_accuracy": accuracy_score(
                        y_fold_train,
                        pipeline.predict(X_fold_train),
                    ),
                    "validation_accuracy": accuracy_score(y_fold_valid, valid_prediction),
                    "true_positive": int(true_positive),
                    "true_negative": int(true_negative),
                    "false_positive": int(false_positive),
                    "false_negative": int(false_negative),
                })

results = pd.DataFrame(records)
assert len(results) == len(SEEDS) * N_SPLITS * len(arms) * len(DEPTHS)


# %% All 18 cells, reported together
cell_summary = (
    results.groupby(["depth", "arm"])
    .agg(
        train_mean=("train_accuracy", "mean"),
        validation_mean=("validation_accuracy", "mean"),
        validation_std=("validation_accuracy", "std"),
    )
    .reset_index()
)
cell_summary["train_minus_validation"] = (
    cell_summary["train_mean"] - cell_summary["validation_mean"]
)

print()
print("== 全18セル ==")
print(
    cell_summary.pivot(index="depth", columns="arm", values="validation_mean")
    .round(4)
    .to_string()
)
print()
print("== 訓練と検証の差 ==")
print(
    cell_summary.pivot(index="depth", columns="arm", values="train_minus_validation")
    .round(4)
    .to_string()
)


# %% Paired differences against the reference arm, fold by fold
# (seed, fold, depth, arm) は1行ずつしか無いはずなので pivot を使う。
# pivot_table は重複を黙って平均するため、想定外の重複が入ると
# 25個のペア差が1個へ潰れ、幅0の区間が「0を含まない」と判定されてしまう。
paired = results.pivot(
    index=["seed", "fold", "depth"],
    columns="arm",
    values="validation_accuracy",
)
EXPECTED_FOLDS = len(SEEDS) * N_SPLITS

rng = np.random.default_rng(TREE_RANDOM_STATE)
comparison_records = []
for arm_name in arms:
    if arm_name == REFERENCE_ARM:
        continue
    for depth in DEPTHS:
        at_depth = paired.xs(depth, level="depth")
        differences = (at_depth[arm_name] - at_depth[REFERENCE_ARM]).to_numpy()
        assert differences.size == EXPECTED_FOLDS, (
            f"{arm_name} depth={depth}: ペア差が {differences.size} 個しかない"
        )

        resample_index = rng.integers(0, differences.size, size=(N_BOOTSTRAP, differences.size))
        bootstrap_means = differences[resample_index].mean(axis=1)
        ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])

        excludes_zero = not (ci_low < 0 < ci_high)
        meets_gain = differences.mean() >= MINIMUM_GAIN

        comparison_records.append({
            "comparison": f"{arm_name} - {REFERENCE_ARM}",
            "depth": depth,
            "folds": differences.size,
            "mean_difference": differences.mean(),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "excludes_zero": excludes_zero,
            "meets_gain": meets_gain,
            "adopt": excludes_zero and meets_gain,
        })

comparisons = pd.DataFrame(comparison_records)
print()
print("== 現行との対応のあるペア差 ==")
print(comparisons.round(4).to_string(index=False))


# %% Decision rule
print()
print("== 採用判定 ==")
for comparison, group in comparisons.groupby("comparison"):
    satisfied = int(group["adopt"].sum())
    verdict = "採用" if satisfied >= REQUIRED_DEPTHS else "採用しない"
    print(
        f"{comparison}: 条件を満たした深さ {satisfied}/{len(DEPTHS)} "
        f"（基準 {REQUIRED_DEPTHS}/{len(DEPTHS)}） -> {verdict}"
    )


# %% Secondary: error direction, not used for the decision
print()
print("== 副指標: 誤りの内訳（採否には使わない、深さ合算） ==")
error_summary = (
    results.groupby("arm")[
        ["true_positive", "true_negative", "false_positive", "false_negative"]
    ]
    .sum()
    .assign(
        missed_survivor=lambda frame: frame["false_negative"],
        false_survivor=lambda frame: frame["false_positive"],
    )[["missed_survivor", "false_survivor"]]
)
print(error_summary.to_string())
