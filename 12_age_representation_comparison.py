# %% Imports
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# %% Registered contract (fixed before running)
# 問い     : Age の欠損177件をどう扱うか
# ベース   : Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, CabinKnown
#            CabinKnown は Experiment 2026-07-31-01 の結論に従って残す
# 腕       : 腕4 Age落とす(10列) / 腕1 全体中央値(11列) /
#            腕3 群別中央値 [Pclass, Sex](11列) / 腕2 全体中央値+欠損フラグ(12列)
# 深さ     : 3 に固定。Experiment 2026-07-31-02 と条件を揃える。
#            複数深さに対する頑健性は本実験では未検証。
# 分割     : 層化5-fold x 20 seed = 100 fold。全腕が同一のfoldを使う。
# 指標     : accuracy
# 判定     : ペア差版の1-SEルール（今回の登録で差し替え）
#            各foldで最良腕とのペア差 delta = 最良腕 - 当該腕 を取り、
#            平均 delta <= 1.0 * SE(delta) を満たす腕を「最良と区別がつかない」とする。
#            その中で最も単純な腕を選ぶ。
# 単純さ   : 第1優先 One-Hot展開後の列数が最小。
#            第2優先 列数が同じ場合は前処理の段数が少ない方（腕1 < 腕3）。
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

TARGET_COLUMN = "Survived"
DEPTH = 3
SEEDS = list(range(20))
N_SPLITS = 5
TREE_RANDOM_STATE = 42
ONE_SE_MULTIPLIER = 1.0

assert len(SEEDS) == 20


# %% Load
train = pd.read_csv(DATA_DIR / "train.csv")
assert train.shape == (891, 12)
y = train[TARGET_COLUMN]


# %% Grouped median imputation, fitted inside each training fold
class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """[Pclass, Sex] ごとの中央値で Age を埋める。学習部分だけで中央値を推定する。"""

    def __init__(self, target_column="Age", group_columns=("Pclass", "Sex")):
        self.target_column = target_column
        self.group_columns = group_columns

    def fit(self, X, y=None):
        observed = X.dropna(subset=[self.target_column])
        self.group_medians_ = observed.groupby(
            list(self.group_columns)
        )[self.target_column].median()
        self.global_median_ = observed[self.target_column].median()
        return self

    def transform(self, X):
        frame = X.copy()
        missing = frame[self.target_column].isna()
        if missing.any():
            keys = pd.MultiIndex.from_frame(
                frame.loc[missing, list(self.group_columns)]
            )
            filled = self.group_medians_.reindex(keys).to_numpy()
            # 学習部分に現れなかった組み合わせは全体中央値へ落とす
            filled = np.where(np.isnan(filled), self.global_median_, filled)
            frame.loc[missing, self.target_column] = filled
        return frame


# %% Arms
BASE_FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "CabinKnown",
]

prepared = train.assign(CabinKnown=train["Cabin"].notna().astype(int))

arm_features = {
    "腕4 Age落とす": prepared[[c for c in BASE_FEATURES if c != "Age"]].copy(),
    "腕1 全体中央値": prepared[BASE_FEATURES].copy(),
    "腕3 群別中央値": prepared[BASE_FEATURES].copy(),
    "腕2 中央値+欠損フラグ": prepared[BASE_FEATURES].assign(
        AgeMissing=train["Age"].isna().astype(int)
    ),
}
# 前処理の段数。列数が同じときのタイブレークに使う。
arm_preprocessing_steps = {
    "腕4 Age落とす": 0,
    "腕1 全体中央値": 0,
    "腕3 群別中央値": 1,
    "腕2 中央値+欠損フラグ": 0,
}
USES_GROUP_IMPUTER = {"腕3 群別中央値"}


def build_pipeline(features: pd.DataFrame, arm_name: str) -> Pipeline:
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

    steps = []
    if arm_name in USES_GROUP_IMPUTER:
        steps.append(("group_median", GroupMedianImputer()))
    steps.append(("preprocess", preprocess))
    steps.append(
        ("model", DecisionTreeClassifier(max_depth=DEPTH, random_state=TREE_RANDOM_STATE))
    )
    return Pipeline(steps)


# %% Encoded width of each arm, recorded before the comparison
encoded_columns = {}
for arm_name, features in arm_features.items():
    fitted = build_pipeline(features, arm_name).named_steps["preprocess"]
    prepared_features = features
    if arm_name in USES_GROUP_IMPUTER:
        prepared_features = GroupMedianImputer().fit_transform(features)
    encoded_columns[arm_name] = fitted.fit_transform(prepared_features).shape[1]

print("== 腕の構成 ==")
for arm_name in arm_features:
    print(
        f"  {arm_name:14s} 元の列 {arm_features[arm_name].shape[1]:2d}  "
        f"One-Hot展開後 {encoded_columns[arm_name]:2d}  "
        f"前処理段数 {arm_preprocessing_steps[arm_name]}"
    )


# %% Run: identical folds for every arm
records = []
for seed in SEEDS:
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold_number, (train_index, valid_index) in enumerate(
        splitter.split(train, y),
        start=1,
    ):
        for arm_name, features in arm_features.items():
            pipeline = build_pipeline(features, arm_name)
            pipeline.fit(features.iloc[train_index], y.iloc[train_index])
            prediction = pipeline.predict(features.iloc[valid_index])
            records.append({
                "seed": seed,
                "fold": fold_number,
                "arm": arm_name,
                "accuracy": (prediction == y.iloc[valid_index].to_numpy()).mean(),
            })

results = pd.DataFrame(records)
assert len(results) == len(SEEDS) * N_SPLITS * len(arm_features)

by_fold = results.pivot(index=["seed", "fold"], columns="arm", values="accuracy")
by_fold = by_fold.reindex(columns=list(arm_features))
EXPECTED_FOLDS = len(SEEDS) * N_SPLITS
assert len(by_fold) == EXPECTED_FOLDS

print()
print("== 腕ごとの平均正解率（100 fold）==")
summary = pd.DataFrame({
    "mean": by_fold.mean(),
    "std": by_fold.std(),
    "encoded_columns": pd.Series(encoded_columns),
    "preprocessing_steps": pd.Series(arm_preprocessing_steps),
})
print(summary.round(4).to_string())


# %% Paired 1-SE rule
best_arm = summary["mean"].idxmax()
print()
print(f"最良腕: {best_arm}（平均 {summary.loc[best_arm, 'mean']:.4f}）")

rows = []
for arm_name in arm_features:
    delta = (by_fold[best_arm] - by_fold[arm_name]).to_numpy()
    assert delta.size == EXPECTED_FOLDS
    mean_delta = delta.mean()
    se_delta = delta.std(ddof=1) / np.sqrt(delta.size) if delta.std(ddof=1) > 0 else 0.0
    rows.append({
        "arm": arm_name,
        "mean_delta": mean_delta,
        "se_delta": se_delta,
        "threshold_1se": ONE_SE_MULTIPLIER * se_delta,
        "within_1se": mean_delta <= ONE_SE_MULTIPLIER * se_delta,
        "encoded_columns": encoded_columns[arm_name],
        "preprocessing_steps": arm_preprocessing_steps[arm_name],
    })

comparison = pd.DataFrame(rows)
print()
print("== 最良腕とのペア差 ==")
print(comparison.round(5).to_string(index=False))

candidates = comparison[comparison["within_1se"]].sort_values(
    ["encoded_columns", "preprocessing_steps"]
)
selected = candidates.iloc[0]

print()
print("== 選択 ==")
print(f"  1SE以内の腕: {', '.join(candidates['arm'])}")
print(
    f"  選択: {selected['arm']}"
    f"（{int(selected['encoded_columns'])}列、前処理段数 {int(selected['preprocessing_steps'])}）"
)
print("  複数深さに対する頑健性は本実験では未検証。")
