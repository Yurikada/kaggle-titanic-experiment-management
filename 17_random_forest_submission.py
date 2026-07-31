# %% Imports
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# %% What this file is
# Experiment 2026-07-31-05 でランダムフォレストが採用された。
# 決定木との差は 0.00712、ペア差のSE 0.00161 の 4.4 倍で、1SE の帯の外だった。
# 事前登録した解釈可能性の宣言（1SE以内なら決定木を維持）に照らして族を変更する。
#
# 最終モデルのハイパーパラメータは、train全体に対して同じ内側3-foldで選び直す。
# 入れ子CVの外側は性能推定のためのもので、採用後の設定選択には使わない。
#
# 探索範囲の変更（結果を見た後の判断であることを明記する）:
#   比較時の grid は max_depth in {3, 4, 5} で、内側CVは上端の 5 を 65% の fold で選んだ。
#   最適値が範囲外にある可能性が高いため、最終モデルの選択に限り上へ広げる。
#   比較の判定そのものは狭い grid のまま確定しており、ここで作り直していない。
#   広げたことで RF が有利になる方向にしか動かないため、採用の結論は脅かされない。
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submissions"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGET_COLUMN = "Survived"
ID_COLUMN = "PassengerId"
RANDOM_STATE = 42
N_INNER_SPLITS = 3

FEATURES = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "CabinKnown",
]
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare", "CabinKnown"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]

# 比較時: max_depth in {3, 4, 5}。最終モデルでは上へ広げる。
WIDENED_GRID = {
    "model__max_depth": [3, 4, 5, 6, 8, 10, None],
    "model__max_features": ["sqrt", "log2"],
}
NUMERIC_DEPTHS = [d for d in WIDENED_GRID["model__max_depth"] if d is not None]


# %% Load
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
assert train.shape == (891, 12)
assert test.shape == (418, 11)

train = train.assign(CabinKnown=train["Cabin"].notna().astype(int))
test_prepared = test.assign(CabinKnown=test["Cabin"].notna().astype(int))
X = train[FEATURES]
y = train[TARGET_COLUMN]
X_test = test_prepared[FEATURES]


# %% Select hyperparameters on the full training set
pipeline = Pipeline([
    ("preprocess", ColumnTransformer([
        (
            "numeric",
            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            CATEGORICAL_FEATURES,
        ),
    ])),
    (
        "model",
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    ),
])

search = GridSearchCV(
    estimator=pipeline,
    param_grid=WIDENED_GRID,
    scoring="accuracy",
    cv=StratifiedKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE),
    n_jobs=1,
    refit=True,
    error_score="raise",
)
search.fit(X, y)

print("選ばれた設定:", search.best_params_)
print(f"内側CVスコア: {search.best_score_:.4f}")

chosen_depth = search.best_params_["model__max_depth"]
if chosen_depth is not None and chosen_depth == max(NUMERIC_DEPTHS):
    print("注意: 広げた grid でも数値上端が選ばれた。さらに外に最適値がある可能性がある。")
elif chosen_depth is None:
    print("注意: 制限なしが選ばれた。深さによる制約は効いていない。")

scores = pd.DataFrame(search.cv_results_)[
    ["param_model__max_depth", "param_model__max_features", "mean_test_score"]
].sort_values("mean_test_score", ascending=False)
print()
print("内側CVの全候補:")
print(scores.round(4).to_string(index=False))


# %% Predict and save
predictions = search.predict(X_test).astype(int)
submission = pd.DataFrame({ID_COLUMN: test[ID_COLUMN], TARGET_COLUMN: predictions})
submission_path = SUBMISSION_DIR / "submission_random_forest.csv"
submission.to_csv(submission_path, index=False)

assert list(submission.columns) == [ID_COLUMN, TARGET_COLUMN]
assert len(submission) == 418
assert submission[ID_COLUMN].is_unique
assert submission[ID_COLUMN].tolist() == test[ID_COLUMN].tolist()
assert submission[TARGET_COLUMN].notna().all()
assert set(submission[TARGET_COLUMN].unique()).issubset({0, 1})

print()
print("saved:", submission_path)
print(
    "predicted survivors:",
    int(submission[TARGET_COLUMN].sum()),
    f"({submission[TARGET_COLUMN].mean():.4f})",
)

baseline = pd.read_csv(SUBMISSION_DIR / "decision_tree_depth3_baseline.csv")
differing = int((baseline[TARGET_COLUMN] != submission[TARGET_COLUMN]).sum())
print(f"深さ3の提出物と異なる行: {differing} / {len(submission)}")
print(f"公開側に入る期待値: 約 {differing / 2:.0f} 行 = 正解率 {differing / 2 / 209:.3f} 相当")


# %% What replaces the readable tree
# 森は個々の予測根拠を追えない。重要度は「どの列が分割に使われたか」までしか示さない。
model = search.best_estimator_.named_steps["model"]
names = search.best_estimator_.named_steps["preprocess"].get_feature_names_out()
importance = (
    pd.DataFrame({"feature": names, "importance": model.feature_importances_})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
print()
print("特徴量重要度（個々の予測根拠は追えない点に注意）:")
print(importance.round(4).to_string(index=False))
