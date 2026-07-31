# %% Imports
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text


# %% What this file is
# Experiment 2026-07-31-01 で、`Cabin` を落とした腕は6つの深さすべてで現行を上回ったが
# （+0.0038〜+0.0103）、登録した実質閾値 +0.015 に届かず不採用とした。
# 1-SE型の基準であれば採用に反転していた構成でもある。
#
# このスクリプトはその構成を提出物として生成する。採用したわけではない。
# 判定ルールの違いが公開スコアにどう出るかを測るための提出であり、
# 結果によって Experiment -01 の採否判定を書き換えない。
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submissions"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGET_COLUMN = "Survived"
ID_COLUMN = "PassengerId"
DEPTH = 3
TREE_RANDOM_STATE = 42

# 現行構成から CabinKnown を外しただけ。他の条件は動かさない。
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]


# %% Load
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
assert train.shape == (891, 12)
assert test.shape == (418, 11)

X = train[FEATURES]
y = train[TARGET_COLUMN]
X_test = test[FEATURES]


# %% Fit on all training rows
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
    ("model", DecisionTreeClassifier(max_depth=DEPTH, random_state=TREE_RANDOM_STATE)),
])
pipeline.fit(X, y)
predictions = pipeline.predict(X_test).astype(int)

submission = pd.DataFrame({ID_COLUMN: test[ID_COLUMN], TARGET_COLUMN: predictions})
submission_path = SUBMISSION_DIR / "submission_drop_cabin_depth3.csv"
submission.to_csv(submission_path, index=False)


# %% Checks
assert list(submission.columns) == [ID_COLUMN, TARGET_COLUMN]
assert len(submission) == 418
assert submission[ID_COLUMN].is_unique
assert submission[ID_COLUMN].tolist() == test[ID_COLUMN].tolist()
assert submission[TARGET_COLUMN].notna().all()
assert set(submission[TARGET_COLUMN].unique()).issubset({0, 1})

print("saved:", submission_path)
print("rows:", len(submission))
print(
    "predicted survivors:",
    int(submission[TARGET_COLUMN].sum()),
    f"({submission[TARGET_COLUMN].mean():.4f})",
)
print("leaves:", pipeline.named_steps["model"].get_n_leaves())

baseline_path = SUBMISSION_DIR / "decision_tree_depth3_baseline.csv"
if baseline_path.exists():
    baseline = pd.read_csv(baseline_path)
    differing = int((baseline[TARGET_COLUMN] != submission[TARGET_COLUMN]).sum())
    print()
    print(f"現行構成の提出物と異なる行: {differing} / {len(submission)}")

print()
print(export_text(
    pipeline.named_steps["model"],
    feature_names=list(pipeline.named_steps["preprocess"].get_feature_names_out()),
))
