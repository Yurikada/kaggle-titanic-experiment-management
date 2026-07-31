# %% Imports
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# %% What this file is
# Experiment 2026-07-31-04 で、深さ5はペア差 +0.00561、SE 0.00166 の 3.4 倍で
# 1-SE の帯からはっきり外れた。深さ3が選ばれ、深さ5は不採用である。
#
# このスクリプトはその不採用の構成を提出物として生成する。採用したわけではない。
# 「登録した基準が明確に落とした構成は、公開スコアでも落ちるか」を測るための提出であり、
# 結果によって Experiment -04 の採否判定を書き換えない。
#
# 深さ3の提出物との予測差は418行中40行で、公開側にはおよそ20行入る。
# 正解率にして 0.096 相当となり、公開スコアの標準誤差 0.029 を超える。
# 深さ4は9行（公開側4.5行、0.022相当）で標準誤差と同程度のため選ばなかった。
# `Cabin` を落とす構成は2行しか違わず、測定として成立しない。
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submissions"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGET_COLUMN = "Survived"
ID_COLUMN = "PassengerId"
DEPTH = 5
TREE_RANDOM_STATE = 42

FEATURES = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "CabinKnown",
]
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare", "CabinKnown"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]


# %% Load
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
assert train.shape == (891, 12)
assert test.shape == (418, 11)

train = train.assign(CabinKnown=train["Cabin"].notna().astype(int))
test_features = test.assign(CabinKnown=test["Cabin"].notna().astype(int))

X = train[FEATURES]
y = train[TARGET_COLUMN]
X_test = test_features[FEATURES]


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
submission_path = SUBMISSION_DIR / "submission_depth5_current_features.csv"
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

baseline = pd.read_csv(SUBMISSION_DIR / "decision_tree_depth3_baseline.csv")
differing = int((baseline[TARGET_COLUMN] != submission[TARGET_COLUMN]).sum())
print()
print(f"深さ3の提出物と異なる行: {differing} / {len(submission)}")
print(f"公開側に入る期待値: 約 {differing / 2:.0f} 行 = 正解率 {differing / 2 / 209:.3f} 相当")
