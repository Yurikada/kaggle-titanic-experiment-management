# %% Imports
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# %% Paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
REPORT_DIR = PROJECT_DIR / "error_analysis"
REPORT_DIR.mkdir(exist_ok=True)


# %% Constants
INFANT_MAX_AGE = 7
ELDERLY_MIN_AGE = 55
RANDOM_STATES = [42, 7, 2026]


# %% Load data
train = pd.read_csv(DATA_DIR / "train.csv")


# %% Feature engineering
def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.copy()
    features["CabinKnown"] = features["Cabin"].notna().astype(int)
    features["FamilySize"] = features["SibSp"] + features["Parch"] + 1
    features["IsAlone"] = (features["FamilySize"] == 1).astype(int)
    features["IsInfant"] = (features["Age"] <= INFANT_MAX_AGE).astype(int)
    features["IsElderly"] = (features["Age"] >= ELDERLY_MIN_AGE).astype(int)
    return features


def build_pipeline() -> Pipeline:
    numeric_features = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "CabinKnown",
        "IsInfant",
        "IsElderly",
    ]
    categorical_features = ["Sex", "Embarked"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = DecisionTreeClassifier(max_depth=4, random_state=42)

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )


train_features = add_features(train)

feature_columns = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "CabinKnown",
    "IsInfant",
    "IsElderly",
]
target_column = "Survived"

X = train_features[feature_columns]
y = train_features[target_column]


# %% Run repeated splits
summary_rows = []
error_rows = []

for random_state in RANDOM_STATES:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    valid_pred = pipeline.predict(X_valid)

    valid_analysis = train_features.loc[X_valid.index].copy()
    valid_analysis["RandomState"] = random_state
    valid_analysis["Predicted"] = valid_pred
    valid_analysis["ErrorType"] = "correct"
    valid_analysis.loc[
        (valid_analysis[target_column] == 1) & (valid_analysis["Predicted"] == 0),
        "ErrorType",
    ] = "missed_survivor"
    valid_analysis.loc[
        (valid_analysis[target_column] == 0) & (valid_analysis["Predicted"] == 1),
        "ErrorType",
    ] = "false_survivor"

    male_survivors = valid_analysis[
        (valid_analysis["Sex"] == "male") & (valid_analysis[target_column] == 1)
    ]
    missed_male_survivors = male_survivors[
        male_survivors["ErrorType"] == "missed_survivor"
    ]
    captured_male_survivors = male_survivors[
        male_survivors["ErrorType"] == "correct"
    ]

    confusion = confusion_matrix(y_valid, valid_pred)
    summary_rows.append(
        {
            "random_state": random_state,
            "validation_accuracy": accuracy_score(y_valid, valid_pred),
            "true0_pred0": int(confusion[0, 0]),
            "true0_pred1": int(confusion[0, 1]),
            "true1_pred0": int(confusion[1, 0]),
            "true1_pred1": int(confusion[1, 1]),
            "male_survivor_count": len(male_survivors),
            "missed_male_survivor_count": len(missed_male_survivors),
            "captured_male_survivor_count": len(captured_male_survivors),
            "missed_male_survivor_rate": (
                len(missed_male_survivors) / len(male_survivors)
                if len(male_survivors) > 0
                else pd.NA
            ),
        }
    )

    error_rows.append(valid_analysis[valid_analysis["ErrorType"] != "correct"].copy())


summary = pd.DataFrame(summary_rows)
errors = pd.concat(error_rows, ignore_index=True)


# %% Print and save
print("model: DecisionTreeClassifier(max_depth=4, random_state=42)")
print("validation split: 80:20 stratified split")
print("random states:", RANDOM_STATES)
print("\nsummary:")
print(summary.round(4).to_string(index=False))

print("\nsummary statistics:")
print(
    summary[
        [
            "validation_accuracy",
            "male_survivor_count",
            "missed_male_survivor_count",
            "captured_male_survivor_count",
            "missed_male_survivor_rate",
        ]
    ]
    .describe()
    .round(4)
)

summary_path = REPORT_DIR / "depth4_age_split_multi_split_summary.csv"
errors_path = REPORT_DIR / "depth4_age_split_multi_split_errors.csv"
summary.to_csv(summary_path, index=False)
errors.to_csv(errors_path, index=False)

print("\nsaved summary:", summary_path)
print("saved errors:", errors_path)
