# %% Imports
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text


# %% Paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submissions"
SUBMISSION_DIR.mkdir(exist_ok=True)


# %% Constants
INFANT_MAX_AGE = 7
ELDERLY_MIN_AGE = 55
MAX_DEPTH = 5
VALID_RANDOM_STATE = 42
MODEL_RANDOM_STATE = 42


# %% Load data
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")


# %% Feature engineering
def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.copy()
    features["CabinKnown"] = features["Cabin"].notna().astype(int)
    features["FamilySize"] = features["SibSp"] + features["Parch"] + 1
    features["IsInfant"] = (features["Age"] <= INFANT_MAX_AGE).astype(int)
    features["IsElderly"] = (features["Age"] >= ELDERLY_MIN_AGE).astype(int)
    return features


train_features = add_features(train)
test_features = add_features(test)

feature_columns = [
    "Pclass",
    "Sex",
    "Age",
    "FamilySize",
    "Fare",
    "Embarked",
    "CabinKnown",
    "IsInfant",
    "IsElderly",
]
target_column = "Survived"

X = train_features[feature_columns]
y = train_features[target_column]
X_test = test_features[feature_columns]


# %% Pipeline
numeric_features = [
    "Pclass",
    "Age",
    "FamilySize",
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

model = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=MODEL_RANDOM_STATE)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", model),
    ]
)


# %% Validation check
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=VALID_RANDOM_STATE,
)

pipeline.fit(X_train, y_train)
valid_pred = pipeline.predict(X_valid)
valid_accuracy = accuracy_score(y_valid, valid_pred)
valid_confusion = confusion_matrix(y_valid, valid_pred)

valid_analysis = train_features.loc[X_valid.index].copy()
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
captured_male_survivors = male_survivors[male_survivors["ErrorType"] == "correct"]

print("model: DecisionTreeClassifier(max_depth=5, random_state=42)")
print("feature change: SibSp and Parch replaced by FamilySize")
print("feature columns:", feature_columns)
print("validation random_state:", VALID_RANDOM_STATE)
print("validation accuracy:", round(valid_accuracy, 4))
print("confusion matrix:")
print(valid_confusion)
print("male survivor count:", len(male_survivors))
print("missed male survivor count:", len(missed_male_survivors))
print("captured male survivor count:", len(captured_male_survivors))
print(
    "missed male survivor rate:",
    round(len(missed_male_survivors) / len(male_survivors), 4)
    if len(male_survivors) > 0
    else pd.NA,
)

feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
tree_text = export_text(
    pipeline.named_steps["model"],
    feature_names=list(feature_names),
)
print("\nTree:")
print(tree_text)


# %% Fit full train and create submission
pipeline.fit(X, y)
test_pred = pipeline.predict(X_test)

submission = pd.DataFrame(
    {
        "PassengerId": test["PassengerId"],
        "Survived": test_pred.astype(int),
    }
)

submission_path = SUBMISSION_DIR / "decision_tree_depth5_family_size_replaced.csv"
submission.to_csv(submission_path, index=False)

print("\nsaved:", submission_path)
print(submission.head())
