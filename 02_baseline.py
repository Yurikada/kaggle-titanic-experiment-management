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


# %% Load data
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")


# %% Feature engineering
def add_baseline_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.copy()
    features["CabinKnown"] = features["Cabin"].notna().astype(int)
    return features


train_features = add_baseline_features(train)
test_features = add_baseline_features(test)

feature_columns = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "CabinKnown",
]
target_column = "Survived"

X = train_features[feature_columns]
y = train_features[target_column]
X_test = test_features[feature_columns]


# %% Validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("train split:", X_train.shape, y_train.value_counts(normalize=True).sort_index().to_dict())
print("valid split:", X_valid.shape, y_valid.value_counts(normalize=True).sort_index().to_dict())


# %% Pipeline
numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "CabinKnown"]
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

model = DecisionTreeClassifier(max_depth=3, random_state=42)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", model),
    ]
)


# %% Fit and validate
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

print("validation accuracy:", round(valid_accuracy, 4))
print("confusion matrix:")
print(valid_confusion)


# %% Error analysis: missed survivors
missed_survivors = valid_analysis[valid_analysis["ErrorType"] == "missed_survivor"].copy()
missed_survivors["FamilySize"] = missed_survivors["SibSp"] + missed_survivors["Parch"] + 1
missed_survivors["IsAlone"] = (missed_survivors["FamilySize"] == 1).astype(int)

print("missed survivors:", len(missed_survivors))
print("by Sex:")
print(missed_survivors["Sex"].value_counts(dropna=False))
print("by Pclass:")
print(missed_survivors["Pclass"].value_counts(dropna=False).sort_index())
print("by Embarked:")
print(missed_survivors["Embarked"].value_counts(dropna=False))
print("CabinKnown:")
print(missed_survivors["CabinKnown"].value_counts(dropna=False).sort_index())
print("Age summary:")
print(missed_survivors["Age"].describe())
print("Fare summary:")
print(missed_survivors["Fare"].describe())
print("FamilySize:")
print(missed_survivors["FamilySize"].value_counts(dropna=False).sort_index())

missed_columns = [
    "PassengerId",
    "Survived",
    "Predicted",
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "FamilySize",
    "IsAlone",
    "Fare",
    "Cabin",
    "CabinKnown",
    "Embarked",
    "Ticket",
]
print("missed survivor sample:")
print(missed_survivors[missed_columns].sort_values(["Sex", "Pclass", "Age"]).to_string(index=False))


# %% Inspect tree
feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
tree_text = export_text(
    pipeline.named_steps["model"],
    feature_names=list(feature_names),
)
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
submission_path = SUBMISSION_DIR / "decision_tree_depth3_baseline.csv"
submission.to_csv(submission_path, index=False)
print("saved:", submission_path)
print(submission.head())
