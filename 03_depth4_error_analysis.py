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
REPORT_DIR = PROJECT_DIR / "error_analysis"
REPORT_DIR.mkdir(exist_ok=True)


# %% Load data
train = pd.read_csv(DATA_DIR / "train.csv")


# %% Feature engineering
def add_baseline_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.copy()
    features["CabinKnown"] = features["Cabin"].notna().astype(int)
    features["FamilySize"] = features["SibSp"] + features["Parch"] + 1
    features["IsAlone"] = (features["FamilySize"] == 1).astype(int)
    return features


train_features = add_baseline_features(train)

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


# %% Validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)


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

model = DecisionTreeClassifier(max_depth=4, random_state=42)

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

errors = valid_analysis[valid_analysis["ErrorType"] != "correct"].copy()

print("model: DecisionTreeClassifier(max_depth=4, random_state=42)")
print("validation accuracy:", round(valid_accuracy, 4))
print("confusion matrix:")
print(valid_confusion)
print("error counts:")
print(valid_analysis["ErrorType"].value_counts())


# %% Error attribute summaries
def print_group_summary(frame: pd.DataFrame, label: str) -> None:
    print(f"\n## {label}")
    print("count:", len(frame))
    if frame.empty:
        return
    print("by Sex:")
    print(frame["Sex"].value_counts(dropna=False))
    print("by Pclass:")
    print(frame["Pclass"].value_counts(dropna=False).sort_index())
    print("by Embarked:")
    print(frame["Embarked"].value_counts(dropna=False))
    print("CabinKnown:")
    print(frame["CabinKnown"].value_counts(dropna=False).sort_index())
    print("FamilySize:")
    print(frame["FamilySize"].value_counts(dropna=False).sort_index())
    print("Age summary:")
    print(frame["Age"].describe())
    print("Fare summary:")
    print(frame["Fare"].describe())


missed_survivors = valid_analysis[valid_analysis["ErrorType"] == "missed_survivor"].copy()
false_survivors = valid_analysis[valid_analysis["ErrorType"] == "false_survivor"].copy()

print_group_summary(missed_survivors, "missed_survivor")
print_group_summary(false_survivors, "false_survivor")


# %% Save error rows
error_columns = [
    "PassengerId",
    "Survived",
    "Predicted",
    "ErrorType",
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
errors_path = REPORT_DIR / "depth4_validation_errors.csv"
errors[error_columns].sort_values(["ErrorType", "Sex", "Pclass", "Age"]).to_csv(
    errors_path,
    index=False,
)
print("\nsaved errors:", errors_path)


# %% Inspect tree
feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
tree_text = export_text(
    pipeline.named_steps["model"],
    feature_names=list(feature_names),
)
print("\nTree:")
print(tree_text)
