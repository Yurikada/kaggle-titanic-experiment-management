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


# %% Constants
INFANT_MAX_AGE = 7
ELDERLY_MIN_AGE = 55


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


# %% Validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)


# %% Pipeline
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

print(
    "model: DecisionTreeClassifier(max_depth=4, random_state=42) "
    "with IsInfant and IsElderly"
)
print("age split:")
print(f"  IsInfant: Age <= {INFANT_MAX_AGE}")
print(f"  IsElderly: Age >= {ELDERLY_MIN_AGE}")
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
    print("IsInfant:")
    print(frame["IsInfant"].value_counts(dropna=False).sort_index())
    print("IsElderly:")
    print(frame["IsElderly"].value_counts(dropna=False).sort_index())
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


# %% Quantify missed male survivor attribute skew
def build_attribute_skew_table(
    reference: pd.DataFrame,
    error_group: pd.DataFrame,
    comparison_group: pd.DataFrame,
    attributes: list[str],
) -> pd.DataFrame:
    rows = []
    for attribute in attributes:
        values = pd.Series(
            pd.concat(
                [
                    reference[attribute],
                    error_group[attribute],
                    comparison_group[attribute],
                ],
                ignore_index=True,
            )
            .dropna()
            .unique()
        ).sort_values()

        for value in values:
            reference_count = int((reference[attribute] == value).sum())
            error_count = int((error_group[attribute] == value).sum())
            comparison_count = int((comparison_group[attribute] == value).sum())

            error_without_attribute = len(error_group) - error_count
            comparison_without_attribute = len(comparison_group) - comparison_count

            reference_share = (
                reference_count / len(reference) if len(reference) > 0 else pd.NA
            )
            error_share = error_count / len(error_group) if len(error_group) > 0 else pd.NA
            comparison_share = (
                comparison_count / len(comparison_group)
                if len(comparison_group) > 0
                else pd.NA
            )
            proportion_diff = (
                error_share - reference_share
                if pd.notna(error_share) and pd.notna(reference_share)
                else pd.NA
            )
            lift = (
                error_share / reference_share
                if pd.notna(error_share)
                and pd.notna(reference_share)
                and reference_share != 0
                else pd.NA
            )

            if len(error_group) == 0 or len(comparison_group) == 0:
                odds_ratio_corrected = pd.NA
            else:
                # Haldane-Anscombe correction keeps the odds ratio finite for small groups.
                odds_ratio_corrected = (
                    (error_count + 0.5) * (comparison_without_attribute + 0.5)
                ) / ((error_without_attribute + 0.5) * (comparison_count + 0.5))

            rows.append(
                {
                    "attribute": attribute,
                    "value": value,
                    "reference_count": reference_count,
                    "error_count": error_count,
                    "comparison_count": comparison_count,
                    "reference_share": reference_share,
                    "error_share": error_share,
                    "comparison_share": comparison_share,
                    "proportion_diff": proportion_diff,
                    "lift": lift,
                    "odds_ratio_corrected": odds_ratio_corrected,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["proportion_diff", "lift"],
        ascending=[False, False],
        na_position="last",
    )


male_survivors = valid_analysis[
    (valid_analysis["Sex"] == "male") & (valid_analysis[target_column] == 1)
].copy()
missed_male_survivors = male_survivors[
    male_survivors["ErrorType"] == "missed_survivor"
].copy()
captured_male_survivors = male_survivors[
    male_survivors["ErrorType"] == "correct"
].copy()

skew_attributes = [
    "Pclass",
    "SibSp",
    "Parch",
    "FamilySize",
    "IsAlone",
    "Embarked",
    "CabinKnown",
    "IsInfant",
    "IsElderly",
]

male_survivor_skew = build_attribute_skew_table(
    reference=male_survivors,
    error_group=missed_male_survivors,
    comparison_group=captured_male_survivors,
    attributes=skew_attributes,
)

print("\n## missed male survivor attribute skew")
print("reference: validation male survivors")
print("error group: male survivors predicted as dead")
print("comparison group: male survivors predicted as survived")
print("reference count:", len(male_survivors))
print("error group count:", len(missed_male_survivors))
print("comparison group count:", len(captured_male_survivors))
if len(captured_male_survivors) == 0:
    print(
        "note: odds ratio is NA because there are no correctly predicted male "
        "survivors in this validation split."
    )
print(
    male_survivor_skew[
        [
            "attribute",
            "value",
            "reference_count",
            "error_count",
            "comparison_count",
            "reference_share",
            "error_share",
            "proportion_diff",
            "lift",
            "odds_ratio_corrected",
        ]
    ]
    .round(3)
    .to_string(index=False)
)


# %% Save error rows
error_columns = [
    "PassengerId",
    "Survived",
    "Predicted",
    "ErrorType",
    "Pclass",
    "Sex",
    "Age",
    "IsInfant",
    "IsElderly",
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
errors_path = REPORT_DIR / "depth4_age_split_validation_errors.csv"
errors[error_columns].sort_values(["ErrorType", "Sex", "Pclass", "Age"]).to_csv(
    errors_path,
    index=False,
)
print("\nsaved errors:", errors_path)

skew_path = REPORT_DIR / "depth4_age_split_missed_male_survivor_skew.csv"
male_survivor_skew.to_csv(skew_path, index=False)
print("saved missed male survivor skew:", skew_path)


# %% Inspect tree
feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
tree_text = export_text(
    pipeline.named_steps["model"],
    feature_names=list(feature_names),
)
print("\nTree:")
print(tree_text)
