# %% Imports
from pathlib import Path

import pandas as pd


# %% Paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submissions"


# %% Load data
train_path = DATA_DIR / "train.csv"
test_path = DATA_DIR / "test.csv"
sample_path = DATA_DIR / "gender_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)


# %% Basic shape and preview
print("train:", train.shape)
print("test:", test.shape)
print("sample_submission:", sample_submission.shape)

print(train.head())
print(test.head())
print(sample_submission.head())


# %% Schema and missing values
print(train.info())
print(test.info())

missing_summary = pd.DataFrame(
    {
        "train_missing": train.isna().sum(),
        "test_missing": test.isna().sum(),
    }
)
print(missing_summary)


# %% Target distribution
target_distribution = train["Survived"].value_counts(normalize=True).sort_index()
print(target_distribution)


# %% Simple group survival rates
sex_survival = train.groupby("Sex")["Survived"].mean().sort_values(ascending=False)
pclass_survival = train.groupby("Pclass")["Survived"].mean().sort_index()
sex_pclass_survival = train.pivot_table(
    index="Sex",
    columns="Pclass",
    values="Survived",
    aggfunc="mean",
)

print(sex_survival)
print(pclass_survival)
print(sex_pclass_survival)


# %% Submission format check
expected_columns = ["PassengerId", "Survived"]
print(sample_submission.columns.tolist())
print(sample_submission["Survived"].value_counts(normalize=True).sort_index())

assert sample_submission.columns.tolist() == expected_columns
assert len(sample_submission) == len(test)
