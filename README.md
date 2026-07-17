# Kaggle Titanic: Experiment Management Case Study

This repository is a learning-focused case study using the Kaggle Titanic competition.

The goal is not to present a high-ranking leaderboard solution. The goal is to show how a small machine-learning experiment can be managed, reviewed, and improved through:

- hypothesis-driven feature changes
- validation split checks
- model comparison
- subgroup error analysis
- failed-experiment review
- reproducible submission generation

Competition: <https://www.kaggle.com/competitions/titanic>

## Why This Project Exists

This project treats Titanic as a compact practice environment for model evaluation and experiment management.

The key learning was that a plausible domain hypothesis can still reduce performance if the experiment changes too many factors at once or if feature aggregation discards useful information.

## Data

Kaggle competition data is not included in this repository.

Download the data from Kaggle and place the files under:

```text
data/
  train.csv
  test.csv
  gender_submission.csv
```

You can also use the Kaggle CLI:

```powershell
kaggle competitions download -c titanic -p data
```

Then unzip the downloaded archive into `data/`.

## Project Structure

```text
.
├── 01_eda.py
├── 02_baseline.py
├── 03_depth4_error_analysis.py
├── 04_depth4_age_split_error_analysis.py
├── 05_multi_split_validation.py
├── 06_family_size_multi_split_validation.py
├── 07_depth_5_6_7_multi_split_validation.py
├── 08_depth5_family_size_replaced_submission.py
├── error_analysis/
├── submissions/
├── experiment_summary.md
├── requirements.txt
└── README.md
```

`data/` is intentionally ignored by Git.

## Experiment Flow

### 1. Baseline

The first submitted model was a simple decision tree:

```text
model: DecisionTreeClassifier(max_depth=3, random_state=42)
features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, CabinKnown
validation: 80:20 stratified split
```

Kaggle public score:

```text
0.77990
```

### 2. Error Analysis

The local validation errors suggested that male survivors were often predicted as non-survivors.

This led to a subgroup question:

```text
Are male survivors being missed because "male" dominates the tree,
or are there additional attributes attached to the male survivors?
```

The analysis introduced:

- missed survivor counts
- male survivor miss rate
- repeated validation splits
- proportion difference
- lift
- odds ratio when comparison groups exist

### 3. Multiple Validation Splits

A single validation split produced an extreme result:

```text
random_state=42:
male survivors: 24
missed male survivors: 24
captured male survivors: 0
```

After repeating the 80:20 stratified split three times:

```text
random_state  male_survivors  missed  captured  miss_rate
42            24              24      0         1.0000
7             23              18      5         0.7826
2026          25              20      5         0.8000
```

This separated two claims:

- Split-dependent: all male survivors were missed.
- More stable: male survivors were difficult for this model to capture.

### 4. FamilySize Hypothesis

Hypothesis:

```text
Men with family and men without family may behave differently.
```

The project tested `FamilySize = SibSp + Parch + 1`.

A later submission replaced `SibSp` and `Parch` with `FamilySize` and used `max_depth=5`.

Kaggle public score:

```text
0.74162
```

This was worse than the baseline.

## Main Lessons

### Low Score Can Still Be a Good Case Study

The depth-5 FamilySize replacement model was not a better model. It was a better learning case.

It showed:

- a plausible domain hypothesis can reduce performance
- feature aggregation can discard useful separate information
- deeper trees can overfit or fail to generalize
- changing multiple factors at once makes failure analysis harder

### Keep Experiment Conditions Explicit

When comparing `max_depth`, keep the feature set fixed.

When comparing features, keep the model depth and validation method fixed.

Otherwise, the effect of each change becomes difficult to interpret.

### Manual Experiment Tables Are Useful Early

Before using MLflow or automated hyperparameter search, manually writing experiment tables helps clarify:

- which parameters matter
- which metrics matter
- what should be logged automatically later

## Reproduce

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the baseline:

```powershell
python 02_baseline.py
```

Run the depth-5 FamilySize replacement submission:

```powershell
python 08_depth5_family_size_replaced_submission.py
```

The generated submission file is written to:

```text
submissions/decision_tree_depth5_family_size_replaced.csv
```

## Notes

This repository intentionally avoids copying a high-scoring public notebook. The purpose is to demonstrate an experiment process that can be explained in an interview:

- what was hypothesized
- what was changed
- what improved or failed
- what was learned
- what should be controlled in future experiments
