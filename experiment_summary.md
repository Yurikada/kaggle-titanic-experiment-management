# Experiment Summary

## Objective

Use the Kaggle Titanic competition as a compact case study for machine-learning experiment management, not as a leaderboard-optimization project.

## Key Results

| Experiment | Feature Set | Model | Validation Notes | Kaggle Public Score |
|---|---|---|---|---:|
| Depth 3 baseline | Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, CabinKnown | Decision tree depth 3 | Validation accuracy 0.7933 on one 80:20 split | 0.77990 |
| Depth 5 FamilySize replacement | Pclass, Sex, Age, FamilySize, Fare, Embarked, CabinKnown, IsInfant, IsElderly | Decision tree depth 5 | Validation accuracy 0.7542 on the same random_state=42 split | 0.74162 |

## Important Observations

### Single Split Risk

The first detailed error analysis found that all validation male survivors were predicted as non-survivors in one split.

After repeating the split three times, the result was less extreme:

| random_state | male_survivor_count | missed_male_survivor_count | captured_male_survivor_count | missed_male_survivor_rate |
|---:|---:|---:|---:|---:|
| 42 | 24 | 24 | 0 | 1.0000 |
| 7 | 23 | 18 | 5 | 0.7826 |
| 2026 | 25 | 20 | 5 | 0.8000 |

Conclusion:

- "All male survivors were missed" was split-dependent.
- "Male survivors were difficult to capture" remained a stable concern.

### FamilySize Replacement Failed

The FamilySize hypothesis was plausible:

```text
Men with family and men without family may behave differently.
```

However, replacing `SibSp` and `Parch` with `FamilySize` and increasing depth to 5 reduced the public score.

This failure is useful because it shows:

- feature aggregation can lose information
- deeper trees can overfit or generalize poorly
- changing model complexity and feature representation at the same time makes attribution difficult

## Depth Comparison

Using the FamilySize-included feature set, depths 5, 6, and 7 were compared across three validation splits.

| max_depth | validation_accuracy_mean | validation_accuracy_min | validation_accuracy_max | missed_male_survivor_rate_mean |
|---:|---:|---:|---:|---:|
| 5 | 0.7858 | 0.7542 | 0.8045 | 0.7498 |
| 6 | 0.7840 | 0.7598 | 0.7989 | 0.8470 |
| 7 | 0.7691 | 0.7486 | 0.7821 | 0.7486 |

Depth 7 had the lowest mean validation accuracy among these three depths.

## Lessons Learned

1. A low score can be a good experiment-management case.
2. Do not change feature representation and model complexity at the same time if the goal is causal interpretation.
3. Repeated validation splits help separate split-specific observations from stable model behavior.
4. A manual experiment table is valuable before moving to MLflow or automated search.
5. GitHub portfolio value can come from clear hypothesis, evaluation, error analysis, and failure review rather than leaderboard score alone.
