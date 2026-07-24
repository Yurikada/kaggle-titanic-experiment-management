# Titanic Notebook Publishing Guide

## Recommended Kaggle title

**Titanic: Why a Plausible Feature Can Make a Model Worse | JP/EN**

## Short description

An explainable bilingual Titanic workflow: denominator-aware EDA, leakage-safe pipelines, controlled feature ablation, repeated cross-validation, subgroup error analysis, permutation importance, and a validated submission.

## Publication checklist

1. Upload `titanic_explainable_bilingual_workflow.ipynb` to Kaggle.
2. Add the official **Titanic - Machine Learning from Disaster** competition data.
3. Turn Internet **off**; the notebook has no network dependency.
4. Run all cells.
5. Confirm the data banner says `official Titanic competition data`.
6. Confirm every chart renders and the final summary contains no smoke-data warning.
7. Confirm `/kaggle/working/submission.csv` has 418 rows.
8. Save a version with all outputs.
9. Use the recommended title and short description above.
10. Add tags: `beginner`, `classification`, `feature-engineering`, `cross-validation`, `data-visualization`, `japanese`.

## Notebook Expert strategy

- Keep the first screen reader-focused: question, learning outcomes, navigation.
- Reply to comments with reproducible evidence and update the notebook when a reader finds an issue.
- Publish follow-up versions that change one experimental factor at a time.
- Do not claim leaderboard superiority; emphasize educational usefulness and transparent validation.
- Share the notebook in relevant Kaggle discussions only when it directly answers the thread.

## Local validation

The repository currently may contain generated smoke data. Local execution validates the code path, not Kaggle statistics. The notebook detects this automatically and shows a warning. Final publication must be executed on the official 891-row training set and 418-row test set.
