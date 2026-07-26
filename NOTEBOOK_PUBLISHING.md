# Titanic Notebook Publishing Guide

## Recommended Kaggle title

**Titanic: スコア悪化の原因を1回1変更で切り分ける | JP/EN**

## Short description

過去の提出で同時に変更した`FamilySize`とモデル複雑度を切り分ける、日本語中心の実験ケーススタディ。入力検査、リークを防ぐPipeline、反復交差検証、誤分類分析、提出検査までを再現可能にまとめています。

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

## Writing and publication policy

- Keep Japanese as the primary narrative. Use English for the abstract, headings, and key result labels rather than translating every sentence.
- Start from the failed comparison and the question it left unresolved.
- Separate observation, controlled comparison, interpretation, and limitations.
- Reply to comments with reproducible evidence and update the notebook when a reader finds an issue.
- Publish follow-up versions that change one experimental factor at a time.
- Do not claim leaderboard superiority; emphasize educational usefulness and transparent validation.
- Do not add generic upvote requests or promotional calls to action.

## Local validation

The repository currently may contain generated smoke data. Local execution validates the code path, not Kaggle statistics. The notebook detects this automatically and shows a warning. Final publication must be executed on the official 891-row training set and 418-row test set.
