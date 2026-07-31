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

## Pre-registered comparisons notebook

### Kaggle title

**Titanic Preregistered Comparisons 判定の基準を結果より先に決める JP EN**

The title is built so that it slugifies to the registered id
`titanic-preregistered-comparisons-jp-en`. Kaggle drops Japanese characters when deriving a slug,
so the ASCII words must match the id in order: `Titanic Preregistered Comparisons ... JP EN`.
Punctuation such as `:` and `|` was removed because it produces repeated hyphens. The Japanese
phrase carries the central proposition and is dropped harmlessly from the slug.

The notebook's own H1 heading stays as 「判定の基準を、結果より先に決める」 and is independent of
this title.

### Short description

採否の基準を結果を見る前に登録してから、`Cabin` の表現、家族による情報の漏れ、`Age` の欠損処理、
木の深さの4つを比較した記録です。4つとも「現行のまま」で終わり、提出スコアも既存と同じでした。
その過程で、判定ルールの形が結論を反転させたこと、継続を決めた推定値が独立な観測で半減したこと、
分解能以下の差に付けた順位が入れ替わったことが観測できています。

### Suggested tags

`beginner`, `classification`, `cross-validation`, `model-evaluation`,
`data-visualization`, `japanese`

### Publication checklist

1. Upload `preregistration_publish/titanic_preregistered_comparisons.ipynb` to Kaggle.
2. Attach the official **Titanic - Machine Learning from Disaster** competition data.
3. Keep Internet off.
4. Run all cells. Local runtime is about two minutes; allow more on Kaggle.
5. Confirm the banner reports `official Titanic competition data` and shapes `(891, 12)` / `(418, 11)`.
6. Confirm all seven figures render and no cell reports an error.
7. Confirm the four selections read: presence flag, no adoption on leakage, global median, depth 3.
   If any differs, the narrative around it must be re-read before publishing.
8. Confirm `/kaggle/working/submission.csv` has 418 rows with `PassengerId`, `Survived`.
9. Save a version with all outputs.
10. `kernel-metadata.json` is `is_private: true`. Switch it only when publication is intended.

### External value that the notebook cannot compute

`PUBLIC_SCORE = 0.77990` is declared as a constant, from submission `55117112`. It is the only
number in the notebook that is not produced by the run. If a different file is ever submitted,
this constant and section 12 must be updated together.

### Local build and validation

```powershell
python build_preregistration_notebook.py
python validate_preregistration_notebook.py
```

Local runs write to `preregistration_publish/outputs/`, separate from `submissions/`. Figures are
written to `preregistration_review/`. The validator also checks that the generated submission is
identical to `submissions/decision_tree_depth3_baseline.csv`.

## Second notebook: uncertainty, missingness, and error structure

### Recommended Kaggle title

**Titanic: 差と呼ぶ前に不確実性・欠損・誤りを見る | JP/EN**

### Short description

生存率の棒グラフを比べる手前で立ち止まり、比率のWilson信頼区間、差のブートストラップ分布、
欠損の共起と欠損行を落としたときの構成変化、OOF誤分類の層別集中と予測確率の較正を
日本語中心で可視化した学習記録です。モデルは既存のdepth 3ベースラインを固定条件として引き継いでいます。

### Suggested tags

`beginner`, `classification`, `data-visualization`, `exploratory-data-analysis`,
`cross-validation`, `japanese`

### Publication checklist

1. Upload `uncertainty_publish/titanic_uncertainty_and_error_structure.ipynb` to Kaggle.
2. Attach the official **Titanic - Machine Learning from Disaster** competition data.
3. Keep Internet off; the notebook has no network dependency.
4. Run all cells.
5. Confirm the banner reports `official Titanic competition data` and shapes `(891, 12)` / `(418, 11)`.
6. Confirm all six figures render and no cell reports an error.
7. Confirm `/kaggle/working/submission.csv` has 418 rows with columns `PassengerId`, `Survived`.
8. Save a version with all outputs.
9. Fill the `［自分で記入］` cells before making the notebook public. The notebook is incomplete
   as a learning record while those cells are blank.
10. Review the Japanese wording and English summaries, then switch `is_private` in
    `uncertainty_publish/kernel-metadata.json` only after the public version is intended.

### Chart language

Figure text is written in English on purpose. The Kaggle image has no Japanese font installed, so
Japanese axis labels and titles render as tofu boxes. The Japanese explanation stays in markdown.

### Local build and validation

```powershell
python build_uncertainty_notebook.py
python validate_uncertainty_notebook.py
```

Local runs write the submission to `uncertainty_publish/outputs/`, which is separate from
`submissions/`, so notebook runs never overwrite the script-generated submissions.
Rendered figures are written to `uncertainty_review/` for inspection.

## Local validation

The repository currently may contain generated smoke data. Local execution validates the code path, not Kaggle statistics. The notebook detects this automatically and shows a warning. Final publication must be executed on the official 891-row training set and 418-row test set.
