from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
PUBLISH_DIR = ROOT / "uncertainty_publish"
PUBLISH_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = PUBLISH_DIR / "titanic_uncertainty_and_error_structure.ipynb"

notebook = nbf.v4.new_notebook()
cells = []


def md(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(dedent(source).strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(dedent(source).strip()))


md(
    """
    # Titanic: 差と呼ぶ前に、不確実性・欠損・誤りの偏りを見る

    ## Reading uncertainty, missingness, and error structure | JP / EN

    Titanicの解説は、性別や客室等級ごとの生存率を棒グラフで並べ、そのまま特徴量選択へ進むものが多くあります。
    このNotebookは、その手前にある問いを扱います。棒の高さが違うとき、どこからを「差」と呼べるのでしょうか。

    扱う軸は3つです。生存率の**不確実性**、欠損の**構造**、モデルの誤りの**偏り**。
    いずれも平均値ひとつでは見えない情報で、可視化の対象にしています。

    高いスコアを目指すNotebookではありません。初学者が判断を保留する場所を決めるための記録です。

    ### English abstract

    Most Titanic tutorials compare survival rates as bare bars and move straight to feature selection.
    This notebook stays one step earlier and asks when a visible gap can be called a difference.

    It looks at three things the mean alone hides: the uncertainty of each rate, the structure of
    missingness, and where a baseline model's errors concentrate. The goal is a documented learning
    process, not leaderboard performance.
    """
)

md(
    """
    ## 1. このNotebookの問い / Question

    > 集計した生存率の差、欠損の多い列、正解率の数字を、それぞれどこまで根拠として扱えるか。

    次の3点を、図から読めるようにします。

    | 軸 | 見るもの | 平均だけでは見えないこと |
    |---|---|---|
    | 不確実性 | 比率の信頼区間、差のブートストラップ分布 | 分母が小さい群の差は、偶然の範囲に収まりうる |
    | 欠損の構造 | 欠損パターンの共起、欠損行を落とした場合の残り方 | 欠損処理が、特定の層を分析から消しうる |
    | 誤りの偏り | 木の深さと性能の関係、層別の誤分類率、誤りの種類、予測確率の較正 | 同じ正解率でも、誤りが集中する層は異なりうる |

    > **English:** The notebook asks how far an observed gap, a missing-heavy column, and an accuracy
    > number can be trusted. Each of the three axes is chosen because a single mean value hides it.

    ### 実行前に自分で書く欄

    グラフを見る前に埋めます。正解を書く欄ではありません。外れた場合も消さずに残します。

    | 問い | 実行前の記録 |
    |---|---|
    | 生存率の差が「偶然ではない」と判断する基準は何か。 | **［自分で記入］** |
    | `Age`の欠損行を落とすと、どの層が減ると予想するか。 | **［自分で記入］** |
    | 正解率が同じでも困る誤り方とは、どんな誤り方か。 | **［自分で記入］** |
    """
)

md(
    """
    ## 2. 実行環境とデータの解決 / Setup and data resolution

    Kaggleでは`/kaggle/input`配下から`Survived`を含むtrainと含まないtestの組を探します。
    フォルダ名を決め打ちにすると、入力の付け替えで壊れるためです。ローカルでは`data/`を使います。

    図中の文字は英語で書きます。Kaggle実行環境には日本語フォントが入っておらず、
    日本語ラベルは文字化けするためです。説明は本文側に日本語で置きます。
    """
)

code(
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import sklearn

    from IPython.display import Markdown, display
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier

    RANDOM_STATE = 42
    N_SPLITS = 5
    BASELINE_DEPTH = 3
    TARGET_COLUMN = "Survived"
    ID_COLUMN = "PassengerId"
    Z_95 = 1.959963984540054
    N_BOOTSTRAP = 4000

    COLORS = {
        "blue": "#2F6690",
        "gold": "#D4A72C",
        "orange": "#D97706",
        "ink": "#263238",
        "muted": "#6B7A82",
        "light": "#D9E1E5",
        "paper": "#FAFBFC",
    }

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.facecolor": COLORS["paper"],
        "axes.facecolor": COLORS["paper"],
        "axes.labelcolor": COLORS["ink"],
        "text.color": COLORS["ink"],
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "axes.unicode_minus": False,
        "font.size": 10,
    })

    print("pandas:", pd.__version__)
    print("scikit-learn:", sklearn.__version__)
    """
)

code(
    """
    def is_titanic_pair(directory: Path) -> bool:
        train_path = directory / "train.csv"
        test_path = directory / "test.csv"
        if not train_path.exists() or not test_path.exists():
            return False
        train_columns = pd.read_csv(train_path, nrows=2).columns
        test_columns = pd.read_csv(test_path, nrows=2).columns
        return (
            TARGET_COLUMN in train_columns
            and TARGET_COLUMN not in test_columns
            and ID_COLUMN in train_columns
            and ID_COLUMN in test_columns
        )


    KAGGLE_INPUT_ROOT = Path("/kaggle/input")
    LOCAL_CANDIDATES = [Path("data"), Path("../data")]

    if KAGGLE_INPUT_ROOT.exists():
        kaggle_candidates = sorted({
            path.parent
            for path in KAGGLE_INPUT_ROOT.rglob("train.csv")
            if is_titanic_pair(path.parent)
        })
    else:
        kaggle_candidates = []

    if len(kaggle_candidates) == 1:
        DATA_DIR = kaggle_candidates[0]
        OUTPUT_DIR = Path("/kaggle/working")
        ENVIRONMENT = "Kaggle"
    elif len(kaggle_candidates) > 1:
        raise RuntimeError(
            "Multiple Titanic train/test pairs were found. "
            f"Attach only the intended competition input: {kaggle_candidates}"
        )
    else:
        matching_local = [path for path in LOCAL_CANDIDATES if is_titanic_pair(path)]
        if len(matching_local) != 1:
            raise FileNotFoundError(
                "Attach the Titanic competition data on Kaggle, or place "
                "train.csv and test.csv under the local data directory."
            )
        DATA_DIR = matching_local[0]
        OUTPUT_DIR = Path("uncertainty_publish/outputs")
        ENVIRONMENT = "Local"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    IS_OFFICIAL_SHAPE = train.shape == (891, 12) and test.shape == (418, 11)
    data_mode = (
        "official Titanic competition data"
        if IS_OFFICIAL_SHAPE
        else "non-official shape (results below are not comparable)"
    )

    display(Markdown(
        f"**Environment / 実行環境:** `{ENVIRONMENT}`  \\n"
        f"**Input / 入力先:** `{DATA_DIR}`  \\n"
        f"**Data check / データ判定:** `{data_mode}`  \\n"
        f"**Shapes / 形状:** train `{train.shape}`, test `{test.shape}`"
    ))
    """
)

md(
    """
    ### 2.1 入力の境界を先に固定する / Fix the input contract first

    可視化やモデルの前に、目的変数、ID、train/testの列構成を確認します。
    ここが崩れていると、後段の図が正しく描けても提出物としては使えません。
    """
)

code(
    """
    assert TARGET_COLUMN in train.columns
    assert TARGET_COLUMN not in test.columns
    assert train[ID_COLUMN].is_unique
    assert test[ID_COLUMN].is_unique
    assert set(train[TARGET_COLUMN].unique()).issubset({0, 1})
    assert set(train.columns) - {TARGET_COLUMN} == set(test.columns)
    assert train[TARGET_COLUMN].notna().all()

    input_checks = pd.DataFrame({
        "check": [
            "train rows",
            "test rows",
            "feature columns",
            "duplicate PassengerId",
            "target values",
            "overall survival rate",
        ],
        "value": [
            len(train),
            len(test),
            test.shape[1] - 1,
            int(train[ID_COLUMN].duplicated().sum() + test[ID_COLUMN].duplicated().sum()),
            ", ".join(str(value) for value in sorted(train[TARGET_COLUMN].unique())),
            f"{train[TARGET_COLUMN].mean():.4f}",
        ],
    })
    display(input_checks)
    """
)

md(
    """
    ## 3. 軸1: 比率には分母と幅がある / Axis 1: every rate carries a denominator and a width

    生存率は、割り算の結果である前に、限られた乗客数から推定した値です。
    同じ「0.75」でも、8人中6人と400人中300人では、次の乗客に対する確からしさが違います。

    ここではWilson信頼区間を使います。比率が0や1へ寄る場合や分母が小さい場合に、
    正規近似（p ± 1.96·√(p(1−p)/n)）よりも区間が妥当な範囲に収まるためです。
    区間の意味は「同じ手続きを繰り返したとき、95%の区間が真の比率を含む」であり、
    「真の比率が95%の確率でこの中にある」ではありません。

    > **English:** A rate is an estimate from a limited denominator. Wilson intervals behave better
    > than the normal approximation for small `n` and for rates near 0 or 1.
    """
)

code(
    """
    def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
        if total == 0:
            return (np.nan, np.nan)
        proportion = successes / total
        denominator = 1.0 + z**2 / total
        center = (proportion + z**2 / (2 * total)) / denominator
        half_width = (
            z
            * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
            / denominator
        )
        return center - half_width, center + half_width


    def error_bar_widths(
        estimate: pd.Series,
        lower: pd.Series,
        upper: pd.Series,
    ) -> list[np.ndarray]:
        # p=1 や p=0 の群では、Wilson区間の端と推定値が浮動小数点誤差の分だけ
        # 逆転しうる。errorbarは負の幅を受け付けないため、0で切り上げる。
        below = np.maximum(estimate.to_numpy() - lower.to_numpy(), 0.0)
        above = np.maximum(upper.to_numpy() - estimate.to_numpy(), 0.0)
        return [below, above]


    def rate_table(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
        grouped = (
            frame.groupby(group_column, dropna=False)[TARGET_COLUMN]
            .agg(survivors="sum", passengers="count")
            .reset_index()
        )
        grouped["rate"] = grouped["survivors"] / grouped["passengers"]
        bounds = [
            wilson_interval(int(row.survivors), int(row.passengers))
            for row in grouped.itertuples()
        ]
        grouped["ci_low"] = [bound[0] for bound in bounds]
        grouped["ci_high"] = [bound[1] for bound in bounds]
        grouped["ci_width"] = grouped["ci_high"] - grouped["ci_low"]
        labels = grouped[group_column].astype(object).where(
            grouped[group_column].notna(),
            "missing",
        )
        grouped["group"] = f"{group_column}=" + labels.astype(str)
        return grouped


    rate_tables = pd.concat(
        [rate_table(train, column) for column in ["Sex", "Pclass", "Embarked"]],
        ignore_index=True,
    )
    display(
        rate_tables[
            ["group", "passengers", "survivors", "rate", "ci_low", "ci_high", "ci_width"]
        ].round(4)
    )
    """
)

code(
    """
    plot_frame = rate_tables.sort_values("rate").reset_index(drop=True)
    positions = np.arange(len(plot_frame))
    overall_rate = train[TARGET_COLUMN].mean()

    figure, axis = plt.subplots(figsize=(9, 5.2))
    axis.errorbar(
        plot_frame["rate"],
        positions,
        xerr=error_bar_widths(
            plot_frame["rate"],
            plot_frame["ci_low"],
            plot_frame["ci_high"],
        ),
        fmt="o",
        color=COLORS["blue"],
        ecolor=COLORS["muted"],
        elinewidth=2,
        capsize=4,
        markersize=7,
    )
    axis.axvline(
        overall_rate,
        color=COLORS["orange"],
        linestyle="--",
        linewidth=1.2,
        label=f"overall rate {overall_rate:.3f}",
    )
    for position, row in zip(positions, plot_frame.itertuples()):
        axis.text(
            row.ci_high + 0.015,
            position,
            f"n={row.passengers}",
            va="center",
            fontsize=9,
            color=COLORS["muted"],
        )

    axis.set_yticks(positions)
    axis.set_yticklabels(plot_frame["group"])
    axis.set_xlim(0, 1.12)
    axis.set_xlabel("Survival rate with 95% Wilson interval")
    axis.set_title("Survival rate by group, shown with its denominator")
    axis.legend(loc="lower right")
    figure.tight_layout()
    plt.show()
    """
)

md(
    """
    区間の幅は分母に依存します。分母の小さい群では、点だけを見て順位を決めると、
    次のデータで入れ替わりうる差を確定した差として扱う危険があります。

    そこで、2群の差そのものをブートストラップで推定します。
    各群から復元抽出で同じ人数を取り直し、生存率の差を4000回計算します。
    差の分布が0を跨ぐかどうかが、判断を保留すべきかの目安になります。

    > **English:** Interval width depends on `n`. The next chart resamples each group with
    > replacement and plots the distribution of the rate difference, so a gap that includes zero
    > stays visibly undecided.
    """
)

code(
    """
    rng = np.random.default_rng(RANDOM_STATE)


    def bootstrap_rate_difference(
        frame: pd.DataFrame,
        column: str,
        left_value,
        right_value,
        n_bootstrap: int = N_BOOTSTRAP,
    ) -> dict:
        left = frame.loc[frame[column] == left_value, TARGET_COLUMN].to_numpy()
        right = frame.loc[frame[column] == right_value, TARGET_COLUMN].to_numpy()
        differences = np.empty(n_bootstrap)
        for index in range(n_bootstrap):
            left_sample = rng.choice(left, size=left.size, replace=True)
            right_sample = rng.choice(right, size=right.size, replace=True)
            differences[index] = left_sample.mean() - right_sample.mean()
        low, high = np.percentile(differences, [2.5, 97.5])
        return {
            "label": f"{column}: {left_value} - {right_value}",
            "observed": left.mean() - right.mean(),
            "n_left": left.size,
            "n_right": right.size,
            "ci_low": low,
            "ci_high": high,
            "crosses_zero": bool(low < 0 < high),
            "draws": differences,
        }


    comparisons = [
        bootstrap_rate_difference(train, "Sex", "female", "male"),
        bootstrap_rate_difference(train, "Embarked", "Q", "S"),
    ]

    display(
        pd.DataFrame([
            {
                key: value
                for key, value in comparison.items()
                if key != "draws"
            }
            for comparison in comparisons
        ]).round(4)
    )
    """
)

code(
    """
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)

    for axis, comparison in zip(axes, comparisons):
        axis.hist(
            comparison["draws"],
            bins=40,
            color=COLORS["blue"],
            edgecolor="white",
            alpha=0.85,
        )
        axis.axvline(0, color=COLORS["ink"], linewidth=1.4)
        axis.axvline(
            comparison["ci_low"],
            color=COLORS["orange"],
            linestyle="--",
            linewidth=1.2,
        )
        axis.axvline(
            comparison["ci_high"],
            color=COLORS["orange"],
            linestyle="--",
            linewidth=1.2,
        )
        verdict = "includes 0" if comparison["crosses_zero"] else "excludes 0"
        axis.set_title(
            f"{comparison['label']}\\n"
            f"n={comparison['n_left']} vs {comparison['n_right']}, 95% CI {verdict}"
        )
        axis.set_xlabel("Bootstrap difference in survival rate")

    axes[0].set_ylabel("Bootstrap draws")
    figure.suptitle("Same chart, different decisions: one gap survives resampling, one does not")
    figure.tight_layout()
    plt.show()
    """
)

md(
    """
    **観察の記録欄**

    | 項目 | 記入 |
    |---|---|
    | 区間が最も広かった群と、その分母 | **［自分で記入］** |
    | 差の分布が0を跨いだ比較 | **［自分で記入］** |
    | この結果から、次に固定したい条件 | **［自分で記入］** |
    """
)

md(
    """
    ## 4. 軸2: 欠損は行の性質であり、列の欠陥ではない / Axis 2: missingness is a property of rows

    欠損は「埋めるか落とすか」の前に、どの行に集中しているかを見る対象です。
    `Age`と`Cabin`の欠損が同じ乗客に重なっていれば、その乗客は複数の列で情報が薄いことになります。

    ここでは3つを確認します。欠損の共起、欠損行を落とした場合に消える層、
    そして欠損の有無そのものが生存率と関係するかどうかです。

    > **English:** Missingness is examined as a row-level pattern: which passengers are missing what,
    > who disappears if those rows are dropped, and whether the fact of being missing is itself
    > associated with survival.
    """
)

code(
    """
    missing_columns = ["Age", "Cabin", "Embarked", "Fare"]
    missing_summary = pd.DataFrame({
        "train_missing_rate": train[missing_columns].isna().mean(),
        "test_missing_rate": test[missing_columns].isna().mean(),
        "train_missing_count": train[missing_columns].isna().sum(),
        "test_missing_count": test[missing_columns].isna().sum(),
    })
    display(missing_summary.round(4))

    pattern_columns = ["Age", "Cabin", "Embarked"]
    missing_flags = train[pattern_columns].isna()
    co_occurrence = pd.crosstab(
        missing_flags["Age"].map({True: "Age missing", False: "Age present"}),
        missing_flags["Cabin"].map({True: "Cabin missing", False: "Cabin present"}),
    )
    display(co_occurrence)
    """
)

code(
    """
    sort_keys = missing_flags.astype(int)
    order = sort_keys.sort_values(
        by=["Cabin", "Age", "Embarked"],
        ascending=[False, False, False],
    ).index
    matrix = sort_keys.loc[order].to_numpy().T

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 4.6),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    axes[0].imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=plt.matplotlib.colors.ListedColormap([COLORS["light"], COLORS["orange"]]),
        vmin=0,
        vmax=1,
    )
    axes[0].set_yticks(range(len(pattern_columns)))
    axes[0].set_yticklabels(pattern_columns)
    axes[0].set_xlabel("Passengers, sorted by missing pattern")
    axes[0].set_title("Where the gaps sit (orange = missing)")
    axes[0].grid(False)

    age_missing_rates = rate_table(
        train.assign(
            AgeRecorded=np.where(train["Age"].isna(), "Age missing", "Age present")
        ),
        "AgeRecorded",
    )
    cabin_rates = rate_table(
        train.assign(
            CabinRecorded=np.where(train["Cabin"].isna(), "Cabin missing", "Cabin present")
        ),
        "CabinRecorded",
    )
    signal_frame = pd.concat([age_missing_rates, cabin_rates], ignore_index=True)
    signal_positions = np.arange(len(signal_frame))

    axes[1].errorbar(
        signal_frame["rate"],
        signal_positions,
        xerr=error_bar_widths(
            signal_frame["rate"],
            signal_frame["ci_low"],
            signal_frame["ci_high"],
        ),
        fmt="o",
        color=COLORS["blue"],
        ecolor=COLORS["muted"],
        elinewidth=2,
        capsize=4,
    )
    axes[1].axvline(
        train[TARGET_COLUMN].mean(),
        color=COLORS["orange"],
        linestyle="--",
        linewidth=1.2,
    )
    axes[1].set_yticks(signal_positions)
    axes[1].set_yticklabels(
        [
            label.split("=", 1)[1] + f" (n={count})"
            for label, count in zip(signal_frame["group"], signal_frame["passengers"])
        ]
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Survival rate with 95% Wilson interval")
    axes[1].set_title("Being missing is itself informative")

    figure.tight_layout()
    plt.show()
    """
)

md(
    """
    次に、`Age`の欠損行を落とした場合に何が起きるかを見ます。
    欠損処理は「行を減らす操作」ではなく、「残る乗客の構成を変える操作」です。
    """
)

code(
    """
    dropped = train[train["Age"].isna()]
    kept = train[train["Age"].notna()]

    composition_records = []
    for column in ["Pclass", "Sex", "Embarked"]:
        for label, frame in [("kept (Age present)", kept), ("dropped (Age missing)", dropped)]:
            shares = frame[column].astype(str).value_counts(normalize=True)
            for value, share in shares.items():
                composition_records.append({
                    "variable": f"{column}={value}",
                    "subset": label,
                    "share": share,
                })

    composition = (
        pd.DataFrame(composition_records)
        .pivot(index="variable", columns="subset", values="share")
        .fillna(0.0)
        .sort_index()
    )
    composition["shift"] = (
        composition["dropped (Age missing)"] - composition["kept (Age present)"]
    )
    display(composition.round(4))

    figure, axis = plt.subplots(figsize=(9.5, 5))
    bar_positions = np.arange(len(composition))
    bar_height = 0.38
    axis.barh(
        bar_positions + bar_height / 2,
        composition["kept (Age present)"],
        height=bar_height,
        color=COLORS["blue"],
        label="kept (Age present)",
    )
    axis.barh(
        bar_positions - bar_height / 2,
        composition["dropped (Age missing)"],
        height=bar_height,
        color=COLORS["orange"],
        label="dropped (Age missing)",
    )
    axis.set_yticks(bar_positions)
    axis.set_yticklabels(composition.index)
    axis.set_xlabel("Share within each subset")
    axis.set_title(
        f"Dropping {len(dropped)} rows with missing Age changes who remains"
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    plt.show()
    """
)

md(
    """
    **観察の記録欄**

    | 項目 | 記入 |
    |---|---|
    | `Age`欠損行を落としたとき、最も構成比が動いた層 | **［自分で記入］** |
    | 欠損の有無自体が生存率と関係して見えたか | **［自分で記入］** |
    | 中央値補完を選ぶ場合、その判断で受け入れる前提 | **［自分で記入］** |
    """
)

md(
    """
    ## 5. 軸3: モデルの複雑さと、誤りの置き場所 / Axis 3: complexity, and where the errors sit

    特徴量は、この学習記録の前段で自分が使った8列（`CabinKnown`を含む）をそのまま引き継ぎます。
    モデルの種類も決定木のままです。この節で変えるのは深さ`max_depth`だけで、
    前処理、特徴量、分割の作り方は固定します。

    深さ3は、前段のベースラインで置いた値です。ただし、深さを1から順に上げて確かめた値ではありません。
    そこで5.1では深さ1から順に上げ、訓練データでの正解率と未知データでの正解率がどう離れるかを見ます。
    5.2では、現在のベースライン条件（深さ3）の誤りが、どの層に集中するかを見ます。

    評価は層化5-foldのOut-of-Fold予測です。全891行に対して、その行を学習に使っていない
    モデルの予測が1つずつ得られます。

    > **English:** Features and model family are inherited from the earlier baseline. Only `max_depth`
    > changes in 5.1, so the depth-3 value can be traced rather than assumed. Evaluation uses
    > stratified 5-fold out-of-fold prediction.
    """
)

code(
    """
    def add_baseline_features(frame: pd.DataFrame) -> pd.DataFrame:
        features = frame.copy()
        features["CabinKnown"] = features["Cabin"].notna().astype(int)
        return features


    FEATURE_COLUMNS = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "CabinKnown",
    ]
    NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare", "CabinKnown"]
    CATEGORICAL_FEATURES = ["Sex", "Embarked"]

    train_features = add_baseline_features(train)
    test_features = add_baseline_features(test)

    X = train_features[FEATURE_COLUMNS]
    y = train_features[TARGET_COLUMN]
    X_test = test_features[FEATURE_COLUMNS]


    def build_baseline_pipeline(max_depth: int | None = BASELINE_DEPTH) -> Pipeline:
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        preprocess = ColumnTransformer([
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ])
        return Pipeline([
            ("preprocess", preprocess),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    random_state=RANDOM_STATE,
                ),
            ),
        ])


    cross_validation = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    print("features:", len(FEATURE_COLUMNS))
    print("baseline depth:", BASELINE_DEPTH)
    """
)

md(
    """
    ### 5.1 深さを1から上げる / Trace the depth from 1

    決定木の深さは、分割を何回まで重ねてよいかを決めます。深さが増えるほど、
    訓練データの細かい違いまで表現できる一方、その違いが次のデータでも成り立つとは限りません。

    ここでは深さ1から10、および制限なしを比較します。深さ以外の条件は固定です。
    分割の当たり外れで順位が入れ替わるため、5つのseed × 5-foldの計25 foldで評価します。
    seed 42は、5.2以降で使うベースラインと同じ分割です。

    上限を10と制限なしまで含めるのは、最も良い値が調べた範囲の端に来たときに、
    その外側を見ていないことに気づけるようにするためです。

    > **English:** Depth is swept from 1 to 10 plus unlimited, with everything else fixed. Each depth
    > is evaluated on 25 folds (5 seeds x 5 folds). The range includes the unlimited case so that a
    > best value sitting at the edge of the grid stays visible as an unsearched boundary.
    """
)

code(
    """
    DEPTH_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]
    DEPTH_LABELS = ["unlimited" if depth is None else str(depth) for depth in DEPTH_GRID]
    DEPTH_SEEDS = [0, 1, 2, 3, RANDOM_STATE]

    depth_records = []
    for seed in DEPTH_SEEDS:
        seed_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold_number, (fold_train_index, fold_valid_index) in enumerate(
            seed_cv.split(X, y),
            start=1,
        ):
            X_fold_train = X.iloc[fold_train_index]
            X_fold_valid = X.iloc[fold_valid_index]
            y_fold_train = y.iloc[fold_train_index]
            y_fold_valid = y.iloc[fold_valid_index]

            for depth, label in zip(DEPTH_GRID, DEPTH_LABELS):
                fold_pipeline = build_baseline_pipeline(max_depth=depth)
                fold_pipeline.fit(X_fold_train, y_fold_train)
                depth_records.append({
                    "seed": seed,
                    "fold": fold_number,
                    "depth_label": label,
                    "train_accuracy": accuracy_score(
                        y_fold_train,
                        fold_pipeline.predict(X_fold_train),
                    ),
                    "validation_accuracy": accuracy_score(
                        y_fold_valid,
                        fold_pipeline.predict(X_fold_valid),
                    ),
                    "leaves": int(fold_pipeline.named_steps["model"].get_n_leaves()),
                })

    depth_results = pd.DataFrame(depth_records)

    depth_summary = (
        depth_results.groupby("depth_label")
        .agg(
            train_mean=("train_accuracy", "mean"),
            validation_mean=("validation_accuracy", "mean"),
            validation_std=("validation_accuracy", "std"),
            validation_low=("validation_accuracy", lambda values: np.percentile(values, 2.5)),
            validation_high=("validation_accuracy", lambda values: np.percentile(values, 97.5)),
            leaves_mean=("leaves", "mean"),
        )
        .reindex(DEPTH_LABELS)
        .reset_index()
    )
    depth_summary["train_minus_validation"] = (
        depth_summary["train_mean"] - depth_summary["validation_mean"]
    )

    assert len(depth_results) == len(DEPTH_SEEDS) * N_SPLITS * len(DEPTH_GRID)
    display(depth_summary.round(4))
    """
)

code(
    """
    fold_accuracy = depth_results.pivot_table(
        index=["seed", "fold"],
        columns="depth_label",
        values="validation_accuracy",
    ).reindex(columns=DEPTH_LABELS)

    baseline_label = str(BASELINE_DEPTH)
    depth_rng = np.random.default_rng(RANDOM_STATE)
    difference_records = []
    for label in DEPTH_LABELS:
        differences = (fold_accuracy[label] - fold_accuracy[baseline_label]).to_numpy()
        resample_index = depth_rng.integers(
            0,
            differences.size,
            size=(N_BOOTSTRAP, differences.size),
        )
        bootstrap_means = differences[resample_index].mean(axis=1)
        low, high = np.percentile(bootstrap_means, [2.5, 97.5])
        difference_records.append({
            "depth_label": label,
            "mean_difference": differences.mean(),
            "ci_low": low,
            "ci_high": high,
            "folds_better": int((differences > 0).sum()),
            "folds_worse": int((differences < 0).sum()),
            "crosses_zero": bool(low < 0 < high),
        })

    depth_differences = pd.DataFrame(difference_records)
    display(depth_differences.round(4))
    """
)

code(
    """
    positions = np.arange(len(DEPTH_LABELS))
    baseline_position = DEPTH_LABELS.index(str(BASELINE_DEPTH))

    best_validation_mean = depth_summary["validation_mean"].max()
    tied_best_labels = depth_summary.loc[
        depth_summary["validation_mean"].round(4) == round(best_validation_mean, 4),
        "depth_label",
    ].tolist()
    is_baseline_row = depth_differences["depth_label"] == baseline_label
    overlapping_labels = depth_differences.loc[
        depth_differences["crosses_zero"] & ~is_baseline_row,
        "depth_label",
    ].tolist()
    separated_labels = depth_differences.loc[
        ~depth_differences["crosses_zero"] & ~is_baseline_row,
        "depth_label",
    ].tolist()

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    axes[0].fill_between(
        positions,
        depth_summary["validation_low"],
        depth_summary["validation_high"],
        color=COLORS["blue"],
        alpha=0.15,
        label="validation spread across 25 folds",
    )
    axes[0].plot(
        positions,
        depth_summary["train_mean"],
        marker="o",
        color=COLORS["gold"],
        label="train accuracy (mean)",
    )
    axes[0].plot(
        positions,
        depth_summary["validation_mean"],
        marker="o",
        color=COLORS["blue"],
        label="validation accuracy (mean)",
    )
    axes[0].axvline(
        baseline_position,
        color=COLORS["muted"],
        linestyle=":",
        linewidth=1.2,
    )
    axes[0].annotate(
        f"baseline depth {BASELINE_DEPTH}",
        (baseline_position, axes[0].get_ylim()[0]),
        textcoords="offset points",
        xytext=(6, 12),
        fontsize=9,
        color=COLORS["muted"],
    )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(DEPTH_LABELS)
    axes[0].set_xlabel("max_depth")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Training accuracy keeps rising, validation accuracy does not")
    axes[0].legend(loc="lower right", fontsize=9)

    axes[1].errorbar(
        positions,
        depth_differences["mean_difference"],
        yerr=error_bar_widths(
            depth_differences["mean_difference"],
            depth_differences["ci_low"],
            depth_differences["ci_high"],
        ),
        fmt="o",
        color=COLORS["blue"],
        ecolor=COLORS["muted"],
        elinewidth=1.8,
        capsize=4,
    )
    axes[1].axhline(0, color=COLORS["ink"], linewidth=1.2)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(DEPTH_LABELS)
    axes[1].set_xlabel("max_depth")
    axes[1].set_ylabel(f"Validation accuracy - depth {BASELINE_DEPTH}")
    axes[1].set_title("Paired difference from the baseline depth, same folds")

    figure.tight_layout()
    plt.show()

    display(Markdown(
        f"- 検証正解率の平均の最大値は`{best_validation_mean:.4f}`で、"
        f"小数4桁ではこの値に`max_depth={'`, `'.join(tied_best_labels)}`が並んだ。"
        "順位を付けられる差ではない。  \\n"
        f"- 深さ{BASELINE_DEPTH}との差の95%区間が0を含んだのは"
        f"`{'`, `'.join(overlapping_labels)}`だった。この設計では差を確定できない。  \\n"
        f"- 0を含まなかったのは`{'`, `'.join(separated_labels)}`で、いずれも深さ"
        f"{BASELINE_DEPTH}より低かった。  \\n"
        f"- 訓練正解率と検証正解率の差は、深さ1で`{depth_summary.loc[0, 'train_minus_validation']:.4f}`、"
        f"`unlimited`で`{depth_summary.iloc[-1]['train_minus_validation']:.4f}`まで開いた。"
        f"葉の数は平均`{depth_summary.loc[0, 'leaves_mean']:.0f}`から"
        f"`{depth_summary.iloc[-1]['leaves_mean']:.0f}`へ増えた。"
    ))
    """
)

md(
    """
    **観察の記録欄**

    | 項目 | 記入 |
    |---|---|
    | 訓練正解率と検証正解率が離れ始めた深さ | **［自分で記入］** |
    | 深さ3との差が0を跨がなかった深さ（あれば） | **［自分で記入］** |
    | 平均が最良の深さを採用しない場合、その理由 | **［自分で記入］** |
    | 次に固定して比べたい条件 | **［自分で記入］** |
    """
)

md(
    """
    ### 5.2 誤りはどの層に集中するか / Where the errors concentrate

    ここからは、現在のベースライン条件（深さ3）に戻して誤りの中身を見ます。
    5.1の結果を受けて深さを変える場合は、この節も同じ条件で読み直す必要があります。
    """
)

code(
    """
    oof_prediction = cross_val_predict(
        build_baseline_pipeline(),
        X,
        y,
        cv=cross_validation,
        n_jobs=1,
    )
    oof_probability = cross_val_predict(
        build_baseline_pipeline(),
        X,
        y,
        cv=cross_validation,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    oof = train_features.assign(
        predicted=oof_prediction,
        predicted_probability=oof_probability,
    )
    oof["is_error"] = (oof["predicted"] != oof[TARGET_COLUMN]).astype(int)
    oof["error_type"] = np.select(
        [
            (oof[TARGET_COLUMN] == 1) & (oof["predicted"] == 0),
            (oof[TARGET_COLUMN] == 0) & (oof["predicted"] == 1),
        ],
        ["missed survivor", "false survivor"],
        default="correct",
    )

    overall_metrics = pd.DataFrame({
        "metric": ["accuracy", "balanced accuracy", "errors", "passengers"],
        "value": [
            f"{accuracy_score(y, oof_prediction):.4f}",
            f"{balanced_accuracy_score(y, oof_prediction):.4f}",
            int(oof['is_error'].sum()),
            len(oof),
        ],
    })
    display(overall_metrics)
    """
)

code(
    """
    error_grid = (
        oof.groupby(["Pclass", "Sex"], observed=True)
        .agg(error_rate=("is_error", "mean"), passengers=("is_error", "size"))
        .reset_index()
    )
    error_matrix = error_grid.pivot(index="Pclass", columns="Sex", values="error_rate")
    count_matrix = error_grid.pivot(index="Pclass", columns="Sex", values="passengers")

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    error_values = error_matrix.to_numpy()
    highest_error = float(np.nanmax(error_values))
    image = axes[0].imshow(
        error_values,
        cmap="YlOrBr",
        vmin=0,
        vmax=highest_error,
    )
    axes[0].set_xticks(range(error_matrix.shape[1]))
    axes[0].set_xticklabels(error_matrix.columns)
    axes[0].set_yticks(range(error_matrix.shape[0]))
    axes[0].set_yticklabels([f"Pclass {value}" for value in error_matrix.index])
    for row in range(error_matrix.shape[0]):
        for column in range(error_matrix.shape[1]):
            value = error_values[row, column]
            axes[0].text(
                column,
                row,
                f"{value:.2f}\\nn={int(count_matrix.to_numpy()[row, column])}",
                ha="center",
                va="center",
                fontsize=9,
                # 濃いセルでは黒字が読めないため、上位の値だけ白字にする。
                color="white" if value > 0.6 * highest_error else COLORS["ink"],
            )
    axes[0].set_title("Out-of-fold error rate by class and sex")
    axes[0].grid(False)
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    error_types = (
        oof.loc[oof["error_type"] != "correct"]
        .groupby(["Pclass", "Sex", "error_type"], observed=True)
        .size()
        .reset_index(name="count")
    )
    error_types["segment"] = (
        "Pclass " + error_types["Pclass"].astype(str) + ", " + error_types["Sex"].astype(str)
    )
    pivot_types = (
        error_types.pivot_table(
            index="segment",
            columns="error_type",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
    )
    for column, color in [
        ("missed survivor", COLORS["blue"]),
        ("false survivor", COLORS["gold"]),
    ]:
        if column not in pivot_types.columns:
            pivot_types[column] = 0

    segment_positions = np.arange(len(pivot_types))
    axes[1].barh(
        segment_positions,
        pivot_types["missed survivor"],
        color=COLORS["blue"],
        label="missed survivor",
    )
    axes[1].barh(
        segment_positions,
        pivot_types["false survivor"],
        left=pivot_types["missed survivor"],
        color=COLORS["gold"],
        label="false survivor",
    )
    axes[1].set_yticks(segment_positions)
    axes[1].set_yticklabels(pivot_types.index)
    axes[1].set_xlabel("Number of out-of-fold errors")
    axes[1].set_title("Which direction the errors take")
    axes[1].legend(loc="lower right")

    figure.tight_layout()
    plt.show()

    display(error_grid.round(4))
    """
)

md(
    """
    誤分類率だけでなく、誤りの向きも分けて見ます。
    生存者を死亡と予測する誤り（missed survivor）と、その逆（false survivor）は、
    同じ1件でも意味が異なるためです。

    最後に、予測確率が観測比率と対応しているかを確認します。
    決定木の予測確率は、葉に含まれる訓練データの生存比率です。深さ3の木ひとつなら値は葉の数だけですが、
    ここではOOF予測を使うため、5つのfoldそれぞれの木が出した値が混ざります。
    予測確率が連続的に分布せず、いくつかの値に固まって現れるのは、この構造によります。

    各点にはWilson区間を付けます。乗客が数人しかいない確率帯では、
    観測比率が対角線から外れて見えても、その差を較正のずれと呼べないためです。

    > **English:** Tree probabilities are leaf frequencies, so the values are discrete. Because these
    > are out-of-fold predictions, the distinct values pool the leaves of five fold-specific trees.
    > Wilson intervals mark the bands that hold only a handful of passengers.
    """
)

code(
    """
    calibration = (
        oof.assign(probability_bin=oof["predicted_probability"].round(2))
        .groupby("probability_bin", observed=True)
        .agg(
            observed_rate=(TARGET_COLUMN, "mean"),
            survivors=(TARGET_COLUMN, "sum"),
            passengers=(TARGET_COLUMN, "size"),
        )
        .reset_index()
    )
    calibration_bounds = [
        wilson_interval(int(row.survivors), int(row.passengers))
        for row in calibration.itertuples()
    ]
    calibration["ci_low"] = [bound[0] for bound in calibration_bounds]
    calibration["ci_high"] = [bound[1] for bound in calibration_bounds]

    figure, axis = plt.subplots(figsize=(7.2, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color=COLORS["muted"], label="perfect calibration")
    axis.errorbar(
        calibration["probability_bin"],
        calibration["observed_rate"],
        yerr=error_bar_widths(
            calibration["observed_rate"],
            calibration["ci_low"],
            calibration["ci_high"],
        ),
        fmt="none",
        ecolor=COLORS["muted"],
        elinewidth=1.6,
        capsize=3,
    )
    axis.scatter(
        calibration["probability_bin"],
        calibration["observed_rate"],
        s=18 + 6 * np.sqrt(calibration["passengers"]),
        color=COLORS["blue"],
        zorder=3,
        label="observed rate (marker size = n)",
    )
    # 人数はマーカーの大きさと下の表で読めるため、図には注記を重ねない。
    axis.set_xlim(-0.05, 1.05)
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel("Predicted probability (out-of-fold)")
    axis.set_ylabel("Observed survival rate")
    axis.set_title("Leaf frequencies pooled over 5 folds, read with their denominators")
    axis.legend(loc="upper left")
    figure.tight_layout()
    plt.show()

    display(calibration.round(4))
    """
)

md(
    """
    **観察の記録欄**

    | 項目 | 記入 |
    |---|---|
    | 誤分類率が最も高かった層と、その人数 | **［自分で記入］** |
    | 誤りの向きに偏りがあった層 | **［自分で記入］** |
    | 予測確率と観測比率がずれた確率帯 | **［自分で記入］** |
    """
)

md(
    """
    ## 6. 提出ファイルを作る / Build the submission

    比較と観察を終えてから、train全体で学習し、testを予測します。
    交差検証では各foldの学習部分だけで前処理をfitしました。提出用モデルでも同じPipelineを使い、
    補完とOne-Hotの基準はtrainからのみ得ます。
    """
)

code(
    """
    final_pipeline = build_baseline_pipeline()
    final_pipeline.fit(X, y)
    test_prediction = final_pipeline.predict(X_test)

    submission = pd.DataFrame({
        ID_COLUMN: test[ID_COLUMN],
        TARGET_COLUMN: test_prediction.astype(int),
    })

    submission_filename = (
        "submission.csv"
        if ENVIRONMENT == "Kaggle"
        else "submission_uncertainty_error.csv"
    )
    submission_path = OUTPUT_DIR / submission_filename
    submission.to_csv(submission_path, index=False)

    assert list(submission.columns) == [ID_COLUMN, TARGET_COLUMN]
    assert submission[ID_COLUMN].is_unique
    assert submission[ID_COLUMN].tolist() == test[ID_COLUMN].tolist()
    assert submission[TARGET_COLUMN].notna().all()
    assert set(submission[TARGET_COLUMN].unique()).issubset({0, 1})
    if IS_OFFICIAL_SHAPE:
        assert len(submission) == 418

    submission_checks = pd.DataFrame({
        "check": [
            "rows",
            "columns",
            "missing predictions",
            "predicted survivors",
            "predicted survival rate",
            "output path",
        ],
        "value": [
            len(submission),
            ", ".join(submission.columns),
            int(submission[TARGET_COLUMN].isna().sum()),
            int(submission[TARGET_COLUMN].sum()),
            f"{submission[TARGET_COLUMN].mean():.4f}",
            str(submission_path),
        ],
    })
    display(submission_checks)
    display(submission.head())
    """
)

md(
    """
    ## 7. 実行結果から読めること / Executed takeaways

    次のセルは、上の実行値から文章を生成します。小さな差を順位として扱わないようにします。
    """
)

code(
    """
    widest = rate_tables.loc[rate_tables["ci_width"].idxmax()]
    narrowest = rate_tables.loc[rate_tables["ci_width"].idxmin()]
    sex_comparison, embarked_comparison = comparisons
    age_missing_row = age_missing_rates.loc[
        age_missing_rates["group"] == "AgeRecorded=Age missing"
    ].iloc[0]
    worst_segment = error_grid.loc[error_grid["error_rate"].idxmax()]
    missed = int((oof["error_type"] == "missed survivor").sum())
    false_positive = int((oof["error_type"] == "false survivor").sum())

    display(Markdown(
        f"- 生存率の区間幅は、`{narrowest['group']}`（n={int(narrowest['passengers'])}）の"
        f"`{narrowest['ci_width']:.3f}`から、`{widest['group']}`（n={int(widest['passengers'])}）の"
        f"`{widest['ci_width']:.3f}`まで開きがあった。  \\n"
        f"- 性別による差はブートストラップ95%区間が"
        f"`[{sex_comparison['ci_low']:.3f}, {sex_comparison['ci_high']:.3f}]`で、0を含まなかった。  \\n"
        f"- `Embarked` Q と S の差は`[{embarked_comparison['ci_low']:.3f}, "
        f"{embarked_comparison['ci_high']:.3f}]`で、0を"
        f"{'含んだ' if embarked_comparison['crosses_zero'] else '含まなかった'}。"
        f"この設計では差を確定できない。  \\n"
        f"- `Age`欠損は{len(dropped)}行あり、その群の生存率は`{age_missing_row['rate']:.3f}`"
        f"（95%区間`[{age_missing_row['ci_low']:.3f}, {age_missing_row['ci_high']:.3f}]`）で、"
        f"全体`{train[TARGET_COLUMN].mean():.3f}`より低い側に位置した。  \\n"
        f"- 深さを1から10と制限なしまで動かしたとき、検証正解率の平均の最大値は"
        f"`{best_validation_mean:.4f}`で、`max_depth={'`, `'.join(tied_best_labels)}`が同値だった。"
        f"深さ{BASELINE_DEPTH}との差が0を跨がなかったのは"
        f"`{'`, `'.join(separated_labels)}`のみで、いずれも深さ{BASELINE_DEPTH}より低かった。  \\n"
        f"- OOF正解率は`{accuracy_score(y, oof_prediction):.4f}`だが、誤りは均一ではなく、"
        f"`Pclass {int(worst_segment['Pclass'])}, {worst_segment['Sex']}`"
        f"（n={int(worst_segment['passengers'])}）で`{worst_segment['error_rate']:.3f}`と最も高かった。  \\n"
        f"- 誤りの内訳は missed survivor `{missed}`件、false survivor `{false_positive}`件だった。  \\n\\n"
        "> **English:** Interval widths differ by an order of magnitude across groups, one of the two "
        "compared gaps survives resampling, dropping missing-Age rows removes a group with a lower "
        "observed survival rate, and the baseline's errors concentrate in specific class-sex segments."
    ))
    """
)

md(
    """
    ## 8. 限界 / Limits

    - 誤りの偏りと較正は`random_state=42`の層化5-foldのみで見ています。深さの比較だけが5 seed × 5-foldです。
    - ブートストラップ区間は再標本化に基づく近似で、標本自体の偏りは補正しません。
    - 深さは1から10と制限なしまで比べましたが、決定木以外のモデルとは比較していません。
      深さの採用もこのNotebookでは決めていません。
    - 深さの比較に使った25 foldは学習行が重複するため、独立した25標本として有意差検定には使えません。
    - 欠損の構造は観察であり、補完方法の優劣を決める根拠にはしていません。
    - 891行という規模では、層別に分けるほど各セルの分母が小さくなります。層別の数値は区間つきで読む必要があります。

    > **English:** Single seed, no model comparison, and small per-segment denominators. The notebook
    > documents where judgement should be suspended rather than resolving these points.
    """
)

md(
    """
    ## 9. 実行後に自分で書く / Written after the run

    1. **実行前の予想と外れた点:** ［自分で記入］
    2. **区間を見たことで保留にした判断:** ［自分で記入］
    3. **欠損処理について決めたこと、決めなかったこと:** ［自分で記入］
    4. **誤りの偏りから立てた次の仮説:** ［自分で記入］
    5. **その仮説を棄却する条件:** ［自分で記入］

    ### 次の一回一変更テンプレート

    | 項目 | 記録 |
    |---|---|
    | 仮説 | **［自分で記入］** |
    | 固定するデータ・分割・指標・モデル | **［自分で記入］** |
    | 一つだけ変えるもの | **［自分で記入］** |
    | 改善とみなす条件 | **［自分で記入］** |
    | 悪化・無効とみなす条件 | **［自分で記入］** |

    ## 10. AI支援の範囲 / AI assistance

    コードの雛形、実行検査、Notebookの構成整理にはAIを利用しました。
    モデルと特徴量は、このプロジェクトで自分が先に実行した決定木ベースラインを引き継いだもので、
    このNotebookでAIが新しいモデル候補を選んだわけではありません。
    観察の解釈、次の仮説、採否の判断は、実行結果を見たうえで実行者が書きます。

    > **English:** AI assisted with scaffolding, execution checks, and structure. The model and
    > features were carried over from the author's earlier baseline, and the interpretation and next
    > hypothesis are written by the author.
    """
)

notebook["cells"] = cells
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.12",
    },
}

nbf.validate(notebook)
nbf.write(notebook, OUTPUT_PATH)
print(f"Wrote {OUTPUT_PATH} with {len(cells)} cells.")
