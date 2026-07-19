from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"


def make_train() -> pd.DataFrame:
    rows = []
    for idx in range(1, 41):
        female = idx % 2 == 0
        pclass = 1 + (idx % 3)
        family = idx % 4
        survived = int(female or (pclass == 1 and family > 0))
        rows.append(
            {
                "PassengerId": idx,
                "Survived": survived,
                "Pclass": pclass,
                "Name": f"Smoke Passenger {idx}",
                "Sex": "female" if female else "male",
                "Age": 18 + (idx % 50),
                "SibSp": family // 2,
                "Parch": family % 2,
                "Ticket": f"SMOKE{idx:03d}",
                "Fare": round(7.25 + pclass * 12 + family * 3.5, 2),
                "Cabin": f"C{idx}" if idx % 5 == 0 else None,
                "Embarked": ["S", "C", "Q"][idx % 3],
            }
        )
    return pd.DataFrame(rows)


def make_test() -> pd.DataFrame:
    rows = []
    for offset, idx in enumerate(range(901, 913), start=1):
        female = offset % 2 == 1
        pclass = 1 + (offset % 3)
        family = offset % 4
        rows.append(
            {
                "PassengerId": idx,
                "Pclass": pclass,
                "Name": f"Smoke Test Passenger {idx}",
                "Sex": "female" if female else "male",
                "Age": 20 + (offset % 45),
                "SibSp": family // 2,
                "Parch": family % 2,
                "Ticket": f"SMOKET{offset:03d}",
                "Fare": round(8.0 + pclass * 11 + family * 4.0, 2),
                "Cabin": f"B{offset}" if offset % 4 == 0 else None,
                "Embarked": ["S", "C", "Q"][offset % 3],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    make_train().to_csv(train_path, index=False)
    make_test().to_csv(test_path, index=False)
    print(f"wrote {train_path}")
    print(f"wrote {test_path}")
    print("This smoke dataset is synthetic and is only for checking that scripts run.")


if __name__ == "__main__":
    main()
