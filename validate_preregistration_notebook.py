from pathlib import Path
import base64

import nbformat
import pandas as pd
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = (
    ROOT / "preregistration_publish" / "titanic_preregistered_comparisons.ipynb"
)
REVIEW_DIR = ROOT / "preregistration_review"
REVIEW_DIR.mkdir(exist_ok=True)

notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
nbformat.validate(notebook)

client = NotebookClient(
    notebook,
    timeout=3_600,
    kernel_name="python3",
    allow_errors=False,
    resources={"metadata": {"path": str(ROOT)}},
)
executed = client.execute()
nbformat.validate(executed)
nbformat.write(executed, NOTEBOOK_PATH)

code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
executed_code_cells = [
    cell for cell in code_cells if cell.get("execution_count") is not None
]
error_outputs = [
    output
    for cell in code_cells
    for output in cell.get("outputs", [])
    if output.get("output_type") == "error"
]
image_outputs = [
    output
    for cell in code_cells
    for output in cell.get("outputs", [])
    if "image/png" in output.get("data", {})
]

submission_path = (
    ROOT / "preregistration_publish" / "outputs" / "submission_preregistered.csv"
)
if not submission_path.exists():
    raise FileNotFoundError(f"The notebook did not produce {submission_path}.")
submission = pd.read_csv(submission_path)

assert len(executed_code_cells) == len(code_cells), "Some code cells were not executed."
assert not error_outputs, "The notebook contains error outputs."
assert len(image_outputs) >= 7, f"Expected at least 7 figures, found {len(image_outputs)}."
assert len(submission) == 418, "Unexpected submission row count."
assert list(submission.columns) == ["PassengerId", "Survived"]
assert submission["PassengerId"].is_unique
assert submission.isna().sum().sum() == 0
assert set(submission["Survived"].unique()).issubset({0, 1})

baseline = ROOT / "submissions" / "decision_tree_depth3_baseline.csv"
if baseline.exists():
    reference = pd.read_csv(baseline)
    identical = reference["Survived"].equals(submission["Survived"])
    print("submitted baseline と同一:", identical)

for image_number, output in enumerate(image_outputs, start=1):
    (REVIEW_DIR / f"figure_{image_number:02d}.png").write_bytes(
        base64.b64decode(output["data"]["image/png"])
    )

print("notebook:", NOTEBOOK_PATH.name)
print("cells:", len(executed.cells))
print("executed code cells:", f"{len(executed_code_cells)}/{len(code_cells)}")
print("figures:", len(image_outputs))
print("error outputs:", len(error_outputs))
print("submission:", submission_path.name, f"({len(submission)} rows)")
print("review images:", REVIEW_DIR)
