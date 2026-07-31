from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "titanic_first_principles_learning_journal.ipynb"

notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
nbformat.validate(notebook)

client = NotebookClient(
    notebook,
    timeout=180,
    kernel_name="python3",
    allow_errors=False,
    resources={"metadata": {"path": str(ROOT)}},
)
executed = client.execute()
nbformat.validate(executed)
nbformat.write(executed, NOTEBOOK_PATH)

code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
executed_code_cells = [cell for cell in code_cells if cell.execution_count is not None]
error_outputs = [
    output
    for cell in code_cells
    for output in cell.get("outputs", [])
    if output.output_type == "error"
]
image_outputs = [
    output
    for cell in code_cells
    for output in cell.get("outputs", [])
    if output.output_type in {"display_data", "execute_result"}
    and "image/png" in output.get("data", {})
]

official_submission = ROOT / "submissions" / "submission_first_principles.csv"
smoke_submission = ROOT / "submissions" / "submission_first_principles_smoke.csv"
submission_path = official_submission if official_submission.exists() else smoke_submission

if not submission_path.exists():
    raise FileNotFoundError("The notebook did not produce an expected submission file.")

submission = pd.read_csv(submission_path)
expected_rows = 418 if submission_path == official_submission else len(submission)

assert len(executed_code_cells) == len(code_cells), "Some code cells were not executed."
assert not error_outputs, "The notebook contains error outputs."
assert len(image_outputs) >= 8, "Expected visual outputs were not rendered."
assert len(submission) == expected_rows, "Unexpected submission row count."
assert list(submission.columns) == ["PassengerId", "Survived"], "Unexpected submission schema."
assert submission["PassengerId"].is_unique, "Duplicate PassengerId values."
assert submission.isna().sum().sum() == 0, "Missing values in submission."
assert set(submission["Survived"].unique()).issubset({0, 1}), "Predictions are not binary."

print(f"Notebook: {NOTEBOOK_PATH.name}")
print(f"Cells: {len(executed.cells)}")
print(f"Code cells executed: {len(executed_code_cells)}/{len(code_cells)}")
print(f"PNG outputs: {len(image_outputs)}")
print(f"Errors: {len(error_outputs)}")
print(f"Submission: {submission_path.name} ({len(submission)} rows)")
