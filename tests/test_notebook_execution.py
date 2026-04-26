from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(
    os.environ.get("RUN_NOTEBOOK_INTEGRATION") == "1",
    "Notebook integration tests are disabled by default. Set RUN_NOTEBOOK_INTEGRATION=1 to execute them.",
)
class NotebookExecutionSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.notebooks_dir = cls.repo_root / "notebooks"

    def _execute_notebook(self, notebook_name: str) -> None:
        notebook_path = self.notebooks_dir / notebook_name
        self.assertTrue(notebook_path.exists(), f"Notebook not found: {notebook_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            output_name = f"executed_{notebook_name}"
            command = [
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                str(notebook_path),
                "--output",
                output_name,
                "--output-dir",
                str(output_dir),
                "--ExecutePreprocessor.timeout=1800",
            ]
            subprocess.run(
                command,
                cwd=self.repo_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            executed_path = output_dir / output_name
            self.assertTrue(executed_path.exists(), f"Executed notebook was not created: {executed_path}")

    def test_feature_engineering_notebook_executes(self) -> None:
        self._execute_notebook("04_feature_engineering.ipynb")

    def test_at_risk_modeling_notebook_executes(self) -> None:
        self._execute_notebook("06_at_risk_modeling.ipynb")


if __name__ == "__main__":
    unittest.main()
