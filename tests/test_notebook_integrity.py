from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "homework.ipynb"


class NotebookIntegrityTest(unittest.TestCase):
    def test_python_cells_are_complete_and_outputs_are_clean(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        required_cells = {
            8,
            10,
            14,
            18,
            20,
            24,
            28,
            30,
            37,
            41,
            45,
            49,
            53,
            58,
            60,
            64,
            69,
            73,
            75,
        }

        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertEqual(cell.get("outputs", []), [], f"cell {index}")
            self.assertIsNone(cell.get("execution_count"), f"cell {index}")
            source = "".join(cell.get("source", []))
            if index == 2:
                continue
            tree = ast.parse(source, filename=f"homework.ipynb:cell-{index}")
            if index in required_cells:
                self.assertTrue(source.strip(), f"required cell {index} is empty")
                self.assertFalse(
                    any(isinstance(node, ast.Pass) for node in ast.walk(tree)),
                    f"pass remains in cell {index}",
                )
                self.assertFalse(
                    any(
                        isinstance(node, ast.Constant)
                        and node.value is Ellipsis
                        for node in ast.walk(tree)
                    ),
                    f"ellipsis remains in cell {index}",
                )


if __name__ == "__main__":
    unittest.main()
