"""Packaging contracts checked without competition inputs or GPU execution."""
import ast
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import build_notebook


class NotebookPackagingTests(unittest.TestCase):
    def test_generated_notebook_preserves_exact_code_and_compiles(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            build_notebook.build("synthetic-user/demo", "synthetic-user/weights", folder)
            notebook = json.loads((folder / "soil-grain-ml.ipynb").read_text())
            source = (Path(build_notebook.__file__).parent / "soil_model.py").read_text()
            tree = ast.parse("".join(notebook["cells"][1]["source"]))
            embedded = ast.literal_eval(tree.body[1].value.args[0])
            self.assertEqual(embedded, source)
            self.assertEqual(notebook["metadata"]["source_sha256"],
                             hashlib.sha256(source.encode()).hexdigest())
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), "synthetic-notebook", "exec")
                    self.assertEqual(cell["outputs"], [])
                    self.assertIsNone(cell["execution_count"])

    def test_metadata_keeps_gpu_offline_private_and_only_selected_inputs(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            build_notebook.build("synthetic-user/demo", "synthetic-user/weights", folder)
            metadata = json.loads((folder / "kernel-metadata.json").read_text())
            self.assertTrue(metadata["is_private"])
            self.assertTrue(metadata["enable_gpu"])
            self.assertFalse(metadata["enable_internet"])
            self.assertFalse(metadata["enable_tpu"])
            self.assertEqual(metadata["competition_sources"], ["soil-grain-size-from-photos"])
            self.assertEqual(metadata["dataset_sources"], ["synthetic-user/weights"])
            self.assertEqual(metadata["kernel_sources"], [])

    def test_invalid_account_identifiers_fail_before_writing_files(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "output"
            for kernel_id, dataset_id in [("missing-owner", "user/weights"),
                                          ("user/demo", "../private/weights"),
                                          ("user/demo?token=value", "user/weights")]:
                with self.assertRaises(ValueError):
                    build_notebook.build(kernel_id, dataset_id, folder)
                self.assertFalse(folder.exists())


if __name__ == "__main__":
    unittest.main()
