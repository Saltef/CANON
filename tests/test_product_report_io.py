import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from canon.product import report_io


class ProductReportIoTests(unittest.TestCase):
    def test_write_json_replaces_existing_file_atomically(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "report.json"
            path.write_text('{"status": "old"}\n', encoding="utf-8")

            report_io.write_json(path, {"status": "new"})

            self.assertEqual(path.read_text(encoding="utf-8"), '{\n  "status": "new"\n}\n')
            self.assertEqual(list(Path(temp_dir).glob("*.tmp")), [])

    def test_write_text_atomic_removes_temp_file_when_replace_fails(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "report.md"

            with patch("canon.product.report_io.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    report_io.write_markdown(path, "# Report\n")

            self.assertFalse(path.exists())
            self.assertEqual(list(Path(temp_dir).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
