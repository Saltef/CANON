import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from canon.product.publishable_export import export_publishable_bundle


class PublishableExportTests(unittest.TestCase):
    def test_export_copies_verified_package_artifacts_and_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            artifact = root / "gold" / "qrels.json"
            artifact.parent.mkdir()
            artifact.write_text('{"queries": []}\n', encoding="utf-8")
            package = reports / "publishable_package_canon_publishable_evidence_workflow_v1.json"
            package.write_text(
                json.dumps(
                    {
                        "report_id": "publishable_evaluation_package_v1",
                        "status": "publication_blocked_evidence_incomplete",
                        "suite": {"suite_id": "unit_suite"},
                        "artifact_manifest": {"artifacts": [manifest_row(root, artifact)]},
                    }
                ),
                encoding="utf-8",
            )
            output_dir = reports / "bundle"
            archive_path = reports / "bundle.zip"
            settings = SimpleNamespace(root=root, reports_dir=reports)
            with patch("canon.product.publishable_export.load_settings", return_value=settings):
                with patch("canon.product.publishable_verify.load_settings", return_value=settings):
                    report = export_publishable_bundle(
                        package_path=package,
                        output_dir=output_dir,
                        archive_path=archive_path,
                    )

            self.assertEqual(report["status"], "exported")
            self.assertEqual(report["verification_status"], "verified")
            self.assertTrue((output_dir / "gold" / "qrels.json").exists())
            self.assertTrue((output_dir / "reports" / package.name).exists())
            self.assertTrue((output_dir / "BUNDLE_README.md").exists())
            self.assertTrue(archive_path.exists())
            with zipfile.ZipFile(archive_path) as archive:
                names = set(archive.namelist())
            self.assertIn("gold/qrels.json", names)
            self.assertIn(f"reports/{package.name}", names)
            self.assertIn("BUNDLE_README.md", names)

    def test_export_blocks_when_verification_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            artifact = root / "gold" / "qrels.json"
            artifact.parent.mkdir()
            artifact.write_text("before\n", encoding="utf-8")
            row = manifest_row(root, artifact)
            artifact.write_text("after\n", encoding="utf-8")
            package = reports / "publishable_package_canon_publishable_evidence_workflow_v1.json"
            package.write_text(
                json.dumps(
                    {
                        "report_id": "publishable_evaluation_package_v1",
                        "status": "publication_blocked_evidence_incomplete",
                        "suite": {"suite_id": "unit_suite"},
                        "artifact_manifest": {"artifacts": [row]},
                    }
                ),
                encoding="utf-8",
            )
            output_dir = reports / "bundle"
            archive_path = reports / "bundle.zip"
            settings = SimpleNamespace(root=root, reports_dir=reports)
            with patch("canon.product.publishable_export.load_settings", return_value=settings):
                with patch("canon.product.publishable_verify.load_settings", return_value=settings):
                    report = export_publishable_bundle(
                        package_path=package,
                        output_dir=output_dir,
                        archive_path=archive_path,
                    )

            self.assertEqual(report["status"], "export_blocked_verification_failed")
            self.assertEqual(report["verification_status"], "failed")
            self.assertFalse(archive_path.exists())


def manifest_row(root: Path, path: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


if __name__ == "__main__":
    unittest.main()
