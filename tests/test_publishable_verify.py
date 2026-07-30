import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from canon.product.publishable_verify import verify_publishable_artifacts


class PublishableVerifyTests(unittest.TestCase):
    def test_verifier_passes_when_manifest_hashes_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            artifact = root / "reports" / "artifact.json"
            artifact.write_text('{"ok": true}\n', encoding="utf-8")
            package = write_package(reports, [manifest_row(root, artifact)])
            settings = SimpleNamespace(root=root, reports_dir=reports)
            with patch("canon.product.publishable_verify.load_settings", return_value=settings):
                report = verify_publishable_artifacts(package_path=package, write_report=False)

        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["verified_count"], 1)
        self.assertEqual(report["rows"][0]["status"], "verified")

    def test_verifier_reports_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            artifact = root / "reports" / "artifact.json"
            artifact.write_text("before\n", encoding="utf-8")
            package = write_package(reports, [manifest_row(root, artifact)])
            artifact.write_text("after\n", encoding="utf-8")
            settings = SimpleNamespace(root=root, reports_dir=reports)
            with patch("canon.product.publishable_verify.load_settings", return_value=settings):
                report = verify_publishable_artifacts(package_path=package, write_report=False)

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["mismatch_count"], 1)
        self.assertEqual(report["rows"][0]["status"], "mismatch")

    def test_verifier_accepts_zero_byte_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            artifact = root / "reports" / "empty.json"
            artifact.write_bytes(b"")
            package = write_package(reports, [manifest_row(root, artifact)])
            settings = SimpleNamespace(root=root, reports_dir=reports)
            with patch("canon.product.publishable_verify.load_settings", return_value=settings):
                report = verify_publishable_artifacts(package_path=package, write_report=False)

        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["verified_count"], 1)
        self.assertEqual(report["rows"][0]["actual_bytes"], 0)

    def test_verifier_reports_missing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            artifact = root / "reports" / "artifact.json"
            artifact.write_text("data\n", encoding="utf-8")
            row = manifest_row(root, artifact)
            artifact.unlink()
            package = write_package(reports, [row])
            settings = SimpleNamespace(root=root, reports_dir=reports)
            with patch("canon.product.publishable_verify.load_settings", return_value=settings):
                report = verify_publishable_artifacts(package_path=package, write_report=False)

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["missing_count"], 1)
        self.assertEqual(report["rows"][0]["status"], "missing")

    def test_verifier_rejects_manifest_paths_outside_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            package = write_package(
                reports,
                [
                    {
                        "path": "../outside.json",
                        "sha256": "abc",
                        "bytes": 1,
                    }
                ],
            )
            settings = SimpleNamespace(root=root, reports_dir=reports)
            with patch("canon.product.publishable_verify.load_settings", return_value=settings):
                report = verify_publishable_artifacts(package_path=package, write_report=False)

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["unsafe_path_count"], 1)
        self.assertEqual(report["rows"][0]["status"], "unsafe_path")


def write_package(reports: Path, artifacts: list[dict]) -> Path:
    path = reports / "package.json"
    path.write_text(
        json.dumps(
            {
                "report_id": "publishable_evaluation_package_v1",
                "status": "publication_blocked_evidence_incomplete",
                "suite": {"suite_id": "unit_suite"},
                "artifact_manifest": {"artifacts": artifacts},
            }
        ),
        encoding="utf-8",
    )
    return path


def manifest_row(root: Path, path: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


if __name__ == "__main__":
    unittest.main()
