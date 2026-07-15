import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from canon import secrets


class LocalSecretsTests(unittest.TestCase):
    def setUp(self):
        secrets.LOADED = False

    def tearDown(self):
        secrets.LOADED = False

    def test_load_local_env_loads_allowed_keys_only(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY='openai-test'",
                        "COHERE_API_KEY=cohere-test",
                        "UNRELATED_SECRET=do-not-load",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                secrets.load_local_env(env_path)

                self.assertEqual(os.environ["OPENAI_API_KEY"], "openai-test")
                self.assertEqual(os.environ["COHERE_API_KEY"], "cohere-test")
                self.assertNotIn("UNRELATED_SECRET", os.environ)

    def test_load_local_env_does_not_override_existing_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("OPENAI_API_KEY=file-value\n", encoding="utf-8")
            with patch.dict(os.environ, {"OPENAI_API_KEY": "existing-value"}, clear=True):
                secrets.load_local_env(env_path)

                self.assertEqual(os.environ["OPENAI_API_KEY"], "existing-value")


if __name__ == "__main__":
    unittest.main()
