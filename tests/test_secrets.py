import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
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
                        "OPENROUTER_API_KEY='openrouter-test'",
                        "COHERE_API_KEY=cohere-test",
                        "OPENAI_API_KEY=ignore-direct-openai",
                        "UNRELATED_SECRET=do-not-load",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                secrets.load_local_env(env_path)

                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "openrouter-test")
                self.assertEqual(os.environ["COHERE_API_KEY"], "cohere-test")
                self.assertNotIn("OPENAI_API_KEY", os.environ)
                self.assertNotIn("UNRELATED_SECRET", os.environ)

    def test_load_local_env_does_not_override_existing_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("OPENROUTER_API_KEY=file-value\n", encoding="utf-8")
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "existing-value"}, clear=True):
                secrets.load_local_env(env_path)

                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "existing-value")

    def test_load_local_env_is_safe_for_parallel_provider_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "OPENROUTER_API_KEY=openrouter-test\nCOHERE_API_KEY=cohere-test\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    list(executor.map(lambda _index: secrets.load_local_env(env_path), range(2)))

                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "openrouter-test")
                self.assertEqual(os.environ["COHERE_API_KEY"], "cohere-test")


if __name__ == "__main__":
    unittest.main()
