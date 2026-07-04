import unittest
from unittest.mock import patch

from canon.generation.providers import TemplateGenerationProvider, get_generation_provider


class GenerationTests(unittest.TestCase):
    def test_template_provider_returns_prompt(self):
        provider = TemplateGenerationProvider()
        result = provider.generate("grounded answer")
        self.assertEqual(result.provider, "template")
        self.assertEqual(result.text, "grounded answer")

    def test_openai_generation_requires_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError):
                get_generation_provider("openai")

    def test_unknown_generation_provider_fails(self):
        with self.assertRaises(ValueError):
            get_generation_provider("missing")


if __name__ == "__main__":
    unittest.main()
