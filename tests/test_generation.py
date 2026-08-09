import unittest
from unittest.mock import patch

from canon.generation.providers import (
    GenerationProviderError,
    OpenRouterGenerationProvider,
    TemplateGenerationProvider,
    extract_chat_completion_text,
    get_generation_provider,
)


class GenerationTests(unittest.TestCase):
    def test_template_provider_returns_prompt(self):
        provider = TemplateGenerationProvider()
        result = provider.generate("grounded answer")
        self.assertEqual(result.provider, "template")
        self.assertEqual(result.text, "grounded answer")

    def test_openai_alias_generation_requires_openrouter_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("canon.generation.providers.load_local_env"):
                with self.assertRaises(RuntimeError):
                    get_generation_provider("openai")

    def test_openai_alias_generation_routes_through_openrouter(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            with patch("canon.generation.providers.load_local_env"):
                provider = get_generation_provider("openai", "gpt-4.1-mini")
        self.assertEqual(provider.provider, "openrouter")
        self.assertEqual(provider.model, "openai/gpt-4.1-mini")

    def test_openrouter_generation_requires_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("canon.generation.providers.load_local_env"):
                with self.assertRaises(RuntimeError):
                    get_generation_provider("openrouter")

    def test_openrouter_generation_returns_chat_completion_text(self):
        provider = OpenRouterGenerationProvider(model="openai/gpt-4.1-mini", api_key="key")
        payload = {
            "id": "chatcmpl_1",
            "model": "openai/gpt-4.1-mini",
            "choices": [{"message": {"content": "Grid risk should be cited [C1]."}}],
        }

        with patch("canon.generation.providers.post_json", return_value=payload):
            result = provider.generate("Use C1.")

        self.assertEqual(result.provider, "openrouter")
        self.assertEqual(result.model, "openai/gpt-4.1-mini")
        self.assertEqual(result.text, "Grid risk should be cited [C1].")

    def test_openrouter_generation_retries_timeout(self):
        provider = OpenRouterGenerationProvider(model="openai/gpt-4.1-mini", api_key="key", max_attempts=2)
        payload = {
            "id": "chatcmpl_1",
            "model": "openai/gpt-4.1-mini",
            "choices": [{"message": {"content": "Recovered after retry [C1]."}}],
        }

        with patch("canon.generation.providers.post_json", side_effect=[TimeoutError("slow"), payload]) as post_json:
            with patch("canon.generation.providers.time.sleep") as sleep:
                result = provider.generate("Use C1.")

        self.assertEqual(result.text, "Recovered after retry [C1].")
        self.assertEqual(post_json.call_count, 2)
        sleep.assert_called_once()

    def test_openrouter_generation_classifies_auth_failure_without_retry(self):
        class Response:
            status_code = 401
            text = "bad auth"

        class HttpStatusError(Exception):
            response = Response()

        provider = OpenRouterGenerationProvider(model="openai/gpt-4.1-mini", api_key="key", max_attempts=3)

        with patch("canon.generation.providers.post_json", side_effect=HttpStatusError("unauthorized")) as post_json:
            with self.assertRaises(GenerationProviderError) as error:
                provider.generate("Use C1.")

        self.assertEqual(error.exception.failure_type, "auth_failed")
        self.assertEqual(error.exception.http_status, 401)
        self.assertFalse(error.exception.retryable)
        self.assertEqual(error.exception.attempts, 1)
        self.assertEqual(post_json.call_count, 1)

    def test_extract_chat_completion_text_handles_empty_choices(self):
        self.assertEqual(extract_chat_completion_text({"choices": []}), "")

    def test_unknown_generation_provider_fails(self):
        with self.assertRaises(ValueError):
            get_generation_provider("missing")

if __name__ == "__main__":
    unittest.main()
