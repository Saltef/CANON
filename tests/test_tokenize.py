from __future__ import annotations

import unittest

from canon.retrieval.tokenize import tokenize


class TokenizeTests(unittest.TestCase):
    def test_tokenize_handles_unicode_words(self) -> None:
        self.assertEqual(tokenize("Müller fragt nach санкции и 制裁."), ["müller", "fragt", "nach", "санкции", "и", "制裁"])

    def test_tokenize_filters_stopwords(self) -> None:
        self.assertEqual(tokenize("what whether how main"), [])

    def test_tokenize_keeps_numbers_and_splits_compounds(self) -> None:
        self.assertEqual(
            tokenize("COVID-19 and model_v2 / 2024 update, don't split."),
            ["covid", "19", "model", "v2", "2024", "update", "don't", "split"],
        )