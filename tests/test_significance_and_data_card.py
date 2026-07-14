import unittest

from canon.eval.significance import paired_significance_report, sign_flip_p_value
from canon.reports.data_card import coverage, chunking_profile, document_type_profile_summary, domain_profile_summary, limitations


class SignificanceAndDataCardTests(unittest.TestCase):
    def test_paired_significance_detects_consistent_advantage(self):
        report = paired_significance_report(
            {
                "benchmark_id": "toy",
                "mode": "m",
                "method_set_id": "methods",
                "method_summary": {"a": {}, "b": {}},
                "queries": [
                    {
                        "method_runs": [
                            {"method_id": "a", "metrics": {"nDCG@10": 1.0}},
                            {"method_id": "b", "metrics": {"nDCG@10": 0.0}},
                        ]
                    },
                    {
                        "method_runs": [
                            {"method_id": "a", "metrics": {"nDCG@10": 0.8}},
                            {"method_id": "b", "metrics": {"nDCG@10": 0.1}},
                        ]
                    },
                ],
            },
            metric="nDCG@10",
            samples=50,
            seed=3,
        )
        comparison = report["comparisons"][0]
        self.assertEqual(comparison["direction"], "left_better")
        self.assertGreater(comparison["bootstrap_ci"]["low"], 0.0)

    def test_sign_flip_p_value_is_bounded(self):
        p_value = sign_flip_p_value([0.2, -0.1, 0.3], samples=50, seed=5)
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)

    def test_data_card_coverage_counts_fields(self):
        works = [
            {
                "abstract": "A",
                "is_open_access": True,
                "pdf_url": "https://example.org/a.pdf",
                "referenced_work_count": 3,
                "author_display_names": ["A"],
                "max_author_cited_by_count": 9,
            },
            {"abstract": "", "is_open_access": False},
        ]
        cov = coverage(works, [{"id": "c1"}, {"id": "c2"}])
        self.assertEqual(cov["abstract_coverage"], 0.5)
        self.assertEqual(cov["chunks_per_work"], 1.0)

    def test_chunking_profile_summarizes_boundary_alignment(self):
        profile = chunking_profile(
            [
                {"token_start": 0, "token_end": 10, "importance": {"chunk_token_count": 10, "boundary_aligned": 1.0}},
                {"token_start": 8, "token_end": 18, "importance": {"boundary_aligned": 0.0}},
            ]
        )

        self.assertEqual(profile["chunk_count"], 2)
        self.assertEqual(profile["average_chunk_tokens"], 10.0)
        self.assertEqual(profile["boundary_aligned_share"], 0.5)
        self.assertEqual(profile["token_window_fallback_share"], 0.5)

    def test_chunking_profile_summarizes_decision_metadata(self):
        profile = chunking_profile(
            [
                {
                    "token_start": 0,
                    "token_end": 10,
                    "importance": {
                        "chunk_token_count": 10,
                        "boundary_aligned": 1.0,
                        "chunk_resolution": "evidence_window",
                        "parent_context_mode": "section",
                        "generation_context_role": "evidence_context",
                        "parent_context_recommended": 1.0,
                        "chunk_safety_contamination_risk": 0.1,
                        "evidence_containment_score": 0.9,
                        "chunk_quality_tier": "strong_boundary_context",
                        "context_expansion_policy": "safe_parent_expansion_allowed",
                    },
                },
                {
                    "token_start": 10,
                    "token_end": 20,
                    "importance": {
                        "chunk_token_count": 10,
                        "boundary_aligned": 1.0,
                        "chunk_resolution": "atomic_public_expression",
                        "parent_context_mode": "thread_or_post",
                        "generation_context_role": "aggregate_only",
                        "parent_context_recommended": 0.0,
                        "chunk_safety_contamination_risk": 0.7,
                        "evidence_containment_score": 0.8,
                        "chunk_quality_tier": "safety_sensitive",
                        "context_expansion_policy": "no_parent_expansion_aggregate_only",
                    },
                },
            ]
        )

        self.assertEqual(profile["chunk_resolution_counts"]["evidence_window"], 1)
        self.assertEqual(profile["generation_context_role_counts"]["aggregate_only"], 1)
        self.assertEqual(profile["parent_context_recommended_share"], 0.5)
        self.assertEqual(profile["average_safety_contamination_risk"], 0.4)
        self.assertEqual(profile["average_evidence_containment_score"], 0.85)
        self.assertEqual(profile["chunk_quality_tier_counts"]["safety_sensitive"], 1)
        self.assertEqual(profile["context_expansion_policy_counts"]["safe_parent_expansion_allowed"], 1)

    def test_data_card_limitations_flag_small_corpus(self):
        limits = limitations([], [], {})
        self.assertIn("Corpus is small; results are suitable for technical testing, not field-level claims.", limits)

    def test_data_card_document_type_profile_summarizes_mixed_corpus(self):
        summary = document_type_profile_summary(
            [
                work("Academic Battery Study", "Journal", {"source_type": "journal"}, doi="10.1/example", referenced_work_count=20),
                work("Company 10-K Annual Report", "SEC EDGAR", {"document_type": "regulatory_filing"}),
                work("Battery Storage API Tips", "SEO Blog", {"source_type": "blog", "provenance": "web"}),
            ]
        )
        self.assertEqual(summary["document_type_counts"]["academic_article"], 1)
        self.assertEqual(summary["document_type_counts"]["regulatory_filing"], 1)
        self.assertEqual(summary["document_type_counts"]["web_page"], 1)
        self.assertGreater(summary["grey_literature_share"], 0.0)
        self.assertGreater(summary["average_adversarial_surface"], 0.0)

    def test_data_card_domain_profile_summarizes_domain_fit(self):
        summary = domain_profile_summary(
            [
                work(
                    "Economics Identification Study",
                    "Journal",
                    {"source_type": "journal", "domain": "economics"},
                    doi="10.1/econ",
                    referenced_work_count=30,
                ),
                work(
                    "GridCo 10-K Annual Report",
                    "SEC EDGAR",
                    {"document_type": "regulatory_filing", "domain": "legal_market", "provenance": "official"},
                ),
            ]
        )
        self.assertEqual(summary["domain_counts"]["economics"], 1)
        self.assertEqual(summary["domain_counts"]["legal_market"], 1)
        self.assertIn("average_domain_fit_score", summary)
        self.assertLess(summary["general_academic_share"], 1.0)

    def test_data_card_limitations_flag_low_document_type_diversity(self):
        limits = limitations(
            [work("Unknown A", None, {}, landing_page_url=None), work("Unknown B", None, {}, landing_page_url=None)],
            [{"id": "c1"}],
            {},
        )
        self.assertTrue(any("Document-type diversity is low" in item for item in limits))

    def test_data_card_limitations_use_public_opinion_standards(self):
        limits = limitations(
            [
                work(
                    "Public post",
                    "MicroBlog",
                    {
                        "document_type": "social_media_post",
                        "domain": "public_opinion",
                        "provenance": "social_media",
                        "platform": "MicroBlog",
                        "visibility": "public",
                    },
                ),
                work(
                    "Forum comment",
                    "Forum",
                    {
                        "document_type": "forum_comment",
                        "domain": "public_opinion",
                        "provenance": "comment",
                        "platform": "Forum",
                        "visibility": "public",
                    },
                ),
            ],
            [{"id": "c1"}, {"id": "c2"}],
            {},
        )

        self.assertFalse(any("PDF URL coverage is low" in item for item in limits))
        self.assertFalse(any("Reference-list coverage is incomplete" in item for item in limits))
        self.assertTrue(any("expression evidence" in item for item in limits))
        self.assertTrue(any("requires aggregation" in item for item in limits))


def work(title, source_name, raw, doi=None, referenced_work_count=0, landing_page_url="https://example.test"):
    return {
        "id": title,
        "doi": doi,
        "title": title,
        "year": 2026,
        "language": "en",
        "abstract": title,
        "source_name": source_name,
        "is_open_access": True,
        "is_retracted": False,
        "cited_by_count": 0,
        "referenced_work_count": referenced_work_count,
        "pdf_url": None,
        "landing_page_url": landing_page_url,
        "raw": raw,
    }


if __name__ == "__main__":
    unittest.main()
