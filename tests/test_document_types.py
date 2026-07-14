import unittest

from canon.models import Work
from canon.quality.document_types import document_type_profile
from canon.retrieval.corpus import document_from_chunk


class DocumentTypeProfileTests(unittest.TestCase):
    def test_academic_article_profile_uses_scholarly_signals(self):
        profile = document_type_profile(
            work(
                title="A Field Experiment on Labor Markets",
                doi="10.123/example",
                source_name="Journal of Economics",
                referenced_work_count=35,
                raw={"source_type": "journal"},
            )
        )
        self.assertEqual(profile.document_type, "academic_article")
        self.assertFalse(profile.grey_literature)
        self.assertGreater(profile.authority_signal, profile.adversarial_surface)

    def test_open_web_profile_flags_adversarial_surface(self):
        profile = document_type_profile(
            work(
                title="Battery market secrets",
                source_name="SEO Blog",
                landing_page_url="https://example.test/blog",
                raw={"provenance": "web", "source_type": "blog"},
            )
        )
        self.assertEqual(profile.document_type, "web_page")
        self.assertTrue(profile.grey_literature)
        self.assertGreaterEqual(profile.adversarial_surface, 0.7)

    def test_legal_authority_profile_requires_authority_context(self):
        profile = document_type_profile(
            work(
                title="Example v. State Supreme Court Opinion",
                source_name="State Supreme Court",
                raw={"document_type": "court_opinion"},
            )
        )
        self.assertEqual(profile.document_type, "legal_authority")
        self.assertEqual(profile.recommended_decision_role, "gate_by_jurisdiction_and_authority")
        self.assertGreaterEqual(profile.timeliness_need, 0.8)

    def test_explicit_preprint_beats_reference_count_academic_fallback(self):
        profile = document_type_profile(
            work(
                title="Working Paper on Battery Storage Adoption",
                source_name="SSRN",
                referenced_work_count=18,
                raw={"document_type": "preprint", "source_type": "preprint"},
            )
        )

        self.assertEqual(profile.document_type, "preprint")
        self.assertEqual(profile.recommended_decision_role, "annotate_review_status")

    def test_explicit_policy_report_beats_reference_count_academic_fallback(self):
        profile = document_type_profile(
            work(
                title="Agency Report on Storage Reliability",
                source_name="Energy Policy Agency",
                referenced_work_count=12,
                raw={"document_type": "policy_report", "source_type": "policy_report"},
            )
        )

        self.assertEqual(profile.document_type, "policy_report")
        self.assertEqual(profile.evidence_family, "institutional_report")

    def test_explicit_market_report_routes_to_market_profile(self):
        profile = document_type_profile(
            work(
                title="Analyst Report on Grid Storage Demand",
                source_name="NorthBridge Markets",
                raw={"document_type": "market_report", "source_type": "market_report"},
            )
        )

        self.assertEqual(profile.document_type, "market_report")
        self.assertEqual(profile.recommended_decision_role, "annotate_incentives_and_corroborate")

    def test_social_public_opinion_profiles_are_aggregate_only(self):
        social = document_type_profile(
            work(
                title="Public post about battery storage",
                source_name="MicroBlog",
                raw={"document_type": "social_media_post", "provenance": "social_media"},
            )
        )
        forum = document_type_profile(
            work(
                title="Forum comment about local planning meeting",
                source_name="Forum",
                raw={"source_type": "forum_comment", "provenance": "comment"},
            )
        )

        self.assertEqual(social.document_type, "social_media_post")
        self.assertEqual(forum.document_type, "forum_comment")
        self.assertEqual(social.evidence_family, "public_opinion")
        self.assertEqual(social.recommended_decision_role, "aggregate_public_opinion_and_screen_safety")
        self.assertGreater(social.adversarial_surface, social.authority_signal)
        self.assertGreater(forum.adversarial_surface, forum.authority_signal)

    def test_explicit_note_beats_incidental_transcript_keyword(self):
        profile = document_type_profile(
            work(
                title="Testing guide",
                abstract="This guide mentions message transcripts as one possible input shape.",
                raw={"document_type": "note", "source_type": "note", "provenance": "user_file"},
            )
        )

        self.assertEqual(profile.document_type, "note")
        self.assertEqual(profile.evidence_family, "user_supplied_text")

    def test_retrieval_document_carries_profile_signals(self):
        doc = document_from_chunk(
            {
                "id": "chunk:filing",
                "work_id": "work:filing",
                "section": "risk factors",
                "text": "The 10-K notes risks from customer concentration.",
                "importance": {},
            },
            {
                "id": "work:filing",
                "doi": None,
                "title": "Company 10-K Annual Report",
                "year": 2026,
                "language": "en",
                "abstract": "Annual report and SEC filing.",
                "source_name": "SEC EDGAR",
                "is_open_access": True,
                "is_retracted": False,
                "cited_by_count": 0,
                "referenced_work_count": 0,
                "pdf_url": None,
                "landing_page_url": "https://sec.example.test",
                "raw": {"document_type": "regulatory_filing", "provenance": "official"},
            },
        )
        self.assertEqual(doc.work_signals["document_type"], "regulatory_filing")
        self.assertTrue(doc.work_signals["grey_literature"])
        self.assertGreater(doc.work_signals["timeliness_need"], 0.8)


def work(**overrides):
    values = {
        "id": "w1",
        "doi": None,
        "title": "Untitled",
        "year": 2024,
        "language": "en",
        "abstract": "",
        "source_name": None,
        "is_open_access": False,
        "is_retracted": False,
        "cited_by_count": 0,
        "referenced_work_count": 0,
        "pdf_url": None,
        "landing_page_url": None,
        "raw": {},
        "author_display_names": [],
        "author_count": 0,
        "max_author_cited_by_count": 0,
        "max_author_works_count": 0,
    }
    values.update(overrides)
    return Work(**values)


if __name__ == "__main__":
    unittest.main()
