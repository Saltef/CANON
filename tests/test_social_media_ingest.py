import json
import tempfile
import unittest
from pathlib import Path

from canon.config import load_settings
from canon.ingest.social_media import (
    PublicPostHTMLParser,
    collection_audit,
    ingest_social_media,
    normalize_records,
)


class SocialMediaIngestTests(unittest.TestCase):
    def test_normalize_records_keeps_only_public_text_and_hashes_author(self):
        records = [
            {
                "id": "post-1",
                "platform": "MicroBlog",
                "visibility": "public",
                "author_id": "person-123",
                "thread_id": "thread-a",
                "text": "Residents want clearer safety plans.",
            },
            {
                "id": "private-1",
                "platform": "MicroBlog",
                "visibility": "private",
                "author_id": "person-private",
                "text": "Do not ingest this.",
            },
            {"id": "empty", "platform": "Forum", "visibility": "public"},
        ]
        normalized, skipped = normalize_records(
            records,
            platform=None,
            collection_method="allowed_public_capture",
        )

        self.assertEqual(len(normalized), 1)
        self.assertEqual(skipped["total"], 2)
        self.assertEqual(skipped["reasons"]["non_public_visibility"], 1)
        self.assertEqual(skipped["reasons"]["missing_text"], 1)
        record = normalized[0]
        self.assertEqual(record["platform"], "MicroBlog")
        self.assertEqual(record["visibility"], "public")
        self.assertEqual(record["thread_id"], "thread-a")
        self.assertEqual(record["document_type"], "social_media_post")
        self.assertNotIn("author_id", record)
        self.assertNotEqual(record["metadata"]["author_hash"], "person-123")

    def test_collection_audit_blocks_overclaiming_without_collection_metadata(self):
        records = [
            {
                "id": "post-1",
                "platform": "MicroBlog",
                "visibility": "public",
                "author_id": "person-123",
                "thread_id": "thread-a",
                "text": "Residents want clearer safety plans.",
            },
            {
                "id": "post-2",
                "platform": "MicroBlog",
                "visibility": "public",
                "email": "person@example.test",
                "text": "Email me about the meeting.",
            },
        ]
        normalized, skipped = normalize_records(records, platform=None, collection_method="allowed_public_capture")
        audit = collection_audit(records, normalized, skipped, collection_method="allowed_public_capture")

        self.assertEqual(audit["status"], "needs_privacy_review")
        self.assertEqual(audit["metadata_coverage"]["terms_review_share"], 0.0)
        self.assertEqual(audit["privacy_risk"]["direct_identifier_record_count"], 1)
        self.assertTrue(any("redact" in action for action in audit["recommended_actions"]))

    def test_html_parser_extracts_public_posts(self):
        html = """
        <article data-canon-social="post" data-id="post-html" data-platform="Forum"
                 data-url="https://example.test/p" data-created-at="2026-01-01T00:00:00Z">
          <span data-canon-social="thread_id">thread-html</span>
          <p data-canon-social="text">Public meeting comments emphasized affordability.</p>
        </article>
        """
        records = PublicPostHTMLParser.parse(html)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["id"], "post-html")
        self.assertEqual(records[0]["platform"], "Forum")
        self.assertEqual(records[0]["thread_id"], "thread-html")
        self.assertIn("affordability", records[0]["text"])

    def test_fixture_ingest_writes_processed_public_opinion_corpus(self):
        settings = load_settings()
        mode = "unit_social_public_opinion"
        report = ingest_social_media(
            settings.fixtures_dir / "social_media_public_opinion_sample.jsonl",
            mode=mode,
            input_format="jsonl",
            chunk_tokens=24,
            overlap_tokens=4,
        )

        self.assertEqual(report["record_count"], 5)
        self.assertEqual(report["normalized_public_record_count"], 4)
        self.assertEqual(report["skipped_reasons"]["non_public_visibility"], 1)
        self.assertEqual(report["platform_counts"]["MicroBlog"], 3)
        self.assertEqual(report["platform_counts"]["Forum"], 1)
        self.assertEqual(report["thread_count"], 3)
        self.assertIn("collection_audit", report)
        self.assertIn(
            report["collection_audit"]["status"],
            {"fixture_only_private_records_skipped", "fixture_only_or_needs_collection_metadata"},
        )
        profile = report["ingest_report"]["document_type_profile"]
        self.assertEqual(profile["document_type_counts"]["social_media_post"], 3)
        self.assertEqual(profile["document_type_counts"]["forum_comment"], 1)
        self.assertGreaterEqual(report["ingest_report"]["chunk_count"], 4)

        works_path = settings.data_dir / "processed" / f"works_{mode}.json"
        works = json.loads(works_path.read_text(encoding="utf-8"))
        self.assertEqual(works[0]["raw"]["domain"], "public_opinion")
        self.assertEqual(works[0]["raw"]["platform"], "MicroBlog")
        self.assertIn("author_hash", works[0]["raw"])
        self.assertNotIn("author_id", works[0]["raw"])

    def test_csv_input_can_be_ingested_with_platform_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "posts.csv"
            path.write_text(
                "id,visibility,text,author_id\n"
                "csv-1,public,People are worried about rents,user-a\n",
                encoding="utf-8",
            )
            report = ingest_social_media(
                path,
                mode="unit_social_csv",
                input_format="csv",
                platform="CivicForum",
                chunk_tokens=24,
                overlap_tokens=4,
            )

        self.assertEqual(report["normalized_public_record_count"], 1)
        self.assertEqual(report["platform_counts"], {"CivicForum": 1})


if __name__ == "__main__":
    unittest.main()
