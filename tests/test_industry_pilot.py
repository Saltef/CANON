import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from canon.product.industry_pilot import (
    ACCEPTANCE_QUESTIONS,
    build_human_review_tasks,
    build_industry_pilot_report,
    build_review_status_report,
    export_review_csv,
    import_review_csv,
    missing_review_fields,
    query_diagnostics_acceptance,
    query_variants_are_labeled,
    validate_review_record,
    write_json,
)


class IndustryPilotTests(unittest.TestCase):
    def test_industry_pilot_fails_closed_without_review_records(self):
        report = build_industry_pilot_report(write_report=False)
        self.assertEqual(report["status"], "fail")
        self.assertIn("acceptance_question_count", report["release_blockers"])

    def test_industry_pilot_passes_with_full_labeled_acceptance_set(self):
        records = [acceptance_record(index) for index in range(30)]
        report = build_industry_pilot_report_from_records(records)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["release_blockers"], [])

    def test_generated_review_packet_does_not_count_as_reviewed(self):
        packet = build_human_review_tasks(write_report=False, answer_fn=fake_answer)
        report = build_industry_pilot_report_from_records(packet["records"])
        self.assertEqual(report["status"], "fail")
        self.assertIn("acceptance_question_count", report["release_blockers"])

    def test_review_status_report_lists_missing_human_labels(self):
        packet = build_human_review_tasks(write_report=False, answer_fn=fake_answer)
        report = build_review_status_report_from_records(packet["records"])
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["reviewed_question_count"], 0)
        self.assertIn("relevant_evidence_count", report["rows"][0]["missing_fields"])
        self.assertIn("query_lingo_ratings", report["rows"][0]["missing_fields"])

    def test_review_status_report_completes_with_full_human_labels(self):
        records = [acceptance_record(index) for index in range(30)]
        report = build_review_status_report_from_records(records)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["reviewed_question_count"], 30)
        self.assertEqual(report["missing_field_count"], 0)

    def test_missing_review_fields_detects_partial_review(self):
        record = acceptance_record(0)
        record["review"]["citation_support"] = None
        record["review"]["query_lingo_ratings"] = []
        self.assertEqual(
            missing_review_fields(record),
            ["citation_support", "query_lingo_ratings"],
        )

    def test_review_validation_rejects_invalid_enums_and_counts(self):
        record = acceptance_record(0)
        record["review"]["citation_support"] = "pretty good"
        record["review"]["unsupported_claim_count"] = 3
        record["review"]["substantive_claim_count"] = 2
        errors = validate_review_record(record)
        self.assertIn(
            {
                "field": "unsupported_claim_count",
                "message": "cannot exceed substantive_claim_count",
                "value": 3,
            },
            errors,
        )
        self.assertTrue(any(error["field"] == "citation_support" for error in errors))

    def test_industry_pilot_blocks_invalid_review_labels(self):
        records = [acceptance_record(index) for index in range(30)]
        records[0]["review"]["brief_review_minutes"] = -1
        report = build_industry_pilot_report_from_records(records)
        self.assertEqual(report["status"], "fail")
        self.assertIn("review_labels_valid", report["release_blockers"])

    def test_review_csv_export_and_import_round_trip(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tasks.json"
            csv_path = Path(temp_dir) / "reviews.csv"
            output_path = Path(temp_dir) / "completed.json"
            packet = build_human_review_tasks(write_report=False, answer_fn=fake_answer)
            write_json(path, packet)

            export_review_csv(path, csv_path)
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
                fieldnames = list(rows[0].keys())
            self.assertIn("answer", fieldnames)
            self.assertIn("citations", fieldnames)
            self.assertIn("evidence_previews", fieldnames)
            self.assertEqual(rows[0]["answer"], "A cautious cited briefing answer.")
            self.assertIn("Evidence title", rows[0]["citations"])
            for row in rows:
                row.update(
                    {
                        "relevant_evidence_count": "3",
                        "substantive_claim_count": "2",
                        "unsupported_claim_count": "0",
                        "brief_review_minutes": "12.5",
                        "evidence_relevance": "relevant",
                        "citation_support": "strong",
                        "answer_usefulness": "usable",
                        "query_lingo_ratings": "useful;very_useful",
                        "drift_risk_review": "acceptable",
                        "final_review_status": "accepted",
                    }
                )
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            report = import_review_csv(path, csv_path, output_path)
            self.assertEqual(report["records_updated"], 30)
            imported_records = __import__("canon.product.industry_pilot", fromlist=["load_records"]).load_records(output_path)
            self.assertEqual(imported_records[0]["review"]["brief_review_minutes"], 12.5)
            self.assertEqual(imported_records[0]["review"]["query_lingo_ratings"], ["useful", "very_useful"])

    def test_review_csv_import_refuses_invalid_labels_without_writing_output(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tasks.json"
            csv_path = Path(temp_dir) / "reviews.csv"
            output_path = Path(temp_dir) / "completed.json"
            packet = build_human_review_tasks(write_report=False, answer_fn=fake_answer)
            write_json(path, packet)
            export_review_csv(path, csv_path)
            fill_csv_reviews(csv_path, citation_support="pretty good")

            report = import_review_csv(path, csv_path, output_path)
            self.assertEqual(report["status"], "failed_validation")
            self.assertFalse(output_path.exists())
            self.assertTrue(any(error["field"] == "citation_support" for error in report["validation_errors"]))

    def test_review_csv_import_refuses_fractional_integer_counts(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tasks.json"
            csv_path = Path(temp_dir) / "reviews.csv"
            output_path = Path(temp_dir) / "completed.json"
            packet = build_human_review_tasks(write_report=False, answer_fn=fake_answer)
            write_json(path, packet)
            export_review_csv(path, csv_path)
            fill_csv_reviews(csv_path, relevant_evidence_count="2.5")

            report = import_review_csv(path, csv_path, output_path)
            self.assertEqual(report["status"], "failed_validation")
            self.assertFalse(output_path.exists())
            self.assertTrue(any("expected integer" in error["message"] for error in report["validation_errors"]))

    def test_review_csv_import_refuses_duplicate_ids_without_writing_output(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tasks.json"
            csv_path = Path(temp_dir) / "reviews.csv"
            output_path = Path(temp_dir) / "completed.json"
            packet = build_human_review_tasks(write_report=False, answer_fn=fake_answer)
            write_json(path, packet)
            export_review_csv(path, csv_path)
            fill_csv_reviews(csv_path)
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
                fieldnames = list(rows[0].keys())
            rows[1]["id"] = rows[0]["id"]
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            report = import_review_csv(path, csv_path, output_path)
            self.assertEqual(report["status"], "failed_validation")
            self.assertFalse(output_path.exists())
            self.assertTrue(any(error["message"] == "duplicate review id in CSV" for error in report["validation_errors"]))

    def test_review_csv_import_refuses_unknown_ids_without_writing_output(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tasks.json"
            csv_path = Path(temp_dir) / "reviews.csv"
            output_path = Path(temp_dir) / "completed.json"
            packet = build_human_review_tasks(write_report=False, answer_fn=fake_answer)
            write_json(path, packet)
            export_review_csv(path, csv_path)
            fill_csv_reviews(csv_path)
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
                fieldnames = list(rows[0].keys())
            rows[0]["id"] = "stale-question-id"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            report = import_review_csv(path, csv_path, output_path)
            self.assertEqual(report["status"], "failed_validation")
            self.assertFalse(output_path.exists())
            self.assertTrue(
                any(
                    error["message"] == "CSV review id does not exist in the review packet"
                    for error in report["validation_errors"]
                )
            )

    def test_industry_pilot_blocks_unsupported_claims(self):
        records = [acceptance_record(index) for index in range(30)]
        records[0]["review"]["unsupported_claim_count"] = 4
        report = build_industry_pilot_report_from_records(records)
        self.assertEqual(report["status"], "fail")
        self.assertIn("unsupported_claim_rate", report["release_blockers"])

    def test_review_task_packet_contains_full_acceptance_set(self):
        packet = build_human_review_tasks(write_report=False, answer_fn=fake_answer)
        self.assertEqual(packet["report_id"], "human_review_tasks_v1")
        self.assertEqual(len(packet["records"]), 30)
        self.assertEqual({item["category"] for item in packet["records"]}, expected_categories())
        first = packet["records"][0]
        self.assertIn("answer_response", first)
        self.assertIsNone(first["review"]["relevant_evidence_count"])
        self.assertEqual(first["review"]["query_lingo_ratings"], [])

    def test_review_task_packet_uses_exploratory_mode_for_beginner_lingo_questions(self):
        calls = []

        def answer_spy(payload):
            calls.append(payload)
            return fake_answer(payload)

        build_human_review_tasks(write_report=False, answer_fn=answer_spy)
        freedom_by_id = {
            question["id"]: call["freedom_level"]
            for question, call in zip(ACCEPTANCE_QUESTIONS, calls)
        }
        self.assertEqual(freedom_by_id["q06"], "exploratory")
        self.assertEqual(freedom_by_id["q30"], "exploratory")
        self.assertEqual(freedom_by_id["q01"], "balanced")

    def test_prepare_review_preserves_existing_review_labels(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "human_review_tasks_v1.json"
            packet = build_human_review_tasks(write_report=False, answer_fn=fake_answer)
            packet["records"][0]["review"] = acceptance_record(1)["review"]
            write_json(path, packet)

            regenerated = build_human_review_tasks(
                write_report=False,
                answer_fn=fake_answer,
                existing_records_path=path,
            )
        self.assertEqual(regenerated["records"][0]["review"]["final_review_status"], "accepted")
        self.assertEqual(regenerated["records"][0]["review"]["query_lingo_ratings"], ["useful"])
        self.assertIsNone(regenerated["records"][1]["review"]["final_review_status"])

    def test_prepare_review_can_reset_existing_review_labels(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "human_review_tasks_v1.json"
            packet = build_human_review_tasks(write_report=False, answer_fn=fake_answer)
            packet["records"][0]["review"] = acceptance_record(1)["review"]
            write_json(path, packet)

            regenerated = build_human_review_tasks(
                write_report=False,
                answer_fn=fake_answer,
                preserve_existing_reviews=False,
                existing_records_path=path,
            )
        self.assertIsNone(regenerated["records"][0]["review"]["final_review_status"])

    def test_query_diagnostics_acceptance_report_is_reviewable(self):
        packet = build_human_review_tasks(write_report=False, answer_fn=fake_answer)
        report = query_diagnostics_acceptance(packet["records"], "demo_mode")
        self.assertEqual(report["report_id"], "query_diagnostics_acceptance_v1")
        self.assertEqual(report["question_count"], 30)
        row = report["rows"][0]
        self.assertIn("matched_terms", row)
        self.assertIn("weak_terms", row)
        self.assertIn("field_phrases", row)
        self.assertIn("query_variants", row)

    def test_variant_metadata_check_allows_no_generated_variants(self):
        record = acceptance_record(0)
        record["answer_response"]["query_diagnostics"]["query_variants"] = []
        self.assertTrue(query_variants_are_labeled(record))


def build_industry_pilot_report_from_records(records):
    from canon.product import industry_pilot

    validation_errors = industry_pilot.review_validation_errors(records)
    checks = industry_pilot.acceptance_checks(records)
    checks.insert(
        0,
        industry_pilot.check(
            "review_labels_valid",
            not validation_errors,
            "Human review labels must use the expected schema and numeric ranges.",
            len(validation_errors),
        ),
    )
    return {
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "release_blockers": industry_pilot.release_blockers(checks),
    }


def build_review_status_report_from_records(records):
    from canon.product import industry_pilot

    rows = [industry_pilot.review_status_row(record) for record in records]
    complete_count = sum(1 for row in rows if row["complete"])
    return {
        "status": "complete" if complete_count >= industry_pilot.MIN_QUESTION_COUNT else "incomplete",
        "reviewed_question_count": complete_count,
        "missing_field_count": sum(len(row["missing_fields"]) for row in rows),
        "rows": rows,
    }


def acceptance_record(index):
    query = f"question {index}"
    return {
        "id": f"q{index}",
        "answer_response": {
            "query": query,
            "citations": [{"citation_id": "C1"}],
            "support_assessment": {"support_level": "supported"},
            "limitations": ["Human review required."],
            "conflict_notes": [],
            "query_diagnostics": {
                "original_query": query,
                "query_to_corpus": {"matched_terms": ["question"], "weak_terms": []},
                "result_neighborhood": {
                    "field_phrases": ["field phrase"],
                    "phrase_details": [
                        {
                            "phrase": "field phrase",
                            "source": "semantic_result_neighborhood",
                            "support_count": 2,
                        }
                    ],
                },
                "query_variants": [
                    {
                        "query": "field phrase",
                        "result_overlap_with_original": 0.5,
                        "drift_risk": "low",
                        "exploration_label": "field_synonym",
                        "freedom_level": "balanced",
                    }
                ],
                "stability": {"status": "sensitive"},
            },
        },
        "review": {
            "relevant_evidence_count": 3,
            "substantive_claim_count": 2,
            "unsupported_claim_count": 0,
            "brief_review_minutes": 12,
            "evidence_relevance": "relevant",
            "citation_support": "strong",
            "answer_usefulness": "usable",
            "query_lingo_ratings": ["useful"],
            "drift_risk_review": "acceptable",
            "final_review_status": "accepted",
        },
    }


def fake_answer(payload):
    query = payload["query"]
    return {
        "query": query,
        "mode": payload.get("mode", "demo_mode"),
        "policy": "rag",
        "answer": "A cautious cited briefing answer.",
        "citations": [{"citation_id": "C1", "title": "Evidence title", "year": 2024}],
        "evidence": [],
        "support_assessment": {"support_level": "supported"},
        "limitations": ["Human review required."],
        "conflict_notes": [],
        "claim_boundaries": {},
        "query_diagnostics": {
            "original_query": query,
            "query_to_corpus": {"matched_terms": ["sanctions"], "weak_terms": ["work"]},
            "result_neighborhood": {
                "field_phrases": ["economic sanctions"],
                "missing_from_query": ["coercive diplomacy"],
                "phrase_details": [
                    {
                        "phrase": "economic sanctions",
                        "source": "semantic_result_neighborhood",
                        "support_count": 2,
                    }
                ],
            },
            "query_variants": [
                {
                    "query": "economic sanctions",
                    "result_overlap_with_original": 0.5,
                    "drift_risk": "low",
                    "exploration_label": "field_phrase",
                    "freedom_level": payload.get("freedom_level", "balanced"),
                }
            ],
            "stability": {"status": "sensitive"},
        },
    }


def expected_categories():
    return {
        "straightforward_evidence",
        "vague_beginner_wording",
        "terminology_sensitive",
        "mixed_evidence",
        "weak_support",
        "recency_sensitive",
        "source_quality_sensitive",
        "overclaim_check",
        "unique_use_case",
    }


def fill_csv_reviews(csv_path, **overrides):
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())
    defaults = {
        "relevant_evidence_count": "3",
        "substantive_claim_count": "2",
        "unsupported_claim_count": "0",
        "brief_review_minutes": "12.5",
        "evidence_relevance": "relevant",
        "citation_support": "strong",
        "answer_usefulness": "usable",
        "query_lingo_ratings": "useful;very_useful",
        "drift_risk_review": "acceptable",
        "final_review_status": "accepted",
    }
    defaults.update(overrides)
    for row in rows:
        row.update(defaults)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
