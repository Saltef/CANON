import unittest
from unittest.mock import patch

from canon.eval.research_lens import evaluate_research_lenses, report_status


class ResearchLensTests(unittest.TestCase):
    @patch("canon.eval.research_lens.write_artifacts")
    @patch("canon.eval.research_lens.build_query_diagnostics")
    @patch("canon.eval.research_lens.synthesize")
    def test_evaluate_research_lenses_compares_calibrated_views(
        self,
        synthesize,
        build_query_diagnostics,
        write_artifacts,
    ):
        synthesize.side_effect = [
            synthesis_report("broad", clusters=[1, 2, 3], source_component=False),
            synthesis_report("depth", clusters=[1, 1, 2], source_component=True),
        ]
        build_query_diagnostics.return_value = {
            "query_to_corpus": {"matched_terms": ["sanctions"], "weak_terms": ["work"]},
            "result_neighborhood": {"field_phrases": ["sanction threat", "target compliance"]},
            "query_variants": [{"query": "sanction threat", "drift_risk": "low"}],
            "stability": {"status": "sensitive"},
        }

        report = evaluate_research_lenses(
            query="do sanctions work",
            mode="unit",
            top_k=3,
            lenses={
                "broad_discovery": {
                    "label": "Broad discovery",
                    "policy": "focus_diverse",
                    "freedom_level": "exploratory",
                },
                "canonical_depth": {
                    "label": "Canonical depth",
                    "policy": "source_quality_heavy",
                    "freedom_level": "balanced",
                },
            },
        )

        self.assertEqual(report["status"], "ready_for_human_calibration")
        self.assertEqual(report["comparison"]["most_broad_lens"], "broad_discovery")
        self.assertEqual(len(report["comparison"]["pairwise_overlap"]), 1)
        self.assertIn("human_review_prompts", report)
        self.assertEqual(synthesize.call_args_list[0].kwargs["policy"], "focus_diverse")
        self.assertEqual(synthesize.call_args_list[1].kwargs["policy"], "source_quality_heavy")
        self.assertEqual(build_query_diagnostics.call_args_list[0].kwargs["freedom_level"], "exploratory")
        write_artifacts.assert_called_once()

    def test_report_status_requires_breadth_and_depth_signal(self):
        self.assertEqual(report_status([]), "review")
        self.assertEqual(
            report_status(
                [
                    {
                        "source_profile": {"distinct_clusters": 3},
                        "top_score_contributors": [{"component": "source_quality"}],
                    }
                ]
            ),
            "ready_for_human_calibration",
        )


def synthesis_report(prefix, clusters, source_component):
    evidence = []
    for index, cluster_id in enumerate(clusters, start=1):
        contributor = {
            "component": "source_quality" if source_component else "semantic_similarity",
            "score_share": 0.5,
        }
        evidence.append(
            {
                "rank": index,
                "citation_id": f"C{index}",
                "title": f"{prefix} title {index}",
                "source_name": f"Source {index}",
                "year": 2020 + index,
                "work_id": f"{prefix}-work-{index}",
                "cluster_id": cluster_id,
                "final_score": 0.9,
                "explanation": {"top_contributors": [contributor]},
            }
        )
    return {
        "answer": f"{prefix} answer",
        "support_assessment": {"support_level": "mixed"},
        "evidence": evidence,
    }


if __name__ == "__main__":
    unittest.main()
