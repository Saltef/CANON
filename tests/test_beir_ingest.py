from __future__ import annotations

import json

from canon.ingest.beir import beir_corpus_to_unstructured, load_beir_qrels


def test_beir_corpus_to_unstructured_maps_claim_benchmark_metadata(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"_id": "doc-1", "title": "Trial abstract", "text": "Evidence text."}) + "\n",
        encoding="utf-8",
    )

    records = beir_corpus_to_unstructured(corpus, dataset_name="scifact")

    assert records == [
        {
            "id": "doc-1",
            "title": "Trial abstract",
            "text": "Trial abstract\n\nEvidence text.",
            "source_name": "BEIR",
            "document_type": "academic_article",
            "provenance": "public_benchmark",
            "domain": "scientific_claim_verification",
            "metadata": {
                "benchmark": "BEIR",
                "beir_doc_id": "doc-1",
                "beir_dataset": "scifact",
                "title_prefixed_for_retrieval": True,
            },
        }
    ]


def test_beir_corpus_to_unstructured_can_profile_nfcorpus(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"_id": "doc-1", "title": "DHA", "text": "Omega 3 discussion."}) + "\n",
        encoding="utf-8",
    )

    records = beir_corpus_to_unstructured(corpus, dataset_name="nfcorpus")

    assert records[0]["document_type"] == "biomedical_web_article"
    assert records[0]["domain"] == "biomedical_nutrition_retrieval"
    assert records[0]["text"].startswith("DHA\n\n")


def test_beir_corpus_to_unstructured_allows_profile_override(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"_id": "doc-1", "title": "DHA", "text": "Omega 3 discussion."}) + "\n",
        encoding="utf-8",
    )

    records = beir_corpus_to_unstructured(
        corpus,
        dataset_name="nfcorpus",
        document_type="academic_article",
        domain="biomedical_nutrition_retrieval",
    )

    assert records[0]["document_type"] == "academic_article"
    assert records[0]["domain"] == "biomedical_nutrition_retrieval"


def test_load_beir_qrels_reads_tsv_header(tmp_path):
    qrels = tmp_path / "test.tsv"
    qrels.write_text(
        "query-id\tcorpus-id\tscore\nq1\tdoc-1\t1\nq1\tdoc-2\t0\n",
        encoding="utf-8",
    )

    assert load_beir_qrels(qrels) == {"q1": {"doc-1": 1.0, "doc-2": 0.0}}
