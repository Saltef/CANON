from __future__ import annotations

import json

from canon.models import Chunk, Work


def sql_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    text = str(value).replace("'", "''")
    return f"'{text}'"


def jsonb_literal(value: object) -> str:
    return sql_literal(json.dumps(value, ensure_ascii=False)) + "::jsonb"


def export_load_sql(works: list[Work], chunks: list[Chunk], mode: str, diagnostics: dict) -> str:
    statements = [
        "BEGIN;",
        "DELETE FROM chunks;",
        "DELETE FROM works;",
    ]
    for work in works:
        statements.append(
            "INSERT INTO works "
            "(id, doi, title, publication_year, language, abstract, source_name, "
            "is_open_access, is_retracted, cited_by_count, referenced_work_count, "
            "author_display_names, author_count, max_author_cited_by_count, "
            "max_author_works_count, pdf_url, landing_page_url, raw_json) VALUES "
            f"({sql_literal(work.id)}, {sql_literal(work.doi)}, {sql_literal(work.title)}, "
            f"{sql_literal(work.year)}, {sql_literal(work.language)}, {sql_literal(work.abstract)}, "
            f"{sql_literal(work.source_name)}, {sql_literal(work.is_open_access)}, "
            f"{sql_literal(work.is_retracted)}, {sql_literal(work.cited_by_count)}, "
            f"{sql_literal(work.referenced_work_count)}, {jsonb_literal(work.author_display_names)}, "
            f"{sql_literal(work.author_count)}, {sql_literal(work.max_author_cited_by_count)}, "
            f"{sql_literal(work.max_author_works_count)}, {sql_literal(work.pdf_url)}, "
            f"{sql_literal(work.landing_page_url)}, {jsonb_literal(work.raw)}) "
            "ON CONFLICT (id) DO NOTHING;"
        )
    for chunk in chunks:
        statements.append(
            "INSERT INTO chunks "
            "(id, work_id, section, text, token_start, token_end, importance_json) VALUES "
            f"({sql_literal(chunk.id)}, {sql_literal(chunk.work_id)}, {sql_literal(chunk.section)}, "
            f"{sql_literal(chunk.text)}, {sql_literal(chunk.token_start)}, {sql_literal(chunk.token_end)}, "
            f"{jsonb_literal(chunk.importance)}) "
            "ON CONFLICT (id) DO NOTHING;"
        )
    statements.append(
        "INSERT INTO ingest_runs (mode, work_count, chunk_count, diagnostics_json) VALUES "
        f"({sql_literal(mode)}, {len(works)}, {len(chunks)}, {jsonb_literal(diagnostics)});"
    )
    statements.append("COMMIT;")
    return "\n".join(statements) + "\n"
