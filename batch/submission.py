"""Project Go DocumentDetail records onto the organizer's seven CSV tables."""

import csv
import json
import re
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

from .api import problem_text
from .result_validation import csv_cell, validate_duplicates


CORPUS_ID = re.compile(r"[DP][0-9]{3}")
DOCUMENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
TABLE_KEYS = {
    "publications": "publication_id",
    "configurations": "config_id",
    "datasets": "dataset_id",
    "tasks": "task_id",
    "experiments": "experiment_id",
    "results": "result_id",
    "issues": "issue_id",
}
SOURCE_FIELDS = ("page", "section", "table", "figure", "fragment")
SOURCE_LOCATORS = ("fragment", "table", "figure")
SCHEMA_MEMBER = "submission_template/schema.json"


def read_schema(schema_path, corpus_zip):
    if schema_path is not None:
        schema = json.loads(schema_path.read_text(encoding="utf-8-sig"))
    else:
        try:
            with zipfile.ZipFile(corpus_zip) as archive:
                schema = json.loads(archive.read(SCHEMA_MEMBER))
        except (zipfile.BadZipFile, KeyError, RuntimeError) as exc:
            raise ValueError(f"{corpus_zip}: cannot read {SCHEMA_MEMBER}: {exc}") from exc
    tables = schema.get("tables") if isinstance(schema, dict) else None
    if not isinstance(tables, dict) or tables.keys() != TABLE_KEYS.keys():
        raise ValueError("submission schema must describe the organizer's seven tables")
    for table, fields in tables.items():
        if (not isinstance(fields, list) or any(not isinstance(field, str) or not field for field in fields)
                or len(set(fields)) != len(fields) or not {"publication_id", TABLE_KEYS[table]}.issubset(fields)):
            raise ValueError(f"invalid submission schema columns for {table}")
    return tables


def pdf_inputs(directory):
    paths = sorted(path for path in directory.iterdir() if path.suffix.lower() == ".pdf")
    if not paths:
        raise ValueError(f"{directory}: no corpus PDFs")
    invalid = [path.name for path in paths
               if path.suffix != ".pdf" or not CORPUS_ID.fullmatch(path.stem) or not path.is_file()]
    if invalid:
        raise ValueError(f"{directory}: expected corpus filenames like D001.pdf or P090.pdf; "
                         + "invalid inputs: " + ", ".join(invalid))
    return {path.stem: path for path in paths}


def document_metadata(detail):
    if not isinstance(detail, dict) or not isinstance(detail.get("document"), dict):
        raise ValueError("expected a Go DocumentDetail object with document metadata")
    return detail["document"]


def job_label(document):
    return f"{document.get('corpus_id') or '<missing corpus_id>'} ({document.get('document_id') or '<missing document_id>'})"


def mode_errors(documents):
    errors = []
    modes = set()
    for document in documents:
        mode = document.get("extraction_mode")
        if isinstance(mode, str):
            modes.add(mode)
        if mode != "python":
            errors.append(f"{job_label(document)}: extraction_mode={mode!r}; submission requires python")
    if {"python", "mock"}.issubset(modes):
        errors.insert(0, "Mixed extraction modes (mock and python) in the selected jobs.")
    return errors


def status_error(document):
    status = document.get("status")
    failure = problem_text(document.get("failure"))
    message = f"{job_label(document)}: status={status!r}"
    return f"{message}: {failure}" if failure else message


def export_details(path):
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and "documents" in payload:
        if payload.get("schema_version") != "1.0":
            raise ValueError(f"{path}: expected Go export schema_version '1.0'")
        if isinstance(payload["documents"], list):
            return payload["documents"]
    if isinstance(payload, dict) and "document" in payload:
        return [payload]
    raise ValueError(f"{path}: expected a Go /api/v1/export object or a DocumentDetail")


def completed_details(details, expected_ids):
    by_corpus = {}
    documents = []
    errors = []
    wanted = set(expected_ids) if expected_ids is not None else None
    for detail in details:
        try:
            document = document_metadata(detail)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        documents.append(document)
        document_id = document.get("document_id")
        if not isinstance(document_id, str) or not DOCUMENT_ID.fullmatch(document_id):
            errors.append(f"{job_label(document)}: missing or invalid document_id")
        corpus_id = document.get("corpus_id")
        if not isinstance(corpus_id, str) or not CORPUS_ID.fullmatch(corpus_id):
            errors.append(f"{job_label(document)}: missing or invalid corpus_id")
        elif corpus_id in by_corpus:
            other = document_metadata(by_corpus[corpus_id])
            errors.append(f"{job_label(document)}: duplicate corpus_id also held by {job_label(other)}")
        else:
            by_corpus[corpus_id] = detail
        if document.get("status") != "completed":
            errors.append(status_error(document))
        elif not isinstance(detail.get("data"), dict):
            errors.append(f"{job_label(document)}: completed job is missing data")

    if wanted is not None:
        for corpus_id in sorted(wanted - by_corpus.keys()):
            errors.append(f"{corpus_id}: missing document")
        for corpus_id in sorted(by_corpus.keys() - wanted):
            errors.append(f"{corpus_id}: unexpected document")
    if not by_corpus and not errors:
        errors.append("No corpus documents in the export.")
    errors.extend(mode_errors(documents))
    if errors:
        raise ValueError("\n".join(errors))
    return by_corpus


def records(data, key):
    rows = data.get(key)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected a list of objects in data.{key}")
    return rows


def joined(values, field):
    if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{field} must be a list of nonempty strings")
    return "; ".join(values)


def has_csv_locator(item):
    return any(csv_cell(item.get(field)) for field in SOURCE_LOCATORS)


def value_evidence_rank(item):
    # Prefer sources that fit the organizer's locator fields. Break ties by
    # specificity and verified quotes, then retain the API's evidence order.
    return (
        not has_csv_locator(item),
        item.get("field") != "raw_value",
        not (csv_cell(item.get("fragment")) and item.get("verification") in ("text_match", "ocr_match")),
    )


def result_row(summary, sources):
    result = summary.get("result")
    if not isinstance(result, dict):
        raise ValueError("result summary is missing its result object")
    row = dict(result)
    result_id = result["result_id"]
    origin = csv_cell(result.get("result_origin"))
    cited_id = csv_cell(result.get("cited_publication_id"))
    if origin == "cited" and not cited_id:
        raise ValueError(f"result {result_id!r}: cited result without citation identifier (cited_publication_id)")
    if origin == "primary" and cited_id:
        raise ValueError(f"result {result_id!r}: primary result has cited_publication_id")
    source = min(sources.get(result_id, ()), key=value_evidence_rank, default={})
    if csv_cell(result.get("extraction_status")) == "extracted" and not has_csv_locator(source):
        raise ValueError(f"result {result_id!r}: extracted result without a CSV source locator "
                         "(requires a page and at least one of fragment, table, or figure)")

    normalized = summary.get("normalized")
    baseline = summary.get("baseline")
    value = normalized["value"] if normalized is not None else None
    unit = normalized["unit"] if normalized is not None else None
    row.update(metric_value=value, normalized_value=value, metric_unit=unit, normalized_unit=unit,
               baseline_value=baseline["value"] if baseline is not None else None)

    row.update({f"source_{field}": source.get(field) for field in SOURCE_FIELDS})
    notes = [row["note"]] if row.get("note") else []
    for field, label in (("table_row", "Table row"), ("table_column", "Table column"),
                         ("block_id", "Block"), ("bbox", "Page bbox")):
        if source.get(field):
            notes.append(f"{label}: {source[field]}")
    row["note"] = "\n".join(notes)
    return row


def add_document(rows, corpus_id, detail):
    data = detail["data"]
    publication = data.get("publication")
    if not isinstance(publication, dict) or type(publication.get("in_scope")) is not bool:
        raise ValueError("missing publication or invalid in_scope decision")
    included = publication["in_scope"]
    if not included and data.get("results"):
        raise ValueError("excluded publication has result rows")
    authors = publication.get("authors")
    rows["publications"].append(dict(publication, publication_id=corpus_id, DOI=publication.get("doi"),
                                     in_scope="true" if included else "false",
                                     authors=joined(authors, "authors") if authors is not None else None))
    if not included:
        return

    for table in ("configurations", "datasets", "tasks", "experiments"):
        rows[table].extend(dict(row, publication_id=corpus_id) for row in records(data, table))

    sources_by_result = defaultdict(list)
    sources_by_id = {}
    for item in records(data, "evidence"):
        sources_by_id[item["evidence_id"]] = item
        if item.get("entity_type") == "result" and item.get("field") in (None, "", "raw_value"):
            page = csv_cell(item.get("page"))
            if (re.fullmatch(r"[0-9]*[1-9][0-9]*", page)
                    and (has_csv_locator(item) or item.get("bbox") or item.get("block_id"))):
                sources_by_result[item["entity_id"]].append(item)

    result_rows = []
    for summary in records(data, "results"):
        mode = summary.get("extraction_mode")
        if mode != "python":
            raise ValueError(f"python document has a result with extraction_mode={mode!r}")
        if summary.get("document_id", detail["document"].get("document_id")) != detail["document"].get("document_id"):
            raise ValueError("result summary belongs to a different document")
        result_rows.append(dict(result_row(summary, sources_by_result), publication_id=corpus_id))

    validate_duplicates(result_rows)
    rows["results"].extend(result_rows)
    result_ids = {row["result_id"] for row in result_rows}

    for issue in records(data, "issues"):
        ids = issue.get("result_ids", [])
        record_ids = joined(ids, "issue.result_ids")
        if not set(ids).issubset(result_ids):
            raise ValueError(f"issue {issue['issue_id']!r} references results outside this publication")
        sources = [sources_by_id[evidence_id] for evidence_id in issue.get("evidence_ids", [])]
        source = next((item for item in sources if item.get("fragment")), sources[0] if sources else {})
        rows["issues"].append(dict(issue, publication_id=corpus_id, record_ids=record_ids,
                                   source_page=source.get("page"), source_fragment=source.get("fragment")))


def check_output(output):
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ValueError(f"{output}: output must be a new or empty directory")


def write_submission(details, tables, output, *, expected_ids=None):
    completed = completed_details(details, expected_ids)
    rows = {table: [] for table in tables}
    for corpus_id, detail in sorted(completed.items()):
        try:
            add_document(rows, corpus_id, detail)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{job_label(detail['document'])}: invalid submission data: {exc}") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sverka-", dir=output.parent) as temporary:
        staged = Path(temporary) / "submission"
        staged.mkdir()
        for table, columns in tables.items():
            with (staged / f"{table}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows[table])
        check_output(output)
        staged.rename(output)
    return {table: len(values) for table, values in rows.items()}
