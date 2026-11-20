"""Map PDF captions, headers, and cells to the Sverka extraction contract."""

from dataclasses import dataclass, field
import math
import re
from typing import Any

from pdf_pages import PDF, Page
from rag_tables import Column, MISSING, Table, read_tables

RAG = re.compile(r"\bRAG(?:s)?\b|retrieval[\s-]+augmented\s+generation", re.I)
DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
TOP_K = re.compile(r"\btop[-_ ]*k\s*[=:]\s*(\d+)\b", re.I)
ORIGIN_CITATION = re.compile(
    r"(?:results?|scores?|values?)\s+(?:were\s+)?(?:reported|taken|copied|reproduced)\s+"
    r"(?:by|from)\s+(\[\d+\])", re.I,
)
UNSOURCED_QUOTATION = re.compile(r"(?:results?|scores?|values?)\s+(?:were\s+)?(?:taken|copied|reproduced)\s+from\b", re.I)
TASK_TYPE = re.compile(r"\b(?:question[\s-]+answering|answering\s+questions)\b", re.I)
SPLIT = re.compile(r"\b(validation|test|train(?:ing)?|dev(?:elopment)?)\s+(?:split|set)\b", re.I)
METRIC_VARIANTS = (
    (re.compile(r"\bnormalized\s+exact\s+match\b", re.I), {"em", "exact match"}, "normalized exact match"),
    (re.compile(r"\bword[\s-]+level\s+f1(?:[\s-]+score)?\b", re.I), {"f1"}, "word-level"),
    (re.compile(r"\btoken[\s-]+level\s+f1(?:[\s-]+score)?\b", re.I), {"f1"}, "token-level"),
)


@dataclass
class Payload:
    document: dict[str, Any]
    publication: dict[str, Any]
    configurations: list[dict[str, Any]] = field(default_factory=list)
    datasets: list[dict[str, Any]] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    experiments: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": "1.0", "extraction_mode": "python", **vars(self)}

    def add(self, collection: str, id_field: str, prefix: str, **values: Any) -> str:
        rows = getattr(self, collection)
        record_id = f"{prefix}_{len(rows) + 1}"
        rows.append({id_field: record_id, **values})
        return record_id

    def cite(self, kind: str, record_id: str, attribute: str | None, page: Page,
             fragment: str, **locator: str) -> None:
        if fragment and fragment in page.text:
            verification = page.verification
        elif locator.get("table"):
            verification = "locator_only"
            fragment = ""
        else:
            return
        self.add("evidence", "evidence_id", "evidence",
                 entity_type=kind, entity_id=record_id, field=attribute,
                 page=page.number, fragment=fragment or None, verification=verification,
                 **locator)


def title_from_page(page: Page) -> tuple[str | None, str | None]:
    block = []
    for line in page.text.splitlines()[:22]:
        value = line.strip()
        if not value:
            if block:
                break
            continue
        if re.match(r"(?:arXiv:|preprint|proceedings of|abstract\b)", value, re.I):
            if block:
                break
            continue
        if len(value) >= 14 and re.search(r"[A-Za-z]{3}", value):
            block.append(value)
        if len(block) == 3:
            break
    return (" ".join(block), block[0]) if block else (None, None)


def author_names(info: str | None) -> list[str] | None:
    if not info:
        return None
    names = [item.strip() for item in re.split(r"[;,]|\s+and\s+", info) if item.strip()]
    return names or None


def extract_publication(pdf: PDF) -> dict[str, Any]:
    first_page = pdf.pages[0]
    first_text = " ".join(first_page.text.split())
    title = pdf.metadata.get("Title") or None
    title_fragment = None
    if not title or re.fullmatch(r"[DP]\d{3}", title, re.I):
        title, title_fragment = title_from_page(first_page)
    doi = DOI.search(first_text[:7000])
    # Reference lists alone are weak signals of relevance to this corpus.
    first_readable_pages = [page.prose for page in pdf.pages if page.text][:3]
    beginning = " ".join(first_readable_pages)
    in_scope = bool(RAG.search(beginning) or RAG.search(title or ""))
    payload = Payload(
        document={"page_count": len(pdf.pages), "unreadable_pages": [page.number for page in pdf.pages if not page.text]},
        publication={
            "publication_id": "publication_1", "title": title, "year": None,
            "doi": doi[0].rstrip(".,;") if doi else None, "authors": author_names(pdf.metadata.get("Author")),
            "in_scope": in_scope,
            "exclusion_reason": None if in_scope else "В доступном начале публикации не найдено описание RAG.",
        },
    )
    if title_fragment:
        payload.cite("publication", "publication_1", "title", first_page, title_fragment)
    if not in_scope:
        return payload.as_dict()

    question_answering = bool(TASK_TYPE.search(beginning) or TASK_TYPE.search(title or ""))
    for page in pdf.pages:
        for table in read_tables(page):
            add_table(payload, table, question_answering)
    return payload.as_dict()


def scale_and_value(raw: str, header_unit: str | None) -> tuple[str | None, str | None, float | None, str]:
    unit = "%" if raw.endswith("%") else header_unit
    scale = {"%": "percent", "fraction": "fraction"}.get((unit or "").lower())
    number = raw.removesuffix("%").replace("−", "-").replace(",", ".")
    if raw in MISSING:
        return unit, scale, None, "absent"
    if re.match(r"^[+-]?0\d", number):
        return unit, scale, None, "present_not_extractable"
    value = float(number)
    if not math.isfinite(value):
        return unit, scale, None, "present_not_extractable"
    return unit, scale, value, "extracted"


def metric_variant(column: Column, page: Page) -> str | None:
    name = column.header.name.lower()
    for pattern, names, variant in METRIC_VARIANTS:
        if name in names and pattern.search(page.prose):
            return variant
    return None


def add_table(payload: Payload, table: Table, question_answering: bool) -> None:
    citation = ORIGIN_CITATION.search(table.caption)
    if not citation and UNSOURCED_QUOTATION.search(table.caption):
        return
    origin = "cited" if citation else "primary"
    source = {"table": table.label}
    split = SPLIT.search(table.caption)
    context = table.page.prose
    qa = question_answering or bool(TASK_TYPE.search(context))
    datasets: dict[tuple[str | None, str | None], tuple[str, str, str]] = {}
    for column in table.columns:
        dataset = column.dataset
        dataset_key = dataset, column.original_row
        if dataset_key in datasets:
            continue
        dataset_id = payload.add(
            "datasets", "dataset_id", "dataset",
            dataset_name=dataset, dataset_version=None, data_split=split[1].lower() if split else None,
        )
        task_type = "question answering" if qa else None
        if qa and dataset and re.search(r"\bmulti[- ]hop\b", dataset, re.I):
            task_type = "multi-hop question answering"
        task_id = payload.add("tasks", "task_id", "task", task_type=task_type)
        protocol = None
        if column.aggregation and re.search(r"\busing the RAGAS framework\b", context, re.I):
            protocol = "RAGAS"
        experiment_id = payload.add(
            "experiments", "experiment_id", "experiment", evaluation_protocol=protocol,
            note=table.caption,
        )
        if dataset:
            payload.cite("dataset", dataset_id, "dataset_name", table.page,
                         dataset if dataset in table.page.text else table.label, table_column=dataset, **source)
        datasets[dataset_key] = dataset_id, task_id, experiment_id

    parent_name = ""
    for row, values in table.rows:
        if row.startswith(("-", "–", "—")) and parent_name:
            config_name = f"{parent_name} / {row}"
        else:
            parent_name = row
            config_name = row
        configurations = {}
        for column, raw in zip(table.columns, values, strict=True):
            generator = column.generator
            if generator not in configurations:
                configurations[generator] = add_configuration(payload, table, row, config_name, generator)
            config_id = configurations[generator]
            dataset_id, task_id, experiment_id = datasets[column.dataset, column.original_row]
            header = column.header
            unit, scale, value, status = scale_and_value(raw, header.unit)
            direction = "lower_is_better" if header.name.lower() in {"latency", "response time"} else None
            if direction is None and header.name.lower() in {"em", "exact match", "f1", "accuracy", "precision", "recall"}:
                direction = "higher_is_better"
            result_id = payload.add(
                "results", "result_id", "result", experiment_id=experiment_id, config_id=config_id,
                dataset_id=dataset_id, task_id=task_id, metric_name=header.name,
                metric_variant=metric_variant(column, table.page), metric_direction=direction,
                metric_k=header.cutoff, metric_k_applicability="applicable" if header.cutoff is not None else "not_applicable",
                reported_value=value, reported_scale=scale, raw_value=raw, raw_unit=unit,
                aggregation=column.aggregation, result_origin=origin,
                cited_publication_id=citation[1] if citation else None, extraction_status=status,
            )
            column_title = " / ".join(item for item in [generator, column.dataset, header.label, column.aggregation] if item)
            if column.original_row:
                column_title = f"{row} / {header.label}"
            payload.cite("result", result_id, "raw_value", table.page, raw,
                         table_row=column.original_row or row, table_column=column_title, **source)


def add_configuration(payload: Payload, table: Table, row: str, name: str, generator: str | None) -> str:
    cutoff = TOP_K.search(row)
    config_id = payload.add(
        "configurations", "config_id", "config", rag_variant=name,
        generator_model=generator, top_k=int(cutoff[1]) if cutoff else None,
    )
    payload.cite("configuration", config_id, "rag_variant", table.page, row,
                 table_row=row, table=table.label)
    if cutoff:
        payload.cite("configuration", config_id, "top_k", table.page, cutoff[0],
                     table_row=row, table=table.label)
    if generator:
        payload.cite("configuration", config_id, "generator_model", table.page, generator,
                     table_column=generator, table=table.label)
    return config_id
