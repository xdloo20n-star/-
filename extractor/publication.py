"""Map PDF captions, headers, and cells to the Sverka extraction contract."""

from collections import defaultdict
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Any

from context_recall_tables import read_context_recall_tables
from coda_tables import read_coda_tables

from benchmark_distribution import read_benchmark_distributions
from efficiency_summary import read_efficiency_summaries

from failure_rates import failure_rate_definition
from factuality_tables import read_factuality_tables
from failure_diagnostics import order_audit_summaries
from failure_ablations import link_ablation_averages
from grouped_scores import enrich_grouped_scores
from knowledge_benchmarks import read_knowledge_benchmarks
from native_benchmark_tables import read_native_benchmark_tables
from opinion_tables import read_opinion_tables
from paired_scores import enrich_repeated_score_rows, read_additional_score_tables
from pdf_pages import PDF, Page
from prostep_context import enrich_prostep_context
from prostep_appendix import read_prostep_appendix
from qa_comparison import enrich_qa_comparisons
from rag_tables import Column, MISSING, Table, read_tables
from reasoning_tables import read_reasoning_tables
from retrieval_tables import read_retrieval_tables
from scanned_score_tables import read_scanned_score_tables
from synthetic_accuracy import read_synthetic_accuracy
from threadrag_tables import read_threadrag_tables
from training_scan import read_training_scan_tables
from training_scan_pairs import read_training_scan_pairs
from training_tables import read_training_tables
from vector_histogram import HistogramFigure, METRIC as HISTOGRAM_METRIC, read_vector_histograms
from vector_scatter import (
    ScatterFigure, read_vector_length_scatter, read_vector_noisy_contours, read_vector_noisy_scatter, read_vector_scatter,
)
from watermark_tables import read_watermark_tables

RAG = re.compile(r"\bRAG(?:s)?\b|retrieval[\s-]+augmented\s+generation", re.I)
DOI = re.compile(r"(?:\bdoi\s*:\s*|\bdoi\.org/)(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I)
TOP_K = re.compile(r"\btop[-_ ]*k\s*[=:]\s*(\d+)\b", re.I)
ORIGIN_CITATION = re.compile(
    r"(?:results?|scores?|values?)\s+(?:were\s+)?(?:reported|taken|copied|reproduced)\s+"
    r"(?:by|from)\s+(\[\d+\])", re.I,
)
UNSOURCED_QUOTATION = re.compile(r"(?:results?|scores?|values?)\s+(?:were\s+)?(?:taken|copied|reproduced)\s+from\b", re.I)
TASK_TYPE = re.compile(r"\b(?:question[\s-]+answering|answering\s+questions)\b", re.I)
SPLIT = re.compile(r"\b(validation|test(?:ing)?|train(?:ing)?|dev(?:elopment)?)\s+(?:split|set)s?\b", re.I)
CAPTION_CUTOFF = re.compile(r"\b(?:at|with)\s+k\s*=\s*(\d+)\b", re.I)
PUBLICATION_ID = re.compile(r"[DP]\s*[\dO]\s*[\dO]\s*[\dO](?:\s*\.\s*pdf)?", re.I)
TEMPLATE_TITLE = re.compile(r"^Microsoft Word\s*-|^Paper Title \(use style: paper title\)$", re.I)
RUNNING_HEAD = re.compile(
    r"^(?:arXiv:|preprint\b|(?:proceedings|findings) of\b|published as a\b|"
    r"available online at\b|www\.|ScienceDirect$|Procedia Computer Science\b|"
    r"\d+(?:st|nd|rd|th) International Conference\b)", re.I,
)
ABSTRACT = re.compile(r"^a\s*b\s*s\s*t\s*r\s*a\s*c\s*t\b", re.I)
AFFILIATION = re.compile(
    r"\b(?:universit(?:y|ies)|institute|institution|laborator(?:y|ies)|labs?|"
    r"department|school|college|academy|faculty|research(?:er| centre| center)|"
    r"technologies|corporation|inc|llc|ltd|limited|corp)\b|@", re.I,
)
AUTHOR_MARK = r"[*∗†‡§¶∥⋆♠♡♣♢♦⋄]"
AUTHOR_INDEX = r"\d{1,2}(?!\d)"
AUTHOR_SUFFIX = re.compile(
    rf"(?<=[^\W\d_]|\.)(?:[ \t]*(?:,[ \t]*)?(?:{AUTHOR_INDEX}|{AUTHOR_MARK}))+"
    rf"(?=[\s,;]|and\b|$)"
)
AFFILIATION_MARKERS = re.compile(rf"(?:[\s,\d]|{AUTHOR_MARK})+")
NAME_WORD = re.compile(r"[^\W\d_]+(?:[-‐’'][^\W\d_]+)*\.?")
NAME_PARTICLES = {"al", "da", "de", "del", "di", "dos", "du", "la", "le", "van", "von", "der", "den"}
PUBLISHED_YEAR = re.compile(r"^Published as a (?:conference|workshop) paper at\b.*?\b((?:19|20)\d{2})\b", re.I)
COPYRIGHT_YEAR = re.compile(r"©\s*((?:19|20)\d{2})\s+\w")
TABLE_REFERENCE = re.compile(r"\btable\s+(?:\d+(?:\.\d+)*|[ivxlcdm]+)\b", re.I)
EVALUATION_STATEMENT = re.compile(
    r"^(?:(?:during|for) (?:the )?evaluation[, :]|(?:evaluation )?metrics?[:.]|we report\b)", re.I,
)
RAGAS_PROTOCOL = re.compile(r"\b(?:using the RAGAS framework|evaluated (?:with|using) RAGAS)\b", re.I)
UNIT_TEXT = r"%|percent(?:ages?)?|fractions?|ms|s|sec|seconds?|milliseconds?"
UNIT_STATEMENT = r"(?:are|is|were)\s+(?:(?:reported|expressed|measured)\s+)?(?:in|as)\s+(?:an?\s+)?"
VALUE_LITERAL = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
NONSCALAR_VALUE = re.compile(
    rf"(?:[<>≤≥]\s*(?:{VALUE_LITERAL}|10-\d+)|{VALUE_LITERAL}\s*[–—-]\s*{VALUE_LITERAL}|[↑↓]\s*{VALUE_LITERAL})"
)
CELL_UNIT = re.compile(r"\s*(pp|MB|GB)$")
APPROXIMATE_PREFIX = re.compile(r"^[∼~≈]\s*(?=[+−\-\d.])")
TOKEN_MULTIPLIER = {"k": 1000, "M": 1000000}
CONTRAST = re.compile(r"\b(?:except|unlike|whereas|rather than|instead of|but)\b", re.I)
METRIC_VARIANTS = (
    (re.compile(r"\bexact\s+match\b.{0,180}\bstandard\s+text\s+normalization\b", re.I | re.S),
     {"em", "exact match"}, "exact match after standard text normalization"),
    (re.compile(r"\btoken[-\s]+level\s+overlap\b.{0,180}\bstandard\s+text\s+normalization\b", re.I | re.S),
     {"f1"}, "token-level overlap after standard text normalization"),
    (re.compile(r"\bnormalized\s+exact\s+match\b", re.I), {"em", "exact match"}, "normalized exact match"),
    (re.compile(r"\bword[\s-]+level\s+f1(?:[\s-]+score)?\b", re.I), {"f1"}, "word-level"),
    (re.compile(r"\btoken[\s-]+level\s+f1(?:[\s-]+score)?\b", re.I), {"f1"}, "token-level"),
    (re.compile(r"\bmacro(?:[\s-]+averaged)?[\s-]+f1\b", re.I), {"f1"}, "macro"),
    (re.compile(r"\bmicro(?:[\s-]+averaged)?[\s-]+f1\b", re.I), {"f1"}, "micro"),
)
METRIC_NAME_ALIASES = (
    (re.compile(r"^entity match$", re.I), re.compile(r"\bEntity Match Rate\b", re.I), "Entity Match Rate"),
    (re.compile(r"^sent-int\s+σ$", re.I), re.compile(r"\bSentiment[- ]Intensity\s+σ\b", re.I), "Sentiment-Intensity sigma"),
    (re.compile(r"^biz impact\s+σ$", re.I), re.compile(r"\bBusiness Impact\s+σ\b", re.I), "Business Impact sigma"),
    (re.compile(r"^emot\.? markers$", re.I), re.compile(r"\bEmotional Markers\b", re.I), "Emotional Markers"),
    (re.compile(r"^revenue band$", re.I), re.compile(r"\bDemographic Coverage\b.{0,180}\bRevenue Band\b", re.I | re.S), "Revenue Band Coverage"),
    (re.compile(r"^tenure$", re.I), re.compile(r"\bDemographic Coverage\b.{0,180}\bTenure\b", re.I | re.S), "Tenure Coverage"),
    (re.compile(r"^evidence$", re.I), re.compile(r"\bDemographic Coverage\b.{0,180}\bEvidence Type\b", re.I | re.S), "Evidence Type Coverage"),
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
             fragment: str, **locator: Any) -> None:
        if fragment and (fragment in page.text or fragment in page.reading_text):
            verification = page.verification
        elif locator.get("table") or locator.get("figure") or locator.get("bbox"):
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
    for line in page.text.split("\n")[:32]:
        value = line.strip()
        if not value or value.isdecimal() or PUBLICATION_ID.fullmatch(value) or RUNNING_HEAD.match(value):
            if block:
                break
            continue
        if ABSTRACT.match(value) or author_line(value) or not re.search(r"[^\W\d_]{3}", value):
            break
        block.append(line)
        if len(block) == 4:
            break
    fragment = "\n".join(block).strip()
    return (" ".join(fragment.split()), fragment) if fragment else (None, None)


def author_names(info: str | None) -> list[str] | None:
    if not info:
        return None
    names = [item.strip() for item in re.split(r"[;,]|\s+and\s+", info) if item.strip()]
    return names or None


def author_line(line: str, *, allow_single: bool = False) -> list[str]:
    if AFFILIATION.search(line):
        return []
    separated, markers = AUTHOR_SUFFIX.subn("|", line)
    names = [part.strip() for part in re.split(r"[,;|]|\s{2,}|\band\s+", separated) if part.strip()]
    if len(names) < 2 and not markers and not allow_single:
        return []
    for name in names:
        words = name.split()
        if not 2 <= len(words) <= 4 or any(
            not NAME_WORD.fullmatch(word) or (not word[0].isupper() and word not in NAME_PARTICLES)
            for word in words
        ):
            return []
    return [" ".join(name.split()) for name in names]


def page_byline(page: Page) -> tuple[str | None, list[tuple[str, list[str]]]]:
    if page.verification != "text_match":
        return None, []
    lines = page.text.split("\n")[:35]
    front = []
    for index, raw in enumerate(lines):
        line = raw.strip()
        if ABSTRACT.match(line):
            break
        if line and not RUNNING_HEAD.match(line) and not PUBLICATION_ID.fullmatch(line):
            front.append((index, line))

    for start in range(1, min(5, len(front))):
        byline = []
        next_line = start
        while next_line < len(front) and (names := author_line(front[next_line][1])):
            byline.append((front[next_line][1], names))
            next_line += 1
        if not byline and start + 1 < len(front) and AFFILIATION.search(front[start + 1][1]):
            if names := author_line(front[start][1], allow_single=True):
                byline.append((front[start][1], names))
                next_line += 1
        if not byline:
            continue
        first, last = front[0][0], front[start - 1][0]
        title = "\n".join(lines[first:last + 1]).strip()
        following = [line for _, line in front[next_line:next_line + 4]
                     if not AFFILIATION_MARKERS.fullmatch(line)]
        if not any(AFFILIATION.search(line) for line in following):
            continue
        # A short organization name can precede the contact line. Longer
        # unparsed lines can contain more authors with no reliable separators.
        if not AFFILIATION.search(following[0]) and not (
            len(following[0].split()) <= 3 and len(following) > 1 and "@" in following[1]
        ):
            return title, []
        # A multi-column byline can alternate lines of authors and affiliations.
        for offset, (_, line) in enumerate(front[next_line + 1:], start=next_line + 1):
            names = author_line(line)
            if names and any(AFFILIATION.search(text) for _, text in front[offset + 1:offset + 4]):
                byline.append((line, names))
        return title, byline
    return None, []


def publication_year(page: Page) -> tuple[int | None, str | None]:
    lines = page.text.splitlines()
    for index, raw in enumerate(lines):
        line = raw.strip()
        match = PUBLISHED_YEAR.match(line)
        if match:
            return int(match[1]), line
        match = COPYRIGHT_YEAR.search(line)
        if match and any(re.match(r"\s*(?:proceedings|findings) of\b", prev, re.I)
                         for prev in lines[max(0, index - 3):index]):
            return int(match[1]), line
    return None, None


def extract_publication(pdf: PDF, *, source_path: Path | None = None) -> dict[str, Any]:
    first_page = pdf.pages[0]
    title = (pdf.metadata.get("Title") or "").strip()
    template_title = bool(TEMPLATE_TITLE.search(title))
    title_fragment = None
    title_from_byline, byline = page_byline(first_page)
    if not title or template_title or PUBLICATION_ID.fullmatch(title):
        if title_from_byline:
            title_fragment = title_from_byline
            title = " ".join(title_fragment.split())
        else:
            title, title_fragment = title_from_page(first_page)
    metadata_authors = pdf.metadata.get("Author")
    authors = author_names(metadata_authors)
    author_fragments = []
    if byline and (not authors or template_title or re.search(r"\d", metadata_authors or "")):
        authors = [name for _, names in byline for name in names]
        author_fragments = [line for line, _ in byline]
    elif template_title or not authors or PUBLICATION_ID.fullmatch(metadata_authors or ""):
        authors = None
    year, year_fragment = publication_year(first_page)
    doi = DOI.search((first_page.reading_text or first_page.text)[:7000])
    doi_value = doi[1].rstrip(".,;") if doi else None
    # Reference lists alone are weak signals of relevance to this corpus.
    first_readable_pages = [page.prose for page in pdf.pages if page.text][:3]
    beginning = " ".join(first_readable_pages)
    in_scope = bool(RAG.search(beginning) or RAG.search(title or ""))
    payload = Payload(
        document={"page_count": len(pdf.pages), "unreadable_pages": [page.number for page in pdf.pages if not page.text]},
        publication={
            "publication_id": "publication_1", "title": title, "year": year,
            "doi": doi_value, "authors": authors,
            "in_scope": in_scope,
            "exclusion_reason": None if in_scope else "В доступном начале публикации не найдено описание RAG.",
        },
    )
    if title_fragment:
        payload.cite("publication", "publication_1", "title", first_page, title_fragment)
    for fragment in author_fragments:
        payload.cite("publication", "publication_1", "authors", first_page, fragment)
    if year_fragment:
        payload.cite("publication", "publication_1", "year", first_page, year_fragment)
    if doi_value:
        payload.cite("publication", "publication_1", "doi", first_page, doi_value)
    if not in_scope:
        return payload.as_dict()

    question_answering = bool(TASK_TYPE.search(beginning) or TASK_TYPE.search(title or ""))
    failure_definition = failure_rate_definition(pdf.pages)
    extracted_tables = [table for page in pdf.pages for table in read_tables(page, failure_definition)]
    extracted_tables += read_additional_score_tables(pdf.pages, extracted_tables)
    extracted_tables += read_knowledge_benchmarks(pdf.pages, extracted_tables)
    extracted_tables += read_watermark_tables(pdf.pages, extracted_tables)
    extracted_tables += read_retrieval_tables(pdf.pages, extracted_tables)
    extracted_tables = read_efficiency_summaries(pdf.pages, extracted_tables)
    extracted_tables += read_benchmark_distributions(pdf.pages, extracted_tables)
    extracted_tables += read_native_benchmark_tables(pdf.pages, extracted_tables)
    extracted_tables += read_prostep_appendix(pdf.pages, extracted_tables)
    extracted_tables += read_threadrag_tables(pdf.pages, extracted_tables)
    extracted_tables = read_synthetic_accuracy(pdf.pages, extracted_tables)
    extracted_tables += read_training_tables(pdf.pages, extracted_tables)
    extracted_tables += read_training_scan_tables(pdf.pages, extracted_tables)
    extracted_tables += read_training_scan_pairs(pdf.pages, extracted_tables)
    extracted_tables += read_context_recall_tables(pdf.pages, extracted_tables)
    extracted_tables += read_coda_tables(pdf.pages, extracted_tables)
    extracted_tables = read_opinion_tables(pdf.pages, extracted_tables)
    extracted_tables = read_factuality_tables(pdf.pages, extracted_tables)
    extracted_tables = read_reasoning_tables(pdf.pages, extracted_tables)
    extracted_tables = read_scanned_score_tables(pdf.pages, extracted_tables)
    extracted_tables = enrich_grouped_scores(pdf.pages, extracted_tables)
    extracted_tables = enrich_repeated_score_rows(pdf.pages, extracted_tables)
    extracted_tables = enrich_prostep_context(pdf.pages, extracted_tables)
    extracted_tables = enrich_qa_comparisons(pdf.pages, extracted_tables)
    extracted_tables = link_ablation_averages(pdf.pages, extracted_tables)
    extracted_tables, audit_summaries = order_audit_summaries(pdf.pages, extracted_tables)
    by_page = defaultdict(list)
    for table in extracted_tables + audit_summaries:
        by_page[table.page.number].append(table)
    duplicate_index = {}
    for tables in by_page.values():
        for table in tables:
            if not (table.family == "judge_summary" and table.columns[0].duplicate_scope):
                add_table(payload, table, question_answering, tables, duplicate_index)
    for table in audit_summaries:
        add_table(payload, table, question_answering, by_page[table.page.number], duplicate_index)
    if source_path is not None:
        noisy = read_vector_noisy_scatter(source_path, pdf.pages)
        figures = (read_vector_scatter(source_path, pdf.pages) + read_vector_length_scatter(source_path, pdf.pages)
                   + noisy + read_vector_noisy_contours(source_path, noisy))
        add_scatter_figures(payload, figures)
        add_histogram_figures(payload, read_vector_histograms(source_path, pdf.pages))
    return payload.as_dict()


def add_scatter_figures(payload: Payload, figures: tuple[ScatterFigure, ...]) -> None:
    datasets = {}
    experiments = {}
    for figure in figures:
        for panel_number, panel in enumerate(figure.panels, start=1):
            if panel.dataset not in datasets:
                datasets[panel.dataset] = (
                    payload.add("datasets", "dataset_id", "dataset", dataset_name=panel.dataset,
                                dataset_version=None, data_split=None),
                    payload.add("tasks", "task_id", "task", task_type="question answering"),
                )
            dataset_id, task_id = datasets[panel.dataset]
            for point_number, point in enumerate(panel.points, start=1):
                vertical = f"~{point.y:.4f}"
                shared_answer = figure.metric is not None and figure.vertical_metric == "Answer F1"
                # Only the matched precision/recall/F1 series has verified
                # point identity across multiple figures.
                series = figure.condition if shared_answer else figure.label
                key = series, panel_number, point_number, panel.dataset, panel.generator, vertical
                location = f"Panel {panel_number}, point {point_number}."
                if point.contour is not None:
                    curve, vertex = point.contour
                    location = f"Contour {curve}, vertex {vertex}. Knowledge precision ~{point.x:.4f}."
                if key not in experiments:
                    experiments[key] = (
                        payload.add("configurations", "config_id", "config",
                                    rag_variant=figure.condition, generator_model=panel.generator,
                                    top_k=None),
                        payload.add("experiments", "experiment_id", "experiment", evaluation_protocol=None,
                                    note=f"{location} {figure.condition}"),
                        None,
                    )
                config_id, experiment_id, canonical = experiments[key]
                for metric, value in ((figure.metric, point.x), (figure.vertical_metric, point.y)):
                    if metric is None:
                        continue
                    raw = f"~{value:.4f}"
                    result_id = payload.add(
                        "results", "result_id", "result", config_id=config_id, dataset_id=dataset_id,
                        task_id=task_id, experiment_id=experiment_id, metric_name=metric,
                        metric_variant=None, metric_direction="higher_is_better", metric_k=None,
                        metric_k_applicability="not_applicable", reported_value=float(raw[1:]),
                        reported_scale="fraction", raw_value=raw, raw_unit=None, aggregation=None,
                        result_origin="primary", cited_publication_id=None, extraction_status="extracted",
                        note=f"{location} Calibrated native vector coordinates "
                             "(approximate ±0.0001 fraction).",
                    )
                    if metric == "Answer F1" and shared_answer:
                        if canonical:
                            payload.results[-1]["duplicate_of"] = canonical
                        else:
                            experiments[key] = config_id, experiment_id, result_id
                    # The caption verifies the figure locator. The number comes from the point geometry.
                    payload.cite("result", result_id, "raw_value", figure.page, f"{figure.label}:",
                                 figure=figure.label, bbox=point.bbox)


def add_histogram_figures(payload: Payload, figures: tuple[HistogramFigure, ...]) -> None:
    for figure in figures:
        config_id = payload.add("configurations", "config_id", "config",
                                rag_variant="single candidate knowledge sentence", generator_model=figure.generator,
                                top_k=None)
        contexts = {}
        for bar in figure.bars:
            if bar.dataset not in contexts:
                contexts[bar.dataset] = (
                    payload.add("datasets", "dataset_id", "dataset", dataset_name=bar.dataset,
                                dataset_version=None, data_split=None),
                    payload.add("tasks", "task_id", "task", task_type="question answering"),
                    payload.add("experiments", "experiment_id", "experiment", evaluation_protocol=None,
                                note="Each candidate knowledge sentence is fed individually to the generator."),
                )
            dataset_id, task_id, experiment_id = contexts[bar.dataset]
            raw = f"~{bar.value:.4f}"
            result_id = payload.add(
                "results", "result_id", "result", config_id=config_id, dataset_id=dataset_id,
                task_id=task_id, experiment_id=experiment_id, metric_name=HISTOGRAM_METRIC,
                metric_variant=bar.metric_variant, metric_direction=None, metric_k=None,
                metric_k_applicability="not_applicable", reported_value=float(raw[1:]),
                reported_scale="fraction", raw_value=raw, raw_unit=None,
                aggregation="frequency / all individual-sentence generations",
                result_origin="primary", cited_publication_id=None, extraction_status="extracted",
                note="Approximate fraction from the calibrated height of the PDF vector bar.",
            )
            payload.cite("result", result_id, "raw_value", figure.page, f"{figure.label}:",
                         figure=figure.label, bbox=bar.bbox)


def scale_and_value(raw: str, header_unit: str | None, *,
                    grouped_digits: bool = False) -> tuple[str | None, str | None, float | None, str]:
    cell = raw[:-1] if raw.endswith(("%*", "%∗")) else raw
    suffix = CELL_UNIT.search(cell)
    unit = "%" if cell.endswith("%") else header_unit or (suffix[1] if suffix else None)
    scale = {"%": "percent", "percent": "percent", "fraction": "fraction"}.get((unit or "").lower())
    if raw in MISSING:
        return unit, scale, None, "absent"
    if suffix and suffix[1] != unit:
        return unit, scale, None, "present_not_extractable"
    cell = (cell[:suffix.start()] if suffix else cell.removesuffix("%")).strip()
    number = cell.replace("−", "-").replace(",", "" if grouped_digits else ".")
    if NONSCALAR_VALUE.fullmatch(number):
        return unit, scale, None, "extracted"
    number = APPROXIMATE_PREFIX.sub("", number)
    factor = 1
    if unit == "tokens" and number[-1:] in TOKEN_MULTIPLIER:
        factor = TOKEN_MULTIPLIER[number[-1]]
        number = number[:-1]
    if re.match(r"^[+-]?0\d", number):
        return unit, scale, None, "present_not_extractable"
    try:
        value = float(number) * factor
    except ValueError:
        return unit, scale, None, "present_not_extractable"
    if not math.isfinite(value):
        return unit, scale, None, "present_not_extractable"
    return unit, scale, value, "extracted"


def caption_fragment(table: Table) -> str:
    words = table.caption.split()
    if not words:
        return ""
    pattern = re.compile(r"(?:^[ \t]*|[ \t]{4,})(" + r"\s+".join(re.escape(word) for word in words) + r")", re.M)
    for text in (table.page.text, table.page.reading_text):
        if match := pattern.search(text):
            return match[1]
    return ""


def table_statements(table: Table, caption: str) -> tuple[list[str], list[str]]:
    source = table.page.reading_text or table.page.text
    label = " ".join(table.label.lower().split())
    explicit = [caption] if caption else []
    general = []
    for statement in re.split(r"(?<=[.!?])\s+|\n[ \t]*\n", source):
        statement = statement.strip()
        if not statement or len(statement) > 500:
            continue
        references = {" ".join(match.lower().split()) for match in TABLE_REFERENCE.findall(statement)}
        if references == {label} and not (caption and statement.startswith(caption)):
            explicit.append(statement)
        elif not references and EVALUATION_STATEMENT.match(" ".join(statement.split())):
            general.append(statement)
    return explicit, general


def unique_context(matches: list[tuple[str, str]]) -> tuple[str | None, str | None]:
    found = dict(matches)
    return next(iter(found.items())) if len(found) == 1 else (None, None)


def metric_definition(column: Column, fragments: list[str]) -> tuple[str | None, str | None]:
    name = column.header.name.lower()
    variants = [
        (variant, fragment)
        for pattern, names, variant in METRIC_VARIANTS if name in names
        for fragment in fragments
        if pattern.search(fragment) and not CONTRAST.search(fragment)
    ]
    return unique_context(variants)


def metric_variant(column: Column, page: Page) -> str | None:
    return metric_definition(column, [page.reading_text or page.text])[0]


def metric_name_alias(column: Column, page: Page) -> tuple[str | None, str | None]:
    source = page.reading_text or page.text
    matches = []
    for header, context, canonical in METRIC_NAME_ALIASES:
        if header.fullmatch(column.header.name.strip()) and (match := context.search(source)):
            matches.append((canonical, match[0]))
    return unique_context(matches)


def metric_unit(column: Column, fragments: list[str]) -> tuple[str | None, str | None]:
    name = column.header.name.lower()
    metric = r"\s+".join(re.escape(word) for word in name.split())
    unit_pattern = re.compile(
        rf"(?<![\w@-])(?:{metric}(?:\s+scores?)?|all\s+(?:reported\s+)?(?:scores|values))\s+"
        rf"{UNIT_STATEMENT}(?P<unit>{UNIT_TEXT})(?![\w%-])", re.I,
    )
    units = []
    for fragment in fragments:
        if CONTRAST.search(fragment):
            continue
        for match in unit_pattern.finditer(fragment):
            unit = match["unit"].lower()
            if unit.startswith("percent"):
                unit = "percent"
            elif unit == "fractions":
                unit = "fraction"
            units.append((unit, match[0]))
    return unique_context(units)


def add_table(payload: Payload, table: Table, question_answering: bool,
              page_tables: list[Table] | None = None,
              duplicate_index: dict[tuple[object, ...], dict[str, Any]] | None = None) -> None:
    citation = ORIGIN_CITATION.search(table.caption)
    if not citation and UNSOURCED_QUOTATION.search(table.caption):
        return
    origin = "cited" if citation else "primary"
    source = {"table": table.label}
    caption = caption_fragment(table)
    explicit, general = table_statements(table, caption)
    splits = []
    for match in SPLIT.finditer(caption):
        split_name = match[1].lower()
        splits.append(("test" if split_name == "testing" else split_name, caption))
    split, split_fragment = unique_context(splits)
    qa = question_answering or bool(TASK_TYPE.search(table.page.prose))
    page_tables = page_tables or [table]
    caption_cutoff = CAPTION_CUTOFF.search(table.caption)
    table_cutoff = int(caption_cutoff[1]) if caption_cutoff else None
    protocol_fragment = next((match[0] for fragment in explicit + (general if len(page_tables) == 1 else [])
                              if not CONTRAST.search(fragment) and (match := RAGAS_PROTOCOL.search(fragment))), None)
    column_context = {}
    column_names = {}
    for column in table.columns:
        name = column.header.name.casefold()
        if name not in column_names:
            column_names[name] = metric_name_alias(column, table.page)
        if name in column_context:
            continue
        matching_tables = sum(any(c.header.name.casefold() == name for c in candidate.columns)
                              for candidate in page_tables)
        definition_sources = explicit + (general if matching_tables == 1 else [])
        column_context[name] = metric_definition(column, definition_sources), metric_unit(column, explicit)
    datasets: dict[tuple[str | None, str | None], tuple[str, str, str]] = {}
    for column in table.columns:
        dataset = column.dataset
        dataset_key = dataset, column.original_row
        if dataset_key in datasets:
            continue
        dataset_id = payload.add(
            "datasets", "dataset_id", "dataset",
            dataset_name=dataset, dataset_version=None, data_split=column.data_split or table.data_split or split,
        )
        task_type = "question answering" if qa else None
        if qa and dataset and re.search(r"\bmulti[- ]hop\b", dataset, re.I):
            task_type = "multi-hop question answering"
        if table.task_type:
            task_type = table.task_type
        task_id = payload.add("tasks", "task_id", "task", task_type=task_type)
        experiment_id = payload.add(
            "experiments", "experiment_id", "experiment", evaluation_protocol="RAGAS" if protocol_fragment else None,
            note=table.caption,
        )
        if dataset:
            if table.family in {"failure_rates", "failure_statistics", "watermark_metrics"}:
                locator = {"table_row": column.original_row or table.row_section}
                if column.original_row:
                    locator["table_column"] = "Dataset"
                elif table.row_section == dataset:
                    locator["table_column"] = "Setting"
                payload.cite("dataset", dataset_id, "dataset_name", table.page, dataset, **locator, **source)
            else:
                payload.cite("dataset", dataset_id, "dataset_name", table.page,
                             dataset if dataset in table.page.text else table.label, table_column=dataset, **source)
        if column.data_split_evidence:
            payload.cite("dataset", dataset_id, "data_split", *column.data_split_evidence)
        elif table.data_split:
            payload.cite("dataset", dataset_id, "data_split", table.page, table.data_split,
                         table_row=table.row_section, table_column="Slice", **source)
        elif split_fragment:
            payload.cite("dataset", dataset_id, "data_split", table.page, split_fragment, **source)
        if protocol_fragment:
            payload.cite("experiment", experiment_id, "evaluation_protocol", table.page, protocol_fragment, **source)
        datasets[dataset_key] = dataset_id, task_id, experiment_id

    parent_name = ""
    for row_index, (row, values) in enumerate(table.rows):
        row_locator = f"{table.row_section} / {row}" if table.row_section else row
        if row.startswith(("-", "–", "—")) and parent_name:
            config_name = f"{parent_name} / {row}"
        else:
            parent_name = row
            config_name = row
        if table.config_name:
            config_name = table.config_name
        if table.family == "benchmark_distribution":
            config_name = None
        configurations = {}
        for column_index, (column, raw) in enumerate(zip(table.columns, values, strict=True)):
            if raw is None:
                continue
            generator = column.generator
            configuration_key = generator, column.generator_version
            if configuration_key not in configurations:
                configurations[configuration_key] = add_configuration(
                    payload, table, row, config_name, row_locator, column,
                )
            config_id = configurations[configuration_key]
            dataset_id, task_id, experiment_id = datasets[column.dataset, column.original_row]
            header = column.header
            (variant, variant_fragment), (context_unit, unit_fragment) = column_context[header.name.casefold()]
            unit, scale, value, status = scale_and_value(raw, header.unit or context_unit,
                                                         grouped_digits=column.grouped_digits)
            direction = column.direction
            if direction is None and header.name.lower() in {"latency", "response time"}:
                direction = "lower_is_better"
            if direction is None and header.name.lower() in {"em", "exact match", "f1", "accuracy", "precision", "recall"}:
                direction = "higher_is_better"
            name = column_names[header.name.casefold()][0] or header.name
            metric_variant = column.metric_variant or variant
            cutoff = header.cutoff if header.cutoff is not None else table_cutoff
            result_scale = column.reported_scale or scale
            result_id = payload.add(
                "results", "result_id", "result", experiment_id=experiment_id, config_id=config_id,
                dataset_id=dataset_id, task_id=task_id, metric_name=name,
                metric_variant=metric_variant, metric_direction=direction, metric_k=cutoff,
                metric_k_applicability="applicable" if cutoff is not None else "not_applicable",
                reported_value=value, reported_scale=result_scale, raw_value=raw, raw_unit=unit,
                aggregation=column.aggregation, result_origin=origin,
                cited_publication_id=citation[1] if citation else None, extraction_status=status,
                **({"sample_size": column.sample_size} if column.sample_size is not None else {}),
            )
            if duplicate_index is not None and column.duplicate_scope and status == "extracted":
                key = (column.duplicate_scope, config_name, *configuration_key, table.retriever_model,
                       table.retrieval_method, table.config_note, table.top_k, column.dataset,
                       column.data_split or table.data_split or split, name, metric_variant, cutoff,
                       column.aggregation, column.sample_size, unit, result_scale, raw, origin,
                       citation[1] if citation else None)
                current = payload.results[-1]
                canonical = duplicate_index.setdefault(key, current)
                if canonical is not current:
                    current["duplicate_of"] = canonical["result_id"]
                    for field in ("config_id", "dataset_id", "experiment_id", "task_id"):
                        current[field] = canonical[field]
            column_generator = column.generator_locator or (generator if not table.row_section else None)
            column_title = " / ".join(item for item in
                                      [column_generator, column.dataset, header.label, column.aggregation] if item)
            if column.original_row:
                column_title = f"{row} / {header.label}"
            if column.source_label:
                column_title = column.source_label
            bbox = table.value_bboxes[row_index][column_index] if table.value_bboxes else None
            # A bounded OCR pass has its own coordinates. Whole-page OCR can
            # merge the digits with a neighbor or recognize a different number.
            fragment = "" if bbox and table.page.verification == "ocr_match" else raw
            payload.cite("result", result_id, "raw_value", table.page, fragment,
                         table_row=column.original_row or row_locator, table_column=column_title,
                         **({"bbox": bbox} if bbox else {}), **source)
            if column.metric_variant_evidence:
                variant_page, variant_text = column.metric_variant_evidence
                locator = ({**source, "table_row": column.original_row or row_locator, "table_column": column_title}
                           if variant_page.number == table.page.number else {})
                payload.cite("result", result_id, "metric_variant", variant_page, variant_text, **locator)
            elif variant_fragment and column.metric_variant is None:
                payload.cite("result", result_id, "metric_variant", table.page, variant_fragment,
                             table_column=column_title, **source)
            if column.sample_size_evidence:
                payload.cite("result", result_id, "sample_size", *column.sample_size_evidence)
            elif column.sample_size is not None:
                payload.cite("result", result_id, "sample_size", table.page, str(column.sample_size),
                             table_row=column.original_row or row_locator, table_column=column.sample_size_label, **source)
            alias, alias_fragment = column_names[header.name.casefold()]
            if alias and alias_fragment:
                payload.cite("result", result_id, "metric_name", table.page, alias_fragment,
                             table_column=column_title, **source)
            elif column.name_evidence:
                payload.cite("result", result_id, "metric_name", *column.name_evidence)
            if column.aggregation_evidence:
                payload.cite("result", result_id, "aggregation", *column.aggregation_evidence)
            if unit_fragment and not header.unit and not raw.endswith("%"):
                payload.cite("result", result_id, "raw_unit", table.page, unit_fragment,
                             table_column=column_title, **source)


def add_configuration(payload: Payload, table: Table, row: str, name: str | None,
                      row_locator: str, column: Column) -> str:
    cutoff = TOP_K.search(row) if name is not None else None
    generator = column.generator
    context = {"retriever_model": table.retriever_model} if table.retriever_model else {}
    if table.retrieval_method:
        context["retrieval_method"] = table.retrieval_method
    if table.config_note:
        context["note"] = table.config_note
    if column.generator_version:
        context["generator_version"] = column.generator_version
    config_id = payload.add(
        "configurations", "config_id", "config", rag_variant=name,
        generator_model=generator, top_k=int(cutoff[1]) if cutoff else table.top_k, **context,
    )
    if name is not None:
        payload.cite("configuration", config_id, "rag_variant", table.page, row,
                     table_row=row_locator, table=table.label)
    if cutoff:
        payload.cite("configuration", config_id, "top_k", table.page, cutoff[0],
                     table_row=row_locator, table=table.label)
    elif table.top_k_evidence:
        payload.cite("configuration", config_id, "top_k", *table.top_k_evidence)
    if table.retriever_model:
        payload.cite("configuration", config_id, "retriever_model", table.page, table.retriever_model, table=table.label)
    if table.retrieval_method:
        payload.cite("configuration", config_id, "retrieval_method", table.page, table.retrieval_method, table=table.label)
    if table.config_note_evidence:
        payload.cite("configuration", config_id, "note", *table.config_note_evidence)
    elif table.config_note:
        payload.cite("configuration", config_id, "note", table.page, table.config_note,
                     table=table.label, table_row=row_locator)
    if column.generator_evidence:
        payload.cite("configuration", config_id, "generator_model", *column.generator_evidence)
    elif generator:
        if table.family == "failure_statistics" and column.original_row:
            locator = {"table_row": column.original_row, "table_column": "Model"}
        elif table.row_section and not column.generator_locator:
            locator = {"table_row": table.row_section}
        else:
            locator = {"table_column": column.generator_locator or generator}
        payload.cite("configuration", config_id, "generator_model", table.page, generator,
                     table=table.label, **locator)
    if column.generator_version_evidence:
        payload.cite("configuration", config_id, "generator_version", *column.generator_version_evidence)
    return config_id
