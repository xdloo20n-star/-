"""Map PDF captions, headers, and cells to the Sverka extraction contract."""

from dataclasses import dataclass, field
import math
import re
from typing import Any

from pdf_pages import PDF, Page
from rag_tables import Column, MISSING, Table, read_tables

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
             fragment: str, **locator: str) -> None:
        if fragment and (fragment in page.text or fragment in page.reading_text):
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


def extract_publication(pdf: PDF) -> dict[str, Any]:
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
    for page in pdf.pages:
        tables = read_tables(page)
        for table in tables:
            add_table(payload, table, question_answering, tables)
    return payload.as_dict()


def scale_and_value(raw: str, header_unit: str | None) -> tuple[str | None, str | None, float | None, str]:
    unit = "%" if raw.endswith("%") else header_unit
    scale = {"%": "percent", "percent": "percent", "fraction": "fraction"}.get((unit or "").lower())
    number = raw.removesuffix("%").replace("−", "-").replace(",", ".")
    if raw in MISSING:
        return unit, scale, None, "absent"
    if re.match(r"^[+-]?0\d", number):
        return unit, scale, None, "present_not_extractable"
    value = float(number)
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
              page_tables: list[Table] | None = None) -> None:
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
            dataset_name=dataset, dataset_version=None, data_split=split,
        )
        task_type = "question answering" if qa else None
        if qa and dataset and re.search(r"\bmulti[- ]hop\b", dataset, re.I):
            task_type = "multi-hop question answering"
        task_id = payload.add("tasks", "task_id", "task", task_type=task_type)
        experiment_id = payload.add(
            "experiments", "experiment_id", "experiment", evaluation_protocol="RAGAS" if protocol_fragment else None,
            note=table.caption,
        )
        if dataset:
            payload.cite("dataset", dataset_id, "dataset_name", table.page,
                         dataset if dataset in table.page.text else table.label, table_column=dataset, **source)
        if split_fragment:
            payload.cite("dataset", dataset_id, "data_split", table.page, split_fragment, **source)
        if protocol_fragment:
            payload.cite("experiment", experiment_id, "evaluation_protocol", table.page, protocol_fragment, **source)
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
            (variant, variant_fragment), (context_unit, unit_fragment) = column_context[header.name.casefold()]
            unit, scale, value, status = scale_and_value(raw, header.unit or context_unit)
            direction = "lower_is_better" if header.name.lower() in {"latency", "response time"} else None
            if direction is None and header.name.lower() in {"em", "exact match", "f1", "accuracy", "precision", "recall"}:
                direction = "higher_is_better"
            result_id = payload.add(
                "results", "result_id", "result", experiment_id=experiment_id, config_id=config_id,
                dataset_id=dataset_id, task_id=task_id, metric_name=column_names[header.name.casefold()][0] or header.name,
                metric_variant=variant, metric_direction=direction,
                metric_k=header.cutoff if header.cutoff is not None else table_cutoff,
                metric_k_applicability="applicable" if header.cutoff is not None or table_cutoff is not None else "not_applicable",
                reported_value=value, reported_scale=scale, raw_value=raw, raw_unit=unit,
                aggregation=column.aggregation, result_origin=origin,
                cited_publication_id=citation[1] if citation else None, extraction_status=status,
            )
            column_title = " / ".join(item for item in [generator, column.dataset, header.label, column.aggregation] if item)
            if column.original_row:
                column_title = f"{row} / {header.label}"
            payload.cite("result", result_id, "raw_value", table.page, raw,
                         table_row=column.original_row or row, table_column=column_title, **source)
            if variant_fragment:
                payload.cite("result", result_id, "metric_variant", table.page, variant_fragment,
                             table_column=column_title, **source)
            alias, alias_fragment = column_names[header.name.casefold()]
            if alias and alias_fragment:
                payload.cite("result", result_id, "metric_name", table.page, alias_fragment,
                             table_column=column_title, **source)
            if unit_fragment and not header.unit and not raw.endswith("%"):
                payload.cite("result", result_id, "raw_unit", table.page, unit_fragment,
                             table_column=column_title, **source)


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
