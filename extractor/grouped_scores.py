"""Read captioned native EM/F1 grids whose dataset groups and average are explicit."""

from dataclasses import replace
from decimal import Decimal
import re

from pdf_pages import Page
from rag_tables import (
    Column, HEADER_CELLS, Header, NATIVE_CAPTION, Table, TOKEN, aligned_label,
    aligned_values, column_boundaries,
)


METHOD = re.compile(r"methods?|variants?", re.I)
BACKBONE = re.compile(r"(?=[\w.-]*\d)[\w.-]+(?: [\w-]+(?:/[\w-]+)+)?")
BACKBONE_CAPTION = re.compile(r"\b(?:backbones?|generator models?)\b", re.I)
DAGGER_LEGEND = re.compile(r"†\s+(?:indicates?|denotes?)\s+(?:statistically\s+)?significan", re.I)
FLASHRAG_METRICS = re.compile(
    r"\b(?:we\s+)?(?:report|evaluate)\s+Exact\s+Match\s*\(EM\)\s+and\s+F1(?:\s+scores?)?"
    r"\s+using\s+the\s+FlashRAG\s+evaluation\s+frame(?:-\s*)?work\b", re.I,
)
MACRO_SCORES = re.compile(r"\bmacro[-\s]+(?:avg|average)\s+EM\s+and\s+F1\b", re.I)
OTHER_AGGREGATION = re.compile(r"\b(?:micro[-\s]+(?:avg|average)|weighted|pooled|median)\b", re.I)
DECIMAL = re.compile(r"[+−-]?\d+(?:\.\d+)?")


def enrich_grouped_scores(pages: list[Page], tables: list[Table]) -> list[Table]:
    # The named framework aggregates its EM and F1 scores over evaluation items.
    # https://github.com/RUC-NLPIR/FlashRAG/blob/main/flashrag/evaluator/metrics.py
    protocol = native_evidence(pages, FLASHRAG_METRICS)
    if protocol is None:
        return tables
    macro = native_evidence(pages, MACRO_SCORES)
    extracted = []
    for table in tables:
        if table.family != "grouped_scores" or OTHER_AGGREGATION.search(table.caption):
            extracted.append(table)
            continue
        columns = []
        for index, column in enumerate(table.columns):
            if column.dataset:
                column = replace(column, aggregation="mean per question", aggregation_evidence=protocol)
            elif macro and is_dataset_mean(table, index):
                column = replace(column, aggregation="macro-average", aggregation_evidence=macro)
            columns.append(column)
        extracted.append(replace(table, columns=columns))
    return extracted


def native_evidence(pages: list[Page], pattern: re.Pattern[str]) -> tuple[Page, str] | None:
    for page in pages:
        if page.verification == "text_match" and (match := pattern.search(page.reading_text or page.text)):
            return page, match[0]
    return None


def is_dataset_mean(table: Table, index: int) -> bool:
    # The reported average must be consistent with a mean over the visible
    # dataset columns, allowing half of the last printed digit in each cell.
    metric = table.columns[index].header.name
    sources = [i for i, column in enumerate(table.columns) if column.dataset and column.header.name == metric]
    for _, cells in table.rows:
        raw = [cells[i] for i in [*sources, index]]
        if not all(DECIMAL.fullmatch(value) for value in raw):
            return False
        *scores, average = [Decimal(value.replace("−", "-")) for value in raw]
        errors = [Decimal("0.5").scaleb(score.as_tuple().exponent) for score in [*scores, average]]
        distance = abs(sum(scores) - average * len(scores))
        if distance > sum(errors[:-1]) + errors[-1] * len(scores):
            return False
    return True


def read_grouped_scores(page: Page, lines: list[str]) -> tuple[list[Table], set[int]]:
    tables = []
    claimed = set()
    if page.verification != "text_match":
        return tables, claimed

    caption_at = -1
    caption_match = None
    for index, line in enumerate(lines):
        if match := NATIVE_CAPTION.match(line):
            caption_at, caption_match = index, match
            continue
        if caption_match is None or not 3 <= index - caption_at <= 8:
            continue
        metrics = list(TOKEN.finditer(line))
        pairs = len(metrics) // 2
        if not 3 <= pairs <= 12 or tuple(metric[0] for metric in metrics) != ("EM", "F1") * pairs:
            continue
        methods = list(HEADER_CELLS.finditer(lines[index - 1].rstrip()))
        if not methods or not METHOD.fullmatch(methods[-1][0]) or methods[-1].end() >= metrics[0].start():
            continue
        groups = list(HEADER_CELLS.finditer(lines[index - 2].rstrip()))
        if not groups or groups[-1][0].casefold() not in {"average", "avg."}:
            continue
        # A malformed group heading must not fall through to proximity-based assignment.
        claimed.add(index)
        if len(groups) != pairs:
            continue
        if len({group[0].casefold() for group in groups}) != pairs:
            continue
        centers = [(metric.start() + metric.end()) / 2 for metric in metrics]
        if any(not centers[2 * i] < (group.start() + group.end()) / 2 < centers[2 * i + 1]
               for i, group in enumerate(groups)):
            continue
        if any(not re.search(r"[^\W\d_]", group[0]) for group in groups):
            continue

        columns = []
        for i, (metric, center) in enumerate(zip(metrics, centers, strict=True)):
            group = groups[i // 2][0]
            average = i // 2 == pairs - 1
            label = f"{group} / {metric[0]}" if average else metric[0]
            columns.append(Column(Header(label, metric[0], center, None, None),
                                  None if average else group, "reported average" if average else None))
        caption = " ".join(" ".join(lines[caption_at:index - 2]).split())
        table = Table(page, caption_match[1], caption, index, columns, [], family="grouped_scores")
        tables.extend(read_sections(table, lines, index + 1, methods[-1].start(), column_boundaries(centers)))
    return tables, claimed


def read_sections(table: Table, lines: list[str], start: int, method_at: int,
                  boundaries: list[float]) -> list[Table]:
    sections = []
    section = None
    rows = []
    annotation = "†" if DAGGER_LEGEND.search(table.caption) else ""
    for raw in lines[start:start + 45]:
        line = raw.rstrip()
        if not line or NATIVE_CAPTION.match(line):
            break
        if len(line) < method_at:
            # A category spanning several rows may appear alone in the Type column.
            continue
        if (BACKBONE_CAPTION.search(table.caption) and abs(len(line) - len(line.lstrip()) - method_at) <= 2
                and BACKBONE.fullmatch(line.strip())):
            if rows:
                sections.append((section, rows))
                rows = []
            section = line.strip()
            continue
        aligned = aligned_values(line, boundaries, annotation)
        if not aligned:
            break
        value_at, values = aligned
        method = aligned_label(line, method_at, value_at)
        if not method:
            break
        rows.append((method, values))
    if rows:
        sections.append((section, rows))
    if sum(len(rows) for _, rows in sections) < 2:
        return []
    return [
        replace(table, row_section=section, rows=rows,
                columns=[replace(column, generator=section if section and "/" not in section else None)
                         for column in table.columns])
        for section, rows in sections
    ]
