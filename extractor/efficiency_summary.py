"""Locate ingestion summary cells and compare them with the organization breakdown."""

from collections import defaultdict
from decimal import Decimal
import re

from failure_rates import visual_rows
from native_benchmark_tables import normalized_boxes, positioned_caption, words_text
from pdf_pages import Page, Word
from rag_tables import Column, Header, NATIVE_CAPTION, Table
from retrieval_tables import BENCHMARK_NAME, EFFICIENCY_CAPTION, EFFICIENCY_HEADINGS, EFFICIENCY_NUMBER, source_statement


CAPTION = re.compile(r"Aggregate Efficiency Summary\b")
METHODS = ("Agentic Chunking", "W-RAC")
HEADINGS = {
    "Total Input Tokens": 1, "Total Output Tokens": 2,
    "Average Input Tokens per File": 5, "Average Output Tokens per File": 6,
    "Total Processing Time (s)": 3, "Average Time per File (s)": 7,
    "P90 Time (s)": 8, "P95 Time (s)": 9,
}
ORGANIZATION_SCOPE = re.compile(r"\bwe report both organization-level and\s+aggregate metrics\b", re.I)
CHANGE = re.compile(r"[+−-]\d+(?:\.\d+)?%")


def read_efficiency_summaries(pages: list[Page], tables: list[Table]) -> list[Table]:
    native = [page for page in pages if page.verification == "text_match"]
    scope = source_statement(native, ORGANIZATION_SCOPE)
    benchmark = source_statement(native, BENCHMARK_NAME)
    dataset = "RAG-Multi-Corpus" if benchmark else None
    replacements = {}
    for page in native:
        if not page.words or not CAPTION.search(page.text):
            continue
        source = organizations(tables, page.number)
        if not source:
            continue
        grid = visual_rows(page.words)
        for index, line in enumerate(page.text.splitlines()):
            locator = NATIVE_CAPTION.search(line)
            if not locator or not CAPTION.match(line[locator.end():].lstrip()):
                continue
            label = locator[1]
            anchor = positioned_caption(grid, label, CAPTION)
            if anchor and (body := summary_rows(grid, anchor)):
                caption = " ".join(line.split())
                replacements[page.number, label.casefold()] = [
                    summary_cell(page, label, caption, index, printed, method, value, source[method], dataset, scope)
                    for printed, values in body for method, value in zip(METHODS, values, strict=True)
                ]
    return ([table for table in tables if (table.page.number, table.label.casefold()) not in replacements]
            + [table for group in replacements.values() for table in group])


def organizations(tables: list[Table], summary_page: int) -> dict[str, list[list[str]]]:
    sources = defaultdict(dict)
    expected = [heading for heading, *_ in EFFICIENCY_HEADINGS]
    for table in tables:
        if table.page.number != summary_page - 1 or not EFFICIENCY_CAPTION.search(table.caption):
            continue
        if not table.row_section or [c.header.label for c in table.columns] != expected or len(table.rows) != 1:
            continue
        method, values = table.rows[0]
        if (method not in METHODS or table.row_section in sources[method] or len(values) != len(expected)
                or any(not value or not EFFICIENCY_NUMBER.fullmatch(value) for value in values)):
            return {}
        sources[method][table.row_section] = values
    if any(len(sources.get(method, {})) < 2 for method in METHODS):
        return {}
    first, second = (sources[method] for method in METHODS)
    if first.keys() != second.keys() or any(first[key][4] != second[key][4] for key in first):
        return {}
    return {method: list(group.values()) for method, group in sources.items()}


def summary_rows(grid: list[list[Word]], anchor: Word) -> list[tuple[str, list[Word]]]:
    for index, row in enumerate(grid):
        if 0 < anchor.y - row[0].y < 140 and words_text(row) == "Metric Agentic Chunking W-RAC Relative Change":
            break
    else:
        return []
    body = [line for line in grid[index + 1:] if line[0].y < anchor.y - 3]
    if len(body) != len(HEADINGS):
        return []
    right_edges = (row[2].x1, row[3].x1, row[5].x1)
    rows = []
    for line, heading in zip(body, HEADINGS, strict=True):
        values = line[-3:]
        if words_text(line[:-3]) != heading or abs(line[0].x0 - row[0].x0) > 5:
            return []
        if any(abs(word.x1 - edge) > 6 for word, edge in zip(values, right_edges, strict=True)):
            return []
        if not all(EFFICIENCY_NUMBER.fullmatch(word.text) for word in values[:2]) or not CHANGE.fullmatch(values[2].text):
            return []
        rows.append((heading, values[:2]))
    return rows


def display_interval(raw: str) -> tuple[Decimal, Decimal]:
    number = Decimal(raw.replace(",", ""))
    # The decimal's last printed digit determines the rounding interval.
    half = Decimal(0) if "." not in raw else Decimal("0.5").scaleb(number.as_tuple().exponent)
    return number - half, number + half


def summary_cell(page, label, caption, index, printed, method, value, organization_rows, dataset, scope):
    position = HEADINGS[printed]
    _, name, unit, _ = EFFICIENCY_HEADINGS[position]
    is_total = printed.startswith("Total ")
    intervals = [display_interval(row[position]) for row in organization_rows]
    low, high = map(sum, zip(*intervals, strict=True))
    divisor = 1 if is_total else len(intervals)
    shown_low, shown_high = display_interval(value.text)
    confirmed = shown_low <= high / divisor and shown_high >= low / divisor
    aggregation = None
    if confirmed:
        aggregation = "sum across organizations" if is_total else "reported aggregate; predominantly macro over organizations"
    column = Column(Header(printed, name, value.x, None, unit), dataset, aggregation,
                    original_row=printed, source_label=method, grouped_digits=True,
                    aggregation_evidence=(scope[0], scope[1][0]) if scope and aggregation else None)
    return Table(page, label, caption, index, [column], [(method, [value.text])],
                 task_type="document chunking", value_bboxes=normalized_boxes(page, [value]))
