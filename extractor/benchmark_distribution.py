"""Read explicitly labelled benchmark composition and query-distribution grids."""

from decimal import Decimal
import re

from pdf_pages import Page
from rag_tables import Column, Header, Table


BENCHMARK = "RAG-Multi-Corpus"
COMPOSITION_HEADER = ("Enterprise", "Domain", "Files", "Queries")
DISTRIBUTION_HEADER = ("Category", "Count", "Percentage", "Description")
COMPOSITION_CAPTION = re.compile(r"^(Table \d+): Composition of the (RAG-Multi-Corpus) benchmark across enterprises and domains\.$")
DISTRIBUTION_CAPTION = re.compile(r"^(Table \d+): Distribution of query categories in the (RAG-Multi-Corpus) benchmark\.$")
PERCENT = re.compile(r"\d+(?:\.\d+)?%")


def read_benchmark_distributions(pages: list[Page], known: list[Table]) -> list[Table]:
    occupied = {(table.page.number, table.label.casefold()) for table in known}
    found = []
    for page in pages:
        if page.verification != "text_match":
            continue
        lines = page.text.splitlines()
        for index, line in enumerate(lines):
            headings = tuple(cells(line))
            if headings == COMPOSITION_HEADER:
                caption = COMPOSITION_CAPTION
            elif headings == DISTRIBUTION_HEADER:
                caption = DISTRIBUTION_CAPTION
            else:
                continue
            grid = read_grid(lines, index, caption)
            if grid is None or (page.number, grid[0][1].casefold()) in occupied:
                continue
            label, caption_text, rows = grid
            if headings == COMPOSITION_HEADER:
                found.extend(composition_rows(page, label[1], caption_text, index, rows))
            else:
                found.extend(distribution_rows(page, label[1], caption_text, index, rows))
    return found


def cells(line: str) -> list[str]:
    return re.split(r"\s{2,}", line.strip())


def read_grid(lines: list[str], index: int, caption: re.Pattern[str]) -> tuple[re.Match[str], str, list[list[str]]] | None:
    rows = []
    for line in lines[index + 1:index + 17]:
        if match := caption.fullmatch(line.strip()):
            if len(rows) >= 3 and len({row[0] for row in rows}) == len(rows) and rows[-1][0] == "Total":
                return match, line.strip(), rows
            return None
        row = cells(line)
        if len(row) != 4:
            return None
        rows.append(row)
    return None


def group_name(row: str) -> str:
    return f"{BENCHMARK} / {'Overall' if row == 'Total' else row}"


def composition_rows(page: Page, label: str, caption: str, index: int, rows: list[list[str]]) -> list[Table]:
    if rows[-1][1] != "—" or not all(row[2].isdecimal() and row[3].isdecimal() for row in rows):
        return []
    if any(sum(int(row[column]) for row in rows[:-1]) != int(rows[-1][column]) for column in (2, 3)):
        return []
    results = []
    for position, (name, _, documents, queries) in enumerate(rows):
        dataset = group_name(name)
        source = (page.number, BENCHMARK) if name == "Total" else None
        columns = [Column(Header("Files", "Documents", 0, None, None), dataset, source_label="Files"),
                   Column(Header("Queries", "Queries", 0, None, None), dataset, source_label="Queries",
                          duplicate_scope=source)]
        results.append(Table(page, label, caption, index + position, columns, [(name, [documents, queries])],
                             family="benchmark_distribution"))
    return results


def distribution_rows(page: Page, label: str, caption: str, index: int, rows: list[list[str]]) -> list[Table]:
    if rows[-1][3] != "—" or not all(row[1].isdecimal() and PERCENT.fullmatch(row[2]) for row in rows):
        return []
    total = int(rows[-1][1])
    if total <= 0 or sum(int(row[1]) for row in rows[:-1]) != total:
        return []
    estimates = [Decimal(row[2].removesuffix("%")) for row in rows]
    if any((Decimal(row[1]) * 100 / total).quantize(estimate) != estimate
           for row, estimate in zip(rows, estimates, strict=True)):
        return []
    denominator = " ".join(rows[-1])
    results = []
    for position, (name, count, percent, _) in enumerate(rows):
        dataset = group_name(name)
        source = (page.number, BENCHMARK) if name == "Total" else None
        columns = [Column(Header("Count", "Queries", 0, None, None), dataset, source_label="Count",
                          duplicate_scope=source),
                   Column(Header("Percentage", "Query proportion", 0, None, "%"), dataset,
                          source_label="Percentage", sample_size=total, sample_size_evidence=(page, denominator))]
        results.append(Table(page, label, caption, index + position, columns, [(name, [count, percent])],
                             family="benchmark_distribution"))
    return results
