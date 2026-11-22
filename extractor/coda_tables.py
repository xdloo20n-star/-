"""Read CoFaithfulQA recall and CoDA latency from positioned PDF tables."""

from dataclasses import replace
import re

from failure_rates import visual_rows
from grouped_scores import native_evidence
from native_benchmark_tables import aligned, body_above, normalized_boxes, positioned_caption, words_text
from pdf_pages import Page, Word
from rag_tables import Column, Header, Table


CAPTION = re.compile(
    r"(?:^ *|(?<= {2}))(Table \d+): (Contextual faithfulness on CoFaithfulQA\.|Inference latency with and without CoDA\.|"
    r"Inference latency under different prompt)", re.M,
)
DEFINITION = re.compile(
    r"\s+".join("ConR measures the proportion of generated facts that are explicitly supported by the retrieved context".split())
    + r"\."
)
SUBSETS = ("HotPotQA", "NQ", "NewsQA", "SearchQA", "SQuAD", "TriviaQA")
TIMING_HEADER = "Baseline (s) CoDA (s)"
TIMING_COLUMNS = (("Baseline (s)", "Latency", "s", "Baseline"),
                  ("CoDA (s)", "Latency", "s", "CoDA"),
                  ("∆(s)", "Latency increment", "s", "CoDA"),
                  ("Overhead (%)", "Relative latency overhead", "%", "CoDA"))
DECIMAL = re.compile(r"\d+\.\d+")
PERCENT = re.compile(r"\d+\.\d+%")
GENERATOR = re.compile(r"Llama\d(?:\.\d+)?-\d+B")
LATENCY_DATASETS = {"RAGTruth", "Dolly (AC)"}


def read_coda_tables(pages: list[Page], known: list[Table]) -> list[Table]:
    occupied = {(table.page.number, table.label.casefold()) for table in known}
    definition = native_evidence(pages, DEFINITION)
    if definition is None:
        return []
    tables = []
    for page in pages:
        if page.verification != "text_match" or not page.words or not page.size:
            continue
        for match in CAPTION.finditer(page.text):
            label, prefix = match.groups()
            if (page.number, label.casefold()) in occupied:
                continue
            rows = visual_rows(page.words)
            caption = positioned_caption(rows, label, re.compile(re.escape(prefix)))
            if caption is None:
                continue
            width, _ = page.size
            left, right = (0, width / 2) if caption.x < width / 2 else (width / 2, width)
            grid = body_above(rows, caption, left, right, 155)
            if prefix.startswith("Contextual"):
                tables.extend(recall_grid(page, label, match[0].strip(), grid, definition))
            else:
                tables.extend(latency_grid(page, label, match[0].strip(), grid, prefix.endswith("prompt")))
    return tables


def recall_grid(page: Page, label: str, caption: str, grid: list[list[Word]],
                definition: tuple[Page, str]) -> list[Table]:
    starts = [i for i, row in enumerate(grid) if tuple(w.text for w in row) == ("Models", *SUBSETS)]
    if len(starts) != 1:
        return []
    heading, *body = grid[starts[0]:]
    if not body or words_text(body[0]).replace(" ↑", "") != " ".join(["ConR"] * len(SUBSETS)):
        return []
    body = body[1:]
    if body and all(word.text == "↑" for word in body[0]):
        body = body[1:]
    columns = [Column(Header("ConR", "ConR", w.x, None, None), w.text, source_label=f"{w.text} / ConR",
                      metric_variant="context-supported facts / all generated facts", metric_variant_evidence=definition,
                      reported_scale="percent", direction="higher_is_better") for w in heading[1:]]
    rows = []
    boxes = []
    centers = [column.header.center for column in columns]
    for row in body:
        cells = row[-len(columns):]
        method = words_text(row[:-len(columns)]).removesuffix(" (Ours)")
        if not method or not aligned(cells, centers, 6) or any(
            not DECIMAL.fullmatch(w.text) or float(w.text) > 100 for w in cells
        ):
            return []
        rows.append((method, [w.text for w in cells]))
        boxes.extend(normalized_boxes(page, cells))
    if len(rows) < 2 or len({name for name, _ in rows}) != len(rows) or all(
        float(value) <= 1 for _, values in rows for value in values
    ):
        return []
    return [Table(page, label, caption, 0, columns, rows, value_bboxes=tuple(boxes))]


def latency_grid(page: Page, label: str, caption: str, grid: list[list[Word]], prompts: bool) -> list[Table]:
    heading = (f"Prompt Length (tokens) {TIMING_HEADER}" if prompts
               else f"Dataset Model {TIMING_HEADER} ∆(s) Overhead (%)")
    starts = [i for i, row in enumerate(grid) if words_text(row) == heading]
    if len(starts) != 1:
        return []
    header, *body = grid[starts[0]:]
    centers = ([(header[3].x + header[4].x) / 2, (header[5].x + header[6].x) / 2] if prompts else
               [(header[2].x + header[3].x) / 2, (header[4].x + header[5].x) / 2, header[6].x,
                (header[7].x + header[8].x) / 2])
    specs = TIMING_COLUMNS[:len(centers)]
    dataset = None
    seen = set()
    tables = []
    for row in body:
        if not prompts and words_text(row) in LATENCY_DATASETS:
            dataset = words_text(row)
            continue
        if len(row) != len(specs) + 1 or not aligned(row[1:], centers, 9):
            return []
        row_name = row[0].text
        if (dataset, row_name) in seen or (
            not row_name.isdecimal() if prompts else not dataset or not GENERATOR.fullmatch(row_name)
        ):
            return []
        seen.add((dataset, row_name))
        for word, (title, name, unit, method) in zip(row[1:], specs, strict=True):
            if not (PERCENT if unit == "%" else DECIMAL).fullmatch(word.text):
                return []
            column = Column(Header(title, name, word.x, None, unit), dataset, source_label=title)
            if prompts:
                column = replace(column, metric_variant=f"input prompt {row_name} tokens",
                                 metric_variant_evidence=(page, row_name))
            else:
                column = replace(column, generator=row_name, generator_evidence=(page, row_name))
            tables.append(Table(page, label, caption, len(tables), [column], [(row_name, [word.text])],
                                row_section=dataset, config_name=method, value_bboxes=normalized_boxes(page, [word])))
    return tables if len(seen) >= 2 else []
