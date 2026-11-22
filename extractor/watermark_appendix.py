"""Read positioned diagnostic measurements from watermark benchmark appendices."""

import re

from failure_rates import visual_rows
from native_benchmark_tables import aligned, normalized_boxes, positioned_caption, words_text
from pdf_pages import Page, Word
from rag_tables import Column, Header, NATIVE_CAPTION, Table
from retrieval_tables import source_statement
from watermark_tables import RETRIEVAL_PROTOCOL, resolve_dataset


QUOTA = re.compile(r"Query Quotas for Reliable Detection\b", re.I)
EXAMPLES = re.compile(r"Factual Incorrect/Unreasonable Texts\b", re.I)
REMOVAL = re.compile(r"Unrelated Content Removal Attack\b", re.I)
SUBSTITUTION = re.compile(r"Synonym Substitution attack\b", re.I)
LOW_CAPACITY = re.compile(r"RAG Setting Restrictions on low-capacity\b", re.I)
EXAMPLE_PROOF = re.compile(r"\bWe select several of the high perplexity texts from\s+the generated texts by (RAG-WM)\b", re.I)
CAPACITY_PROOF = re.compile(r"\bWe reduce the watermark capacity to (\d+) bits and\s+segment it so that each segment is "
                            r"(\d+) bit long\..{0,160}?robustness of (KMW) against contextual summarization and style-control "
                            r"system prompts\. The\s+results, reported in (Table \d+),", re.I | re.S)
DEFAULT_DATASET = re.compile(r"\bUnless otherwise specified, the evaluation is carried\s+out on watermarked (NFCorpus)\b", re.I)
SUMMARY_RETRIEVAL = re.compile(r"\bwe applied context compression by retrieving the top-(\d+) texts and using "
                               r"GPT-4\.1\s+to summarize them\b", re.I)
COUNT = re.compile(r"[1-9]\d*")
DECIMAL = re.compile(r"\d+\.\d+")
FRACTION = re.compile(r"(?:0?\.\d+|1(?:\.0+)?)")
SCHEMES = ("WARD", "RAG-WM", "KMW(1)", "KMW")


def read_watermark_appendix(pages: list[Page], datasets: set[str], occupied: set[tuple[int, str]]) -> list[Table]:
    retrieval = source_statement(pages, RETRIEVAL_PROTOCOL)
    capacity = source_statement(pages, CAPACITY_PROOF)
    default_dataset = source_statement(pages, DEFAULT_DATASET)
    summary_retrieval = source_statement(pages, SUMMARY_RETRIEVAL)
    tables = []
    for page in pages:
        if page.verification != "text_match" or not page.words or not page.size:
            continue
        rows = visual_rows(page.words)
        locators = (match for line in page.text.splitlines() for match in NATIVE_CAPTION.finditer(line))
        for locator in locators:
            label = locator[1]
            prefix = locator.string[locator.end():].lstrip()
            if (page.number, label.casefold()) in occupied:
                continue
            for pattern, height in ((QUOTA, 72), (EXAMPLES, 117), (REMOVAL, 61), (SUBSTITUTION, 56), (LOW_CAPACITY, 72)):
                if not pattern.match(prefix) or not (anchor := positioned_caption(rows, label, pattern)):
                    continue
                grid = following_region(page, rows, anchor, height)
                if pattern is QUOTA:
                    tables.extend(quota_grid(page, label, grid))
                elif pattern is EXAMPLES:
                    tables.extend(example_grid(page, label, grid))
                elif pattern is LOW_CAPACITY:
                    tables.extend(capacity_grid(page, label, grid, capacity, default_dataset, summary_retrieval, retrieval))
                else:
                    tables.extend(wrr_grid(page, label, grid, datasets, retrieval, removal=pattern is REMOVAL))
    return tables


def following_region(page: Page, rows: list[list[Word]], anchor: Word, height: int) -> list[list[Word]]:
    middle = page.size[0] / 2
    bounds = (0, middle) if anchor.x < middle else (middle, page.size[0])
    return [part for row in rows if (part := [word for word in row
            if bounds[0] <= word.x < bounds[1] and anchor.y - 2 <= word.y < anchor.y + height])]


def quota_grid(page: Page, label: str, grid: list[list[Word]]) -> list[Table]:
    starts = [i for i, row in enumerate(grid) if tuple(w.text for w in row) == SCHEMES]
    if len(starts) != 1 or starts[0] + 2 != len(grid):
        return []
    header, row = grid[starts[0]:]
    cells = row[2:]
    if (words_text(row[:2]) != "Query Quota" or not aligned(cells, [word.x for word in header], 7)
            or any(not COUNT.fullmatch(cell.text) for cell in cells)):
        return []
    caption = " ".join(words_text(row) for row in grid[:starts[0]])
    return [Table(page, label, caption, int(word.y),
                  [Column(Header("Query Quota", "Queries required for stable detection", word.x, None, None), None,
                          original_row="Query Quota", source_label=scheme)], [(scheme, [word.text])],
                  config_name=scheme, value_bboxes=normalized_boxes(page, [word]))
            for scheme, word in zip(SCHEMES, cells, strict=True)]


def example_grid(page: Page, label: str, grid: list[list[Word]]) -> list[Table]:
    scope = source_statement([page], EXAMPLE_PROOF)
    starts = [i for i, row in enumerate(grid) if words_text(row) == "PPL Texts"]
    if scope is None or len(starts) != 1:
        return []
    index = starts[0]
    marker, text_head = grid[index]
    words = [word for row in grid[index + 1:] for word in row]
    scores = [word for word in words if word.x < text_head.x0 - 2]
    texts = [word for word in words if word.x >= text_head.x0 - 2]
    if len(scores) < 2 or any(not DECIMAL.fullmatch(w.text) or abs(w.x - marker.x) > 9 for w in scores):
        return []
    # Multiline text wraps around the vertically centered PPL number.
    assignments = [[word for word in texts if abs(word.y - score.y) <= 6.3] for score in scores]
    if sum(map(len, assignments)) != len(texts) or {w for group in assignments for w in group} != set(texts):
        return []
    caption = " ".join(words_text(row) for row in grid[:index])
    tables = []
    for score, assigned in zip(scores, assignments, strict=True):
        example = words_text(assigned)
        if not example.endswith("."):
            return []
        column = Column(Header("PPL", "Example text perplexity", score.x, None, None),
                        "Selected RAG-WM generated examples", metric_variant=example,
                        metric_variant_evidence=(page, example), source_label="PPL")
        tables.append(Table(page, label, caption, int(score.y), [column], [(example, [score.text])],
                            config_name=scope[1][1], value_bboxes=normalized_boxes(page, [score])))
    return tables


def wrr_grid(page: Page, label: str, grid: list[list[Word]], datasets: set[str], retrieval,
             *, removal: bool) -> list[Table]:
    reference = (r"illustrates that (KMW) can resist this" if removal
                 else r"exhibits the W\s*RR of (KMW) under\s+synonym substitution")
    scope = source_statement([page], re.compile(rf"\b{re.escape(label)}\s+{reference}\b", re.I))
    starts = [i for i, row in enumerate(grid) if row[0].text == "WRR"]
    if scope is None or len(starts) != 1 or starts[0] < 2:
        return []
    index = starts[0]
    header, row = grid[index - 1:index + 1]
    shown = [resolve_dataset(word.text, datasets) for word in header]
    cells = row[1:]
    if (len(shown) != len(datasets) or None in shown or set(shown) != datasets
            or not aligned(cells, [word.x for word in header], 5)
            or any(not FRACTION.fullmatch(word.text) for word in cells)):
        return []
    caption = " ".join(words_text(row) for row in grid[:index - 1])
    columns = [Column(Header("WRR", "WRR", word.x, None, None), dataset, source_label=word.text,
                      reported_scale="fraction", name_evidence=(scope[0], scope[1][0]))
               for word, dataset in zip(header, shown, strict=True)]
    return [Table(page, label, caption, int(row[0].y), columns, [("WRR", [word.text for word in cells])],
                  config_name=scope[1][1], value_bboxes=normalized_boxes(page, cells),
                  top_k=int(retrieval[1][1]) if retrieval else None,
                  top_k_evidence=(retrieval[0], retrieval[1][0]) if retrieval else None)]


def capacity_grid(page: Page, label: str, grid: list[list[Word]], setup, dataset, summary, retrieval) -> list[Table]:
    if setup is None or setup[1][4].casefold() != label.casefold():
        return []
    starts = [i for i, row in enumerate(grid) if words_text(row) == "ConSum Sty"]
    if len(starts) != 1 or starts[0] + 2 != len(grid):
        return []
    index = starts[0]
    header, row = grid[index:]
    cells = row[1:]
    if (row[0].text != "WRR" or not aligned(cells, [word.x for word in header], 5)
            or any(not FRACTION.fullmatch(word.text) for word in cells)):
        return []
    caption = " ".join(words_text(row) for row in grid[:index])
    total, segment, method, _ = setup[1].groups()
    tables = []
    for heading, cell, cutoff in zip(header, cells, (summary, retrieval), strict=True):
        column = Column(Header(heading.text, "WRR", heading.x, None, None), dataset[1][1] if dataset else None,
                        source_label=heading.text, reported_scale="fraction", name_evidence=(setup[0], setup[1][0]))
        tables.append(Table(page, label, f"{caption} Watermark capacity {total} bits, {segment} bit per segment.",
                            int(row[0].y), [column], [("WRR", [cell.text])], config_name=method,
                            value_bboxes=normalized_boxes(page, [cell]), top_k=int(cutoff[1][1]) if cutoff else None,
                            top_k_evidence=(cutoff[0], cutoff[1][0]) if cutoff else None))
    return tables
