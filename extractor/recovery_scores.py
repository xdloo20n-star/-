"""Read score tables for recovered and initially successful retrieval subsets."""

from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_HALF_UP
from itertools import pairwise
import re

from grouped_scores import DECIMAL
from paired_scores import CELL, DELTA_ROW, caption_regions, cell_center
from pdf_pages import Page
from rag_tables import DATASET_HEADER, MODEL_HEADER, NATIVE_CAPTION, Column, Header, Table, column_boundaries


RECOVERED = re.compile(r"(?:Final EM|Token-level F1) on recovered trajectories\b", re.I)
INITIAL = re.compile(r"Final EM when the initial retrieval (?:already\s*)?succeeds\b", re.I)
DEPTH = re.compile(r"Final EM on ([\w-]+) by recovery depth\b", re.I)
COUNT = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
COUNT_PAIR = re.compile(rf"({COUNT})\s*/\s*({COUNT})")
COUNT_ORDER = re.compile(r"(?:n\s+(?:is given|is reported)\s+as|subset sizes\s*\(n,)\s*"
                         r"([^/(),.]+?)\s*/\s*([^(),.]+?)\s*(?:[.),]|$)", re.I)
SUBSET_REFERENCE = re.compile(r"\b(Table \d+) reports token-level F1 on the same recovered subsets as (Table \d+)\b", re.I)
DEPTH_LABEL = re.compile(r"k\s*(=|≥|>=)\s*(\d+)", re.I)


@dataclass(frozen=True)
class RecoveryGrid:
    page: Page
    label: str
    caption: str
    index: int
    metric: str
    condition: str
    condition_fragment: str
    methods: tuple[str, str]
    rows: list[tuple[str, list[str]]]
    denominators: list[tuple[int, int, Page, str] | None]
    delta: bool


def read_recovery_scores(pages: list[Page], occupied: set[tuple[int, str]]) -> list[Table]:
    grids = []
    tables = []
    for page, start, label, lines in caption_regions(pages, occupied):
        for i in range(1, min(9, len(lines))):
            if NATIVE_CAPTION.match(lines[i]):
                break
            headings = list(CELL.finditer(lines[i]))
            if not headings or not (DATASET_HEADER.fullmatch(headings[0][0]) or MODEL_HEADER.fullmatch(headings[0][0])):
                continue
            prefix = caption_body(lines[:i])
            if match := DEPTH.search(prefix):
                tables.extend(read_depth_grid(page, start + i, label, prefix, match, headings, lines[i + 1:]))
            elif match := RECOVERED.search(prefix) or INITIAL.search(prefix):
                grid = read_subset_grid(page, start + i, label, prefix, match, headings, lines[i + 1:])
                if grid:
                    grids.append(grid)
            break
    for grid in grids:
        if grid.metric == "F1" and not any(grid.denominators):
            grid = attach_matching_subsets(grid, grids)
        tables.extend(subset_tables(grid))
    return tables


def caption_body(lines: list[str]) -> str:
    # Ignore parallel prose and keep wrapped caption lines in the same column.
    start = len(lines[0]) - len(lines[0].lstrip())
    fragments = []
    for line in lines:
        if line and abs(len(line) - len(line.lstrip()) - start) <= 6:
            fragments.append(re.split(r" {3,}", line.strip(), maxsplit=1)[0])
    return " ".join(fragments)


def read_subset_grid(page: Page, index: int, label: str, caption: str, condition: re.Match[str],
                     headings: list[re.Match[str]], body: list[str]) -> RecoveryGrid | None:
    has_counts = len(headings) > 1 and headings[1][0] == "n"
    width = 4 if has_counts or (len(headings) >= 4 and DELTA_ROW.fullmatch(headings[3][0])) else 3
    headings = headings[:width]
    if len(headings) != width or not DATASET_HEADER.fullmatch(headings[0][0]):
        return None
    methods = tuple(heading[0] for heading in headings[1 + has_counts:3 + has_counts])
    if len({method.casefold() for method in methods}) != 2 or any(not re.search(r"[^\W\d_]", m) for m in methods):
        return None
    patterns = [COUNT_PAIR, DECIMAL, DECIMAL] if has_counts else [DECIMAL] * (width - 1)
    rows = aligned_rows(headings, body, patterns)
    if len(rows) < 2 or len({row.casefold() for row, _ in rows}) != len(rows):
        return None
    initial = bool(INITIAL.fullmatch(condition[0]))
    counts = [values[0] for _, values in rows] if has_counts else [match[0] for match in COUNT_PAIR.finditer(caption)]
    denominators = [None] * len(rows)
    if len(counts) == len(rows) and (order := count_order(caption, methods)) is not None:
        denominators = [denominator(page, raw, order) for raw in counts]
    if has_counts:
        rows = [(dataset, values[1:]) for dataset, values in rows]
    metric = "F1" if condition[0].startswith("Token-level") else "EM"
    variant = "initial retrieval succeeds" if initial else "recovered after initial failure"
    return RecoveryGrid(page, label, caption, index, metric, variant, condition[0], methods, rows, denominators,
                        width == 4 and not has_counts)


def aligned_rows(headings: list[re.Match[str]], lines: list[str], patterns: list[re.Pattern[str]]) -> list[tuple[str, list[str]]]:
    centers = [cell_center(heading) for heading in headings[1:]]
    bounds = list(pairwise(column_boundaries(centers)))
    rows = []
    gap = 0
    for line in lines:
        if NATIVE_CAPTION.match(line):
            break
        cells = list(CELL.finditer(line))
        if not cells or abs(cells[0].start() - headings[0].start()) > 2:
            gap += 1
            if gap > 2:
                break
            continue
        row, *values = cells[:len(headings)]
        if len(values) != len(patterns):
            break
        if any(not pattern.fullmatch(value[0]) or not left < cell_center(value) < right
               for pattern, value, (left, right) in zip(patterns, values, bounds, strict=True)):
            break
        if len(cells) > len(headings) and cells[len(headings)].start() < bounds[-1][1]:
            return []
        rows.append((row[0], [value[0] for value in values]))
        gap = 0
    return rows


def count_order(text: str, methods: tuple[str, str]) -> tuple[int, int] | None:
    method_keys = tuple(compact(method) for method in methods)
    if method_keys[0] == method_keys[1]:
        return None
    orders = set()
    for match in COUNT_ORDER.finditer(text):
        pair = [compact(group) for group in match.groups()]
        if pair[0] == "ours" and pair[1] in method_keys:
            pair[0] = next(key for key in method_keys if key != pair[1])
        if set(pair) == set(method_keys):
            orders.add(tuple(pair.index(key) for key in method_keys))
    return next(iter(orders)) if len(orders) == 1 else None


def compact(value: str) -> str:
    # Poppler can omit hyphens at a line break in the reading-order stream.
    return re.sub(r"[-\s]+", "", value.casefold())


def denominator(page: Page, raw: str, order: tuple[int, int]) -> tuple[int, int, Page, str] | None:
    if not (match := COUNT_PAIR.fullmatch(raw)):
        return None
    counts = [int(group.replace(",", "")) for group in match.groups()]
    return counts[order[0]], counts[order[1]], page, raw


def attach_matching_subsets(grid: RecoveryGrid, grids: list[RecoveryGrid]) -> RecoveryGrid:
    source = grid.page.reading_text or grid.page.text
    refs = {match[2].casefold() for match in SUBSET_REFERENCE.finditer(source)
            if match[1].casefold() == grid.label.casefold()}
    candidates = [candidate for candidate in grids if candidate.label.casefold() in refs and candidate.metric == "EM"
                  and candidate.methods == grid.methods and candidate.condition == grid.condition]
    if len(candidates) != 1:
        return grid
    reference = candidates[0]
    counts = {row: size for (row, _), size in zip(reference.rows, reference.denominators, strict=True)}
    return replace(grid, denominators=[counts.get(row) for row, _ in grid.rows])


def compatible_size(raw: str, count: int) -> bool:
    if count <= 0:
        return False
    pct = Decimal(raw.replace("−", "-"))
    successes = (pct * count / 100).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    half_step = Decimal("0.5").scaleb(pct.as_tuple().exponent)
    return 0 <= successes <= count and abs(pct - 100 * successes / count) <= half_step


def subset_tables(grid: RecoveryGrid) -> list[Table]:
    tables = []
    for operator, method in enumerate(grid.methods):
        columns = [
            recovery_column(dataset, method, grid.metric, grid.condition, grid.page, grid.condition_fragment,
                            raw=values[operator], size=counts, operator=operator)
            for (dataset, values), counts in zip(grid.rows, grid.denominators, strict=True)
        ]
        tables.append(Table(grid.page, grid.label, grid.caption, grid.index, columns,
                            [(method, [values[operator] for _, values in grid.rows])], family="retrieval_recovery"))
    if grid.delta:
        metric = f"Delta {grid.metric}"
        delta_columns = [recovery_column(dataset, "∆", metric, grid.condition, grid.page, grid.condition_fragment)
                         for dataset, _ in grid.rows]
        tables.append(Table(grid.page, grid.label, grid.caption, grid.index, delta_columns,
                            [(grid.methods[1], [values[-1] for _, values in grid.rows])], family="retrieval_recovery"))
    return tables


def recovery_column(dataset: str, method: str, metric: str, condition: str, page: Page, fragment: str,
                    *, raw: str | None = None, size: tuple[int, int, Page, str] | None = None,
                    operator: int = 0, depth: str | None = None) -> Column:
    sample_size, size_evidence = None, None
    if size is not None and raw is not None and (metric != "EM" or compatible_size(raw, size[operator])):
        sample_size, size_evidence = size[operator], (size[2], size[3])
    return Column(Header(metric, metric, 0, None, None), dataset, original_row=dataset if depth is None else method,
                  name_evidence=(page, fragment), metric_variant=condition, metric_variant_evidence=(page, fragment),
                  sample_size=sample_size, sample_size_evidence=size_evidence,
                  source_label=depth or (method if method == "∆" else f"{method} / {metric}"))


def read_depth_grid(page: Page, index: int, label: str, caption: str, match: re.Match[str],
                    headings: list[re.Match[str]], lines: list[str]) -> list[Table]:
    if len(headings) < 3 or not MODEL_HEADER.fullmatch(headings[0][0]):
        return []
    depths = [DEPTH_LABEL.fullmatch(heading[0]) for heading in headings[1:]]
    if not all(depths):
        return []
    counts = aligned_rows(headings, lines[:1], [COUNT_PAIR] * len(depths))
    if len(counts) != 1 or counts[0][0] != "n":
        return []
    rows = aligned_rows(headings, lines[1:], [DECIMAL] * len(depths))
    if len(rows) != 3 or not DELTA_ROW.fullmatch(rows[-1][0]):
        return []
    methods = rows[0][0], rows[1][0]
    order = count_order(page.reading_text or page.text, methods)
    sizes = [denominator(page, value, order) if order is not None else None for value in counts[0][1]]
    tables = []
    for operator, (method, values) in enumerate(rows):
        metric = "EM" if operator < 2 else "Delta EM"
        columns = [
            recovery_column(match[1], method, metric, f"recovery depth {'>=' if depth[1] != '=' else ''}{depth[2]}",
                            page, depth[0], raw=raw, size=size if operator < 2 else None, operator=operator, depth=depth[0])
            for depth, raw, size in zip(depths, values, sizes, strict=True)
        ]
        tables.append(Table(page, label, caption, index, columns, [(method if operator < 2 else methods[1], values)],
                            family="retrieval_recovery"))
    return tables
