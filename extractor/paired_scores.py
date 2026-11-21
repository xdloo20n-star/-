"""Read native benchmark grids with explicitly ordered EM / F1 pairs."""

from decimal import Decimal
from itertools import pairwise
import re

from grouped_scores import DECIMAL
from pdf_pages import Page
from rag_tables import Column, Header, MODEL_HEADER, NATIVE_CAPTION, Table, column_boundaries


PAIR = re.compile(rf"({DECIMAL.pattern})\s*/\s*({DECIMAL.pattern})")
CELL = re.compile(r"\S+(?: \S+)*")
METRIC_ORDER = re.compile(r"\b(EM|F1)\s*/\s*(EM|F1)\b", re.I)
AVERAGE_IMPROVEMENT = re.compile(
    rf"\bimproves?\s+average\s+performance\s+by\s+(({DECIMAL.pattern})\s+(EM|F1)\s+and\s+"
    rf"({DECIMAL.pattern})\s+(EM|F1))\b", re.I,
)
PARAMETER = re.compile(r"[αβλ]")
PARAMETER_VALUE = re.compile(rf"{DECIMAL.pattern}(?: \([^()]+\)| [⋆*])?")
ROW_HEADER = re.compile(r"variants?", re.I)
DELTA_ROW = re.compile(r"[∆Δ]|(?:relative |absolute )?(?:change|difference|delta)\b", re.I)


def read_paired_scores(page: Page, lines: list[str], occupied: set[str]) -> list[Table]:
    if page.verification != "text_match":
        return []
    tables = []
    caption_at = -1
    caption_match = None
    for index, line in enumerate(lines):
        if match := NATIVE_CAPTION.match(line):
            caption_at, caption_match = index, match
            continue
        if caption_match is None or caption_match[1].casefold() in occupied or not 1 <= index - caption_at <= 7:
            continue
        header = list(CELL.finditer(line))
        if not 4 <= len(header) <= 13 or header[-1][0].casefold() not in {"average", "avg."}:
            continue
        method, *groups = header
        parameter = bool(PARAMETER.fullmatch(method[0]))
        if not (MODEL_HEADER.fullmatch(method[0]) or ROW_HEADER.fullmatch(method[0]) or parameter):
            continue
        if len({group[0].casefold() for group in groups}) != len(groups):
            continue
        if any(not re.search(r"[^\W\d_]", group[0]) for group in groups):
            continue
        centers = [(group.start() + group.end()) / 2 for group in groups]
        rows = read_paired_rows(lines, index + 1, method.start(), column_boundaries(centers), parameter)
        if len(rows) < 2:
            continue
        caption = "\n".join(lines[caption_at:index]).strip()
        order = score_order(caption, rows)
        if order is None:
            continue
        metrics, fragment = order
        columns = []
        for group, center in zip(groups, centers, strict=True):
            average = group is groups[-1]
            for metric in metrics:
                label = f"{group[0]} / {metric}" if average else metric
                columns.append(Column(Header(label, metric, center, None, None),
                                      None if average else group[0], "reported average" if average else None,
                                      name_evidence=(page, fragment)))
        tables.append(Table(page, caption_match[1], " ".join(caption.split()), index, columns, rows,
                            family="grouped_scores"))
        occupied.add(caption_match[1].casefold())
    return tables


def read_paired_rows(lines: list[str], index: int, method_at: int, boundaries: list[float],
                     parameter: bool) -> list[tuple[str, list[str]]]:
    rows = []
    for line in lines[index:index + 45]:
        if not line.strip() or NATIVE_CAPTION.match(line):
            break
        cells = list(CELL.finditer(line))
        if len(cells) != len(boundaries):
            return []
        method, *values = cells
        if abs(method.start() - method_at) > 2:
            return []
        if DELTA_ROW.fullmatch(method[0]):
            continue
        if parameter:
            if not PARAMETER_VALUE.fullmatch(method[0]):
                return []
        elif not re.search(r"[^\W\d_]", method[0]):
            return []
        scores = []
        for cell, (left, right) in zip(values, pairwise(boundaries), strict=True):
            center = (cell.start() + cell.end()) / 2
            pair = PAIR.fullmatch(cell[0])
            if pair is None or not left < center < right:
                return []
            scores.extend(pair.groups())
        rows.append((method[0], scores))
    return rows


def score_order(caption: str, rows: list[tuple[str, list[str]]]) -> tuple[tuple[str, str], str] | None:
    explicit = {tuple(metric.upper() for metric in match.groups()): match[0]
                for match in METRIC_ORDER.finditer(caption)}
    if explicit:
        if len(explicit) == 1 and all(len(set(metrics)) == 2 for metrics in explicit):
            return next(iter(explicit.items()))
        return None

    # A caption may name both metrics while giving the exact difference between
    # the only two displayed averages. Distinct changes pin the order of the pair.
    changes = list(AVERAGE_IMPROVEMENT.finditer(caption))
    if len(rows) != 2 or len(changes) != 1:
        return None
    fragment, first, first_name, second, second_name = changes[0].groups()
    metrics = first_name.upper(), second_name.upper()
    expected = Decimal(first.replace("−", "-")), Decimal(second.replace("−", "-"))
    left, right = ([Decimal(value.replace("−", "-")) for value in values[-2:]] for _, values in rows)
    difference = tuple(b - a for a, b in zip(left, right, strict=True))
    if metrics[0] != metrics[1] and expected[0] != expected[1] and expected == difference:
        return metrics, fragment
    return None
