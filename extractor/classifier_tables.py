"""Read captioned QA and query-routing diagnostics with separate column groups."""

import re
from itertools import pairwise
from statistics import median

from failure_rates import visual_rows
from grouped_scores import DECIMAL
from pdf_pages import Page, Word
from rag_tables import Column, Header, NATIVE_CAPTION, Table, column_boundaries, distinct_labels


ROUTE_CAPTION = re.compile(r"^Table \d+: The exact elapsed time per query and the percentage of the predicted labels\b", re.I)
ROUTE_HEADER = re.compile(r"^\s*Labels\s{2,}(Time/Query \((Sec\.)\))\s{2,}(Percentage \((%)\))\s*$", re.I)
ROUTE_ROW = re.compile(rf"^\s*(No \(A\)|One \(B\)|Multi \(C\))\s{{2,}}({DECIMAL.pattern})\s{{2,}}({DECIMAL.pattern})\s*$", re.I)
CLASSIFIER_CAPTION = re.compile(
    r"^Table \d+: Results (?:on QA and complexity classification with varying the data annotation strategies|"
    r"with varying model sizes for classifiers)\b", re.I,
)
CLASSIFIER_GROUPS = re.compile(r"\b(QA)\s{2,}(Classifier \(Accuracy\))", re.I)
CLASSIFIER_HEADER = re.compile(r"^\s*(Training Strategies|Sizes)\s{2,}(F1) +(Step) +(All) +(No) +(One) +(Multi)\b", re.I)
ROW_LABEL = re.compile(r"Adaptive-RAG \(Ours\)|w/o [A-Za-z][A-Za-z -]*|[A-Za-z]+ \(\d+M\)", re.I)
CLASSIFIER_ROW = re.compile(
    rf"^\s*({ROW_LABEL.pattern})\s+(?P<f1>{DECIMAL.pattern})\s+(?P<steps>\d+)\s+"
    rf"(?P<all>{DECIMAL.pattern})\s+(?P<no>{DECIMAL.pattern})\s+(?P<one>{DECIMAL.pattern})\s+"
    rf"(?P<multi>{DECIMAL.pattern})(?:\s{{3,}}\S.*|\s*)$", re.I,
)
ROUTER_ABLATION = re.compile(r"^Table \d+[:.] Ablation study of router[’']s input features\.", re.I)
ROUTER_FEATURES = ("query", "centroid", "source-id")
ROUTER_DATASETS = ("MIRAGE", "FeB4RAG")


def read_classifier_table(page: Page, label: str, start: int, lines: list[str]) -> list[Table]:
    if ROUTER_ABLATION.match(lines[0]):
        return read_router_ablation(page, label, start, " ".join(" ".join(lines[:2]).split()))
    for index in range(2, min(8, len(lines))):
        caption = " ".join(" ".join(lines[:index - 1]).split())
        if (groups := CLASSIFIER_GROUPS.search(lines[index - 1])) and CLASSIFIER_CAPTION.match(caption):
            if header := CLASSIFIER_HEADER.match(lines[index]):
                return read_classifier_matrix(page, label, start + index, caption, groups, header, lines[index + 1:])
        caption = " ".join(" ".join(lines[:index]).split())
        if ROUTE_CAPTION.match(caption) and (header := ROUTE_HEADER.fullmatch(lines[index])):
            return read_routes(page, label, start + index, caption, header, lines[index + 1:index + 9])
    return []


def read_router_ablation(page: Page, label: str, start: int, caption: str) -> list[Table]:
    if not page.words or not page.size:
        return []
    rows = visual_rows(page.words)
    anchors = [row[0] for row in rows if ROUTER_ABLATION.match(" ".join(word.text for word in row))]
    if len(anchors) != 1:
        return []
    grid = [row for row in rows if 15 < row[0].y - anchors[0].y < 100]
    leading = [("Recall",), ROUTER_FEATURES, ROUTER_DATASETS]
    for index in range(len(grid) - 5):
        if [tuple(word.text for word in row) for row in grid[index:index + 3]] == leading:
            break
    else:
        return []
    recall, features, datasets, *body = grid[index:index + 6]
    if not datasets[0].x < recall[0].x < datasets[1].x:
        return []
    centers = [word.x for word in features + datasets]
    bounds = list(pairwise(column_boundaries(centers)))
    tables = []
    for row, expected in zip(body, (("✓", "✓", "✗"), ("✓", "✗", "✓"), ("✓", "✓", "✓")), strict=True):
        if len(row) != 5 or tuple(word.text for word in row[:3]) != expected:
            return []
        if any(not left < word.x < right for word, (left, right) in zip(row, bounds, strict=True)):
            return []
        values = row[3:]
        if any(not DECIMAL.fullmatch(value.text) or not 0 <= float(value.text.replace("−", "-")) <= 100
               for value in values):
            return []
        columns = [Column(Header("Recall", "Recall", word.x, None, None), word.text,
                          source_label=f"{word.text} / Recall", direction="higher_is_better") for word in datasets]
        used = ", ".join(feature for feature, mark in zip(ROUTER_FEATURES, expected, strict=True) if mark == "✓")
        tables.append(Table(page, label, caption, start, columns, [(" ".join(expected), [word.text for word in values])],
                            config_name="RAGRoute", config_note=f"Router input features: {used}",
                            task_type="classification of data source relevance",
                            value_bboxes=(tuple(normalized_box(word, page.size) for word in values),)))
    return tables


def read_classifier_tables(pages: list[Page], occupied: set[tuple[int, str]]) -> list[Table]:
    tables = []
    for page in pages:
        if page.verification != "text_match":
            continue
        lines = page.text.splitlines()
        for index, line in enumerate(lines):
            captions = list(NATIVE_CAPTION.finditer(line))
            for position, caption in enumerate(captions):
                label = caption[1]
                if (page.number, label.casefold()) in occupied:
                    continue
                left = caption.start(1)
                right = captions[position + 1].start(1) if position + 1 < len(captions) else None
                tables.extend(read_classifier_table(page, label, index, [line[left:right] for line in lines[index:index + 18]]))
    return tables


def column_centers(header: re.Match[str], groups: range | tuple[int, ...]) -> list[float]:
    return [(header.start(group) + header.end(group)) / 2 for group in groups]


def read_classifier_matrix(page: Page, label: str, index: int, caption: str, groups: re.Match[str],
                           header: re.Match[str], lines: list[str]) -> list[Table]:
    centers = column_centers(header, range(2, 8))
    if not centers[0] <= (groups.start(1) + groups.end(1)) / 2 <= centers[1] or not (
        centers[2] <= (groups.start(2) + groups.end(2)) / 2 <= centers[-1]
    ):
        return []
    physical = positioned_classifier_rows(page, label, header[1]) if page.words and page.size else None
    if physical is None:
        if page.words and page.size:
            return []
        rows, bboxes = aligned_classifier_rows(lines, header, centers), ()
    else:
        rows, bboxes = physical
    if len(rows) < 2 or not distinct_labels([row for row, _ in rows]):
        return []
    metrics = ("F1", "Reported step", *("Classifier accuracy",) * 4)
    variants = (None, None, "All", "A (No)", "B (One)", "C (Multi)")
    columns = []
    for i, (printed, name, center, variant) in enumerate(zip(header.groups()[1:], metrics, centers, variants, strict=True)):
        group = groups[1 if i < 2 else 2]
        source = f"{group} / {printed}"
        columns.append(Column(Header(printed, name, center, None, None), None, metric_variant=variant,
                              source_label=source, name_evidence=(page, printed if i < 2 else group)))
    return [Table(page, label, caption, index, columns, rows, value_bboxes=bboxes,
                  task_type="question answering and query complexity classification")]


def aligned_classifier_rows(lines: list[str], header: re.Match[str], centers: list[float]) -> list[tuple[str, list[str]]]:
    bounds = list(pairwise(column_boundaries(centers)))
    rows = []
    for line in lines[:8]:
        if not line.strip():
            continue
        if not (row := CLASSIFIER_ROW.fullmatch(line)):
            if ROW_LABEL.match(line.lstrip()):
                return []
            break
        if abs(row.start(1) - header.start(1)) > 3:
            return []
        positions = column_centers(row, range(2, 8))
        # Poppler can shift the entire compact row when the method label is long.
        shift = median(position - center for position, center in zip(positions, centers, strict=True))
        if abs(shift) > min(right - left for left, right in bounds) or any(
            not left < position - shift <= right for position, (left, right) in zip(positions, bounds, strict=True)
        ):
            return []
        rows.append((row[1].strip(), list(row.groups()[1:])))
    return rows


def positioned_classifier_rows(page: Page, label: str, row_header: str):
    visual = visual_rows(page.words)
    expected = (row_header + " F1 Step All No One Multi").split()
    captions = [words[i] for words in visual for i in range(len(words) - 1)
                if f"{words[i].text} {words[i + 1].text}".casefold() == f"{label}:".casefold()]
    headers = [words[i:i + len(expected)] for words in visual for i in range(len(words) - len(expected) + 1)
               if [word.text for word in words[i:i + len(expected)]] == expected
               and any(0 < words[i].y - caption.y < 45 and abs(words[i].x0 - caption.x0) < 25
                       for caption in captions)]
    if len(headers) != 1:
        return None
    start, *_, end = headers[0]
    boundaries = column_boundaries([word.x for word in headers[0][-6:]])
    left = start.x0 - 3
    right = boundaries[-1]
    rows = []
    bboxes = []
    for words in visual:
        body = [word for word in words if 2 < word.y - end.y < 65 and left <= word.x < right]
        if not body:
            continue
        name = " ".join(word.text for word in body if word.x < boundaries[0])
        if not ROW_LABEL.fullmatch(name):
            break
        values = [word for word in body if boundaries[0] <= word.x]
        if len(values) != 6 or not values[1].text.isdecimal() or any(
            not left <= word.x < right or not DECIMAL.fullmatch(word.text)
            for word, (left, right) in zip(values, pairwise(boundaries), strict=True)
        ):
            return None
        rows.append((name, [word.text for word in values]))
        bboxes.append(tuple(normalized_box(word, page.size) for word in values))
    return rows, tuple(bboxes)


def normalized_box(word: Word, size: tuple[float, float]) -> tuple[float, float, float, float]:
    width, height = size
    return word.x0 / width, word.y0 / height, word.x1 / width, word.y1 / height


def read_routes(page: Page, label: str, index: int, caption: str, header: re.Match[str], lines: list[str]) -> list[Table]:
    centers = column_centers(header, (1, 3))
    bounds = list(pairwise(column_boundaries(centers)))
    rows = []
    for line in lines:
        if not line.strip():
            continue
        if not (row := ROUTE_ROW.fullmatch(line)):
            break
        positions = column_centers(row, (2, 3))
        if any(not left < center < right for center, (left, right) in zip(positions, bounds, strict=True)):
            return []
        rows.append((row[1], [row[2], row[3]]))
    if {row.casefold() for row, _ in rows} != {"no (a)", "one (b)", "multi (c)"} or len(rows) != 3:
        return []
    columns = [Column(Header(header[1], "Latency", centers[0], None, header[2]), None,
                      aggregation="mean per query conditional on predicted route", source_label=header[1]),
               Column(Header(header[3], "Predicted route proportion", centers[1], None, header[4]), None)]
    return [Table(page, label, caption, index, columns, rows, task_type="query complexity classification")]
