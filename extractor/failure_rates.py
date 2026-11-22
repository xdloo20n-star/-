"""Read native grids of leakage, stale citation and unsupported-answer rates."""

from collections import defaultdict
from dataclasses import replace
from itertools import pairwise
import re
from statistics import mean

from pdf_pages import Page, Word
from rag_tables import Column, Header, Table, column_boundaries


LABELS = ("Leak", "Stale", "Unsup.")
NAMES = ("Leakage rate", "Stale citation rate", "Unsupported answer rate")
FOOTER = re.compile(r"^Table ([A-Z]?\d+)\s*:", re.I)
PERCENT_RATES = re.compile(r"\bfailure\s+rates?\s*\(\s*%\s*\)", re.I)
RATE = re.compile(r"\d{1,3}(?:\.\d+)?")
DATASET = re.compile(r"[A-Z]{3,}(?:[-_][A-Z]{2,})*")
MODEL = re.compile(r"[A-Za-z][\w]*(?:[-.][\w]+)+")
LEGEND = re.compile(
    r"\bL\s*eak\s*,\s*S\s*tale\s*,\s*(?:and\s+)?U\s*nsup\s*\.\s+denote\s+"
    r"unauthorized\s+leakage\s*,\s*stale\s+citation\s*,\s*(?:and\s+)?unsupported[-\s]+answer\s+rates",
    re.I,
)


def failure_rate_definition(pages: list[Page]) -> tuple[Page, str] | None:
    for page in pages:
        if page.verification != "text_match":
            continue
        for source in (page.reading_text, page.text):
            if match := LEGEND.search(source):
                return page, match[0]
    return None


def visual_rows(words: tuple[Word, ...]) -> list[list[Word]]:
    rows = []
    for word in sorted(words, key=lambda item: item.y):
        if not rows or word.y - rows[-1][0].y > 2:
            rows.append([])
        rows[-1].append(word)
    return [sorted(row, key=lambda item: item.x) for row in rows]


def text(words: list[Word]) -> str:
    return " ".join(word.text for word in words)


def dataset_name(words: list[Word]) -> str | None:
    if not words or not all(re.fullmatch(r"[A-Z_-]+", word.text) for word in words):
        return None
    if any(right.x0 - left.x1 > 3 for left, right in pairwise(words)):
        return None
    label = "".join(word.text for word in words)
    return label if DATASET.fullmatch(label) else None


def read_failure_rates(page: Page, occupied: set[str], definition: tuple[Page, str] | None) -> list[Table]:
    if page.verification != "text_match" or not page.words:
        return []
    definition = failure_rate_definition([page]) or definition
    if definition is None:
        return []
    rows = visual_rows(page.words)
    tables = []
    for row in rows:
        y = mean(word.y for word in row)
        metrics = [word for word in row if word.text in LABELS]
        if len(metrics) < 3 or tuple(word.text for word in metrics) != LABELS * (len(metrics) // 3):
            continue
        boundaries = column_boundaries([word.x for word in metrics])
        slice_headers = [word for word in row if word.text in {"Method", "Slice"} and word.x < boundaries[0]]
        context = [word for word in page.words if y - 28 < word.y < y - 4 and word.x < boundaries[0]]
        if len(metrics) == 3 and [word.text for word in slice_headers] == ["Method", "Slice"]:
            start, separator = slice_headers
            groups = None
        elif len(metrics) >= 6 and not slice_headers:
            layout = {word.text: word for word in context if word.text in {"Setting", "Method"}}
            if layout.keys() != {"Setting", "Method"} or layout["Setting"].x >= layout["Method"].x:
                continue
            start, separator = layout["Setting"], layout["Method"]
            groups = model_groups(page.words, y, boundaries)
            if not groups:
                continue
        else:
            continue

        left, right = start.x0 - 5, boundaries[-1] + 6
        body, footer = [], []
        label = None
        for line in rows:
            words = [word for word in line if y + 2 < word.y < y + 335 and left < word.x < right]
            if not words:
                continue
            if label:
                if words[0].y - footer[-1][0].y > 15 or len(footer) >= 7:
                    break
                footer.append(words)
            elif words[0].y > y + 265:
                break
            elif match := FOOTER.match(text(words)):
                label = f"Table {match[1]}"
                footer.append(words)
            else:
                body.append(words)
        caption = " ".join(text(line) for line in footer)
        if not label or label.casefold() in occupied or not PERCENT_RATES.search(caption):
            continue
        if groups:
            parsed = read_model_rows(page, label, caption, int(y), body, separator.x0 - 4, boundaries, groups)
        else:
            parsed = read_slice_rows(page, label, caption, int(y), body, separator.x0 - 4, boundaries)
        if parsed:
            tables.extend(replace(table, columns=[replace(column, name_evidence=definition)
                                                 for column in table.columns]) for table in parsed)
            occupied.add(label.casefold())
    return tables


def model_groups(words: tuple[Word, ...], y: float, boundaries: list[float]) -> list[str] | None:
    labels = []
    for index in range(0, len(boundaries) - 1, 3):
        group = [word for word in words if y - 28 < word.y < y - 4
                 and boundaries[index] < word.x < boundaries[index + 3]]
        if len(group) == 2:
            first, last = sorted(group, key=lambda word: word.y)
            if first.text.endswith("-") and last.text.isdecimal() and abs(first.x - last.x) <= 5:
                label = first.text + last.text
            else:
                return None
        elif len(group) == 1:
            label = group[0].text
        else:
            return None
        if not MODEL.fullmatch(label) and label not in {"Avg.", "Average"}:
            return None
        labels.append(label)
    if len(set(labels)) != len(labels) or any(label in {"Avg.", "Average"} for label in labels[:-1]):
        return None
    return labels


def read_rate_cells(words: list[Word], boundaries: list[float]) -> list[str] | None:
    if len(words) != len(boundaries) - 1:
        return None
    values = []
    for word, (left, right) in zip(words, pairwise(boundaries), strict=True):
        if not left < word.x < right or not RATE.fullmatch(word.text) or not 0 <= float(word.text) <= 100:
            return None
        values.append(word.text)
    return values


def failure_columns(dataset: str, groups: list[str | None], aggregation: str | None = None) -> list[Column]:
    columns = []
    for group in groups:
        average = group in {"Avg.", "Average"}
        for label, name in zip(LABELS, NAMES, strict=True):
            columns.append(Column(
                Header(label, name, 0, None, "%"), dataset,
                ("reported average" if average else "per-model rate") if group else aggregation,
                None if average else group, direction="lower_is_better",
                source_label=f"{group} / {label}" if group else label,
                generator_locator=group if group and not average else None,
            ))
    return columns


def read_model_rows(page: Page, label: str, caption: str, index: int, body: list[list[Word]],
                    method_x: float, boundaries: list[float], groups: list[str]) -> list[Table]:
    blocks = []
    scores, anchors = [], []
    for line in body:
        y = mean(word.y for word in line)
        setting = [word for word in line if word.x < method_x]
        method = [word for word in line if method_x <= word.x < boundaries[0]]
        cells = [word for word in line if word.x >= boundaries[0]]
        if not setting and not method and cells and all(not re.search(r"\d", word.text) for word in cells):
            if scores:
                blocks.append((scores, anchors))
                scores, anchors = [], []
            continue
        if setting:
            name = dataset_name(setting)
            if not name:
                return []
            anchors.append((y, name))
        if method or cells:
            values = read_rate_cells(cells, boundaries)
            if not method or values is None:
                return []
            scores.append((y, text(method), values))
    if scores:
        blocks.append((scores, anchors))

    tables = []
    for scores, anchors in blocks:
        if not anchors:
            return []
        assigned = defaultdict(list)
        for y, method, values in scores:
            distances = sorted((abs(y - center), i) for i, (center, _) in enumerate(anchors))
            if len(distances) > 1 and distances[1][0] - distances[0][0] < 2:
                return []
            assigned[distances[0][1]].append((y, method, values))
        for i, (center, dataset) in enumerate(anchors):
            entries = assigned[i]
            if not entries or abs(center - (entries[0][0] + entries[-1][0]) / 2) > 2.5:
                return []
            tables.append(Table(page, label, caption, index, failure_columns(dataset, groups),
                                [(method, values) for _, method, values in entries], row_section=dataset,
                                family="failure_rates"))
    return tables


def read_slice_rows(page: Page, label: str, caption: str, index: int, body: list[list[Word]],
                    slice_x: float, boundaries: list[float]) -> list[Table]:
    reported = {
        "governance": "macro-average" if re.search(r"\bmacro-averaged\b", caption, re.I) else None,
        "retrieval": "single retrieval-pipeline run"
        if re.search(r"\bone end-to-end retrieval-pipeline\b", caption, re.I) else None,
    }
    if not any(reported.values()):
        return []
    tables = []
    dataset, aggregate = None, None
    for line in body:
        if all(word.text in {"↓", "↑"} for word in line):
            continue
        panel = re.match(r"^(Governance|Retrieval) on ", text(line), re.I)
        if panel:
            suffix = line[len(panel[0].split()):]
            dataset = dataset_name(suffix)
            aggregate = reported[panel[1].casefold()]
            if not dataset:
                return []
            continue
        method = text([word for word in line if word.x < slice_x])
        selection = text([word for word in line if slice_x <= word.x < boundaries[0]])
        values = read_rate_cells([word for word in line if word.x >= boundaries[0]], boundaries)
        if not dataset or not method or not re.fullmatch(r"[A-Za-z][\w.-]*", selection) or values is None:
            return []
        columns = failure_columns(dataset, [None], aggregate)
        tables.append(Table(page, label, caption, index, columns, [(method, values)],
                            row_section=f"{dataset} / {selection}", family="failure_rates",
                            data_split=None if selection.casefold() == "avg." else selection))
    return tables
