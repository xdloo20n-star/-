"""Read explicit column and row labels from evaluation tables."""

from dataclasses import dataclass
from itertools import pairwise
import re
from statistics import mean

from pdf_pages import Page


MISSING = {"-", "–", "—", "N/A", "n/a", "NA"}
NUMBER = re.compile(r"[+−-]?(?:\d+(?:[.,]\d+)?|\.\d+)(?:[eE][+-]?\d+)?%?")
CAPTION = re.compile(r"^\s*(table\s+(?:\d+(?:\.\d+)*|[ivxlcdm]+))(?=[:.\s-]|$)", re.I)
SCORE = re.compile(
    r"(?P<name>(?:answer\s+)?(?:exact\s+match|precision|recall|relevance|correctness|similarity|accuracy)"
    r"|EM|F1|MRR|MAP|(?:n?DCG|Hit(?:s|\s+rate)?)|ROUGE(?:[-_ ]?[1-9L])?"
    r"|BLEU(?:[-_ ]?[1-9])?|BERTScore|faithfulness|groundedness|AUC|latency|response\s+time)"
    r"(?:\s*@\s*(?P<cutoff>\d+))?"
    r"(?:\s*\(\s*(?P<unit>%|fraction|ms|s|sec|seconds?|milliseconds?)\s*\)|\s*(?P<percent>%))?",
    re.I,
)
GROUP = re.compile(
    r"Natural\s+Questions|Huatuo(?:-26M)?|Hotpot\s*QA|2Wiki(?:MultiHopQA)?|Trivia\s*QA|Pop\s*QA|"
    r"WebQuestions|MuSiQue|SQuAD(?:\s+v?[12](?:\.\d+)?)?|PubMed\s*QA|"
    r"MS\s*MARCO|Multi[- ]Hop\s+Cybersecurity\s+Questions|"
    r"General\s+Cybersecurity\s+Ques(?:\s*t(?:ions)?)?|\bDS\s*\d+\b|\bDataset\s+[A-Z0-9][\w.-]*",
    re.I,
)
GENERATOR = re.compile(
    r"LLaMA[- ]?\d(?:\.\d+)?(?:[- ]\d+B)?(?:[- ]Instruct)?|"
    r"Qwen[- ]?\d(?:\.\d+)?(?:[- ](?:plus|\d+B|instruct))*|"
    r"GPT[- ]?4o?(?:[- ]mini)?|Claude[- ]?\d(?:\.\d+)?(?:[- ](?:sonnet|opus|haiku))?",
    re.I,
)
MODEL_HEADER = re.compile(r"^(?:methods?|models?|systems?|configurations?|approach(?:es)?)\b", re.I)
AGGREGATION_DESCRIPTION = re.compile(r"\bmean\b.{0,100}\bstandard\s+deviation\b", re.I | re.S)
DATASET_HEADER = re.compile(r"^\s*(?:benchmarks?|datasets?)\b", re.I)
TRANSPOSED_METRIC = re.compile(r"\b(factuality|accuracy|faithfulness)\s+scores?\b", re.I)
HEADER_CELLS = re.compile(r"\S(?:.*?\S)?(?=\s{2,}|$)")


@dataclass(frozen=True)
class Header:
    label: str
    name: str
    center: float
    cutoff: int | None
    unit: str | None


@dataclass(frozen=True)
class Column:
    header: Header
    dataset: str | None
    aggregation: str | None = None
    generator: str | None = None
    original_row: str | None = None


@dataclass(frozen=True)
class Table:
    page: Page
    label: str
    caption: str
    index: int
    columns: list[Column]
    rows: list[tuple[str, list[str]]]


def headers(line: str) -> list[Header]:
    line = line.rstrip()
    found = []
    for match in SCORE.finditer(line):
        # Keep full words and labels such as ROUGE_L together.
        if (match.start() and re.match(r"[\w-]", line[match.start() - 1])) or (
            match.end() < len(line) and re.match(r"[\w@.-]", line[match.end()])
        ):
            continue
        cutoff = match["cutoff"]
        name = match["name"].strip()
        if cutoff is not None:
            name += f"@{cutoff}"
        found.append(Header(
            match[0].strip(), name, (match.start() + match.end()) / 2,
            int(cutoff) if cutoff is not None else None, match["unit"] or match["percent"],
        ))
    if not found:
        return []
    if found[0].center < 8 and not MODEL_HEADER.match(line.lstrip()):
        return []
    # The region between recognized labels may contain separator rules from OCR.
    offset = max(0, int(found[0].center - len(found[0].label) / 2))
    tail = SCORE.sub("", line[offset:])
    if re.search(r"[A-Za-z]{2,}|\d{2,}", tail):
        return []
    return found


def read_row(line: str, first_column: float, expected: int) -> tuple[str, list[str]] | None:
    tokens = list(re.finditer(r"[^\s|]+", line))
    if len(tokens) <= expected:
        return None
    values = tokens[-expected:]
    if any(not NUMBER.fullmatch(token[0]) and token[0] not in MISSING for token in values):
        return None
    start = values[0].start()
    if abs(start - first_column) > max(10, first_column * 0.25):
        return None
    label = line[:start].strip()
    if not re.search(r"[^\W\d_]", label) or len(label) > 100 or label.endswith(":"):
        return None
    return label, [token[0] for token in values]


def groups_for(line: str, first_column: float, pattern: re.Pattern[str]) -> list[tuple[str, float]]:
    found = []
    for match in pattern.finditer(line):
        center = (match.start() + match.end()) / 2
        if center < first_column - 5:
            continue
        label = " ".join(match[0].split())
        if label.lower().startswith("general cybersecurity ques"):
            label = "General Cybersecurity Questions"
        found.append((label, center))
    return found


def merged_headers(lines: list[str], first: float, pattern: re.Pattern[str]) -> list[tuple[str, float]]:
    candidates = [groups_for(line, first, pattern) for line in lines if not CAPTION.match(line)]
    widest = max(candidates, key=len, default=[])
    if len(widest) >= 2:
        return widest
    distinct = {}
    for named in reversed(candidates):
        for label, center in named:
            distinct.setdefault(label.casefold(), (label, center))
    return sorted(distinct.values(), key=lambda item: item[1])


def assign_groups(columns: list[Header], groups: list[tuple[str, float]]) -> list[str | None]:
    if not groups:
        return [None] * len(columns)
    if len(groups) == 1:
        return [groups[0][0]] * len(columns)
    if len(groups) > len(columns):
        return [None] * len(columns)
    # Fit ordered spans to the visual centers of their merged headers.
    options = {0: (0.0, [])}
    for label, center in groups:
        next_options = {}
        for start, (cost, assigned) in options.items():
            for end in range(start + 1, len(columns) + 1):
                fitted = mean(column.center for column in columns[start:end])
                value = cost + (center - fitted) ** 2
                if end not in next_options or value < next_options[end][0]:
                    next_options[end] = value, assigned + [label] * (end - start)
        options = next_options
    return options[len(columns)][1]


def read_tables(page: Page) -> list[Table]:
    lines = page.text.splitlines()
    captions = [(i, match) for i, line in enumerate(lines) if (match := CAPTION.match(line))]
    tables = []
    used_until = -1
    for index, line in enumerate(lines):
        if index <= used_until:
            continue
        found = headers(line)
        if not found or len(found) > 24:
            continue
        previous = lines[max(0, index - 4):index]
        if len(found) == 1 and not any(MODEL_HEADER.match(value.lstrip()) for value in [*previous, line]):
            continue
        nearby = [item for item in captions if -12 <= item[0] - index <= 60]
        if not nearby:
            continue
        first = found[0].center
        groups = merged_headers(previous, first, GROUP)
        generator_headers = merged_headers(previous, first, GENERATOR)
        # A generator heading is considered a column dimension only if the dataset
        # headers repeat under separate model headings.
        repeated_datasets = len({name.casefold() for name, _ in groups}) < len(groups)
        generators = assign_groups(found, generator_headers if repeated_datasets else [])
        names = assign_groups(found, groups)
        paired = bool(AGGREGATION_DESCRIPTION.search(page.prose) or AGGREGATION_DESCRIPTION.search(page.text))
        widths = [len(found), len(found) * 2] if paired else [len(found)]
        for width in widths:
            rows, row_end = scan_rows(lines, index + 1, first, width)
            if not rows:
                continue
            caption_at, match = min(nearby, key=lambda item: min(abs(item[0] - index), abs(item[0] - row_end)))
            if index < caption_at <= row_end:
                continue
            label = match[1]
            caption = " ".join(lines[caption_at].split())
            if not groups:
                caption_groups = list(GROUP.finditer(caption))
                if len(caption_groups) == 1:
                    names = [caption_groups[0][0]] * len(found)
            aggregations = ("mean", "std") if width == 2 * len(found) else (None,)
            columns = [Column(header, name, aggregate, generator)
                       for header, name, generator in zip(found, names, generators, strict=True)
                       for aggregate in aggregations]
            tables.append(Table(page, label, caption, index, columns, rows))
            used_until = row_end
            break
    occupied = {table.label.casefold() for table in tables}
    tables.extend(read_transposed_tables(page, lines, captions, occupied))
    return sorted(tables, key=lambda table: table.index)


def read_transposed_tables(page: Page, lines: list[str], captions: list[tuple[int, re.Match[str]]],
                           occupied: set[str]) -> list[Table]:
    tables = []
    for caption_index, caption_match in captions:
        label = caption_match[1]
        caption = " ".join(lines[caption_index].split())
        metric = TRANSPOSED_METRIC.search(caption)
        if not metric or label.casefold() in occupied:
            continue
        start = max(0, caption_index - 38)
        for index in range(caption_index - 1, start - 1, -1):
            heading = lines[index]
            if not DATASET_HEADER.match(heading):
                continue
            models = list(HEADER_CELLS.finditer(heading.rstrip()))[1:]
            if not 2 <= len(models) <= 24:
                break
            entries, _ = scan_rows(lines[:caption_index], index + 1, models[0].start(), len(models))
            if len(entries) < 2:
                break
            columns = []
            for row, _ in entries:
                is_average = re.fullmatch(r"(?:avg\.?|average|mean)", row, re.I)
                dataset = None if is_average else row.split("(", 1)[0].strip()
                header = Header(metric[1], metric[1].capitalize(), 0, None, None)
                columns.append(Column(header, dataset or None, "mean" if is_average else None, original_row=row))
            rows = [
                (model[0], [values[model_index] for _, values in entries])
                for model_index, model in enumerate(models)
            ]
            tables.append(Table(page, label, caption, index, columns, rows))
            occupied.add(label.casefold())
            break
    return tables


def scan_rows(lines: list[str], start: int, first_column: float, width: int) -> tuple[list[tuple[str, list[str]]], int]:
    rows = []
    last = start
    for index in range(start, min(len(lines), start + 54)):
        if CAPTION.match(lines[index]):
            break
        row = read_row(lines[index], first_column, width)
        if row is None:
            if (rows and index - last > 1) or (not rows and index - start >= 5):
                break
            continue
        rows.append(row)
        last = index
    if width > 1 and rows and any(a == b for (a, _), (b, _) in pairwise(rows)):
        # A repeated identical method label on consecutive lines indicates a split row.
        return [], last
    return rows, last
