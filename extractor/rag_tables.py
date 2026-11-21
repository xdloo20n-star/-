"""Read explicit column and row labels from evaluation tables."""

from dataclasses import dataclass
from itertools import pairwise
import re
from statistics import mean, median

from pdf_pages import Page


MISSING = {"-", "–", "—", "N/A", "n/a", "NA"}
NUMBER = re.compile(r"[+−-]?(?:\d+(?:[.,]\d+)?|\.\d+)(?:[eE][+-]?\d+)?%?")
CAPTION = re.compile(r"^\s*(table\s+(?:\d+(?:\.\d+)*|[ivxlcdm]+))(?=[:.\s-]|$)", re.I)
SCORE = re.compile(
    r"(?P<name>(?:answer\s+)?(?:classification\s+accuracy|exact\s+match|precision|recall|relevance|correctness|similarity|accuracy)"
    r"|EM|F1(?:[- ]Score)?|MRR|MAP|(?:n?DCG|Hit(?:s|\s+rate)?)|ROUGE(?:[-_ ]?[1-9L])?"
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
NATIVE_CAPTION = re.compile(r"(?:^\s*|(?<= {2}))(table\s+(?:\d+(?:\.\d+)*|[ivxlcdm]+))\s*[:.](?=\s|$)", re.I)
DELTA_HEADER = re.compile(r"Δ|∆|(?:relative\s+|absolute\s+|%\s*)?(?:change|difference)", re.I)
GROUPED_NUMBER = re.compile(r"[+−-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?")
METRIC_UNIT = re.compile(r"\s*\((%|fraction|ms|s|sec|seconds?|milliseconds?)\)$", re.I)
TOKEN = re.compile(r"\S+")


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
    name_evidence: tuple[Page, str] | None = None
    aggregation_evidence: tuple[Page, str] | None = None
    generator_locator: str | None = None
    generator_evidence: tuple[Page, str] | None = None
    direction: str | None = None
    source_label: str | None = None


@dataclass(frozen=True)
class Table:
    page: Page
    label: str
    caption: str
    index: int
    columns: list[Column]
    rows: list[tuple[str, list[str]]]
    row_section: str | None = None
    family: str | None = None
    config_name: str | None = None
    data_split: str | None = None


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
        if name.casefold() == "f1-score":
            name = "F1"
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
    tail = re.sub(r"\bbenchmarks?\b", " ", tail, flags=re.I)
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


def read_tables(page: Page, failure_definition: tuple[Page, str] | None = None) -> list[Table]:
    from failure_rates import read_failure_rates
    from failure_statistics import read_failure_statistics
    from grouped_scores import read_grouped_scores
    from paired_scores import read_paired_scores
    from qa_comparison import read_qa_comparisons

    lines = page.text.splitlines()
    captions = [(i, match) for i, line in enumerate(lines) if (match := CAPTION.match(line))]
    tables, grouped_headers = read_grouped_scores(page, lines)
    used_until = -1
    for index, line in enumerate(lines):
        if index <= used_until or index in grouped_headers:
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
            aggregate_label = ("reported average"
                               if re.search(r"\bmetrics?\s*\(\s*averages?\s*\)", caption, re.I)
                               else None)
            if re.search(r"\bclassification\s+metrics\b", caption, re.I):
                found = [Header(
                    header.label,
                    "Classification accuracy" if header.name.casefold() == "accuracy" else header.name,
                    header.center, header.cutoff, None,
                ) for header in found]
            if not groups:
                caption_groups = list(GROUP.finditer(caption))
                if len(caption_groups) == 1:
                    names = [caption_groups[0][0]] * len(found)
            aggregations = ("mean", "std") if width == 2 * len(found) else (aggregate_label,)
            columns = [Column(header, name, aggregate, generator)
                       for header, name, generator in zip(found, names, generators, strict=True)
                       for aggregate in aggregations]
            tables.append(Table(page, label, caption, index, columns, rows))
            used_until = row_end
            break
    occupied = {table.label.casefold() for table in tables}
    tables.extend(read_transposed_tables(page, lines, captions, occupied))
    if page.verification == "text_match":
        tables.extend(read_native_comparisons(page, lines, occupied))
        tables.extend(read_qa_comparisons(page, lines, occupied))
        tables.extend(read_paired_scores(page, lines, occupied))
        tables.extend(read_failure_rates(page, occupied, failure_definition))
        tables.extend(read_failure_statistics(page, lines, occupied))
    return sorted(tables, key=lambda table: table.index)


def read_native_comparisons(page: Page, lines: list[str], occupied: set[str]) -> list[Table]:
    tables = []
    captions = []
    for index, line in enumerate(lines):
        for match in NATIVE_CAPTION.finditer(line):
            caption = re.split(r"\s{2,}", line[match.start(1):].rstrip(), maxsplit=1)[0]
            captions.append((index, match, caption))
    for index, line in enumerate(lines):
        cells = list(HEADER_CELLS.finditer(line.rstrip()))
        if len(cells) < 3 or not DELTA_HEADER.fullmatch(cells[-1][0]):
            continue
        if len(cells) >= 5 and DATASET_HEADER.fullmatch(cells[0][0]) and re.fullmatch(
            r"models?|generators?|llms?", cells[1][0], re.I,
        ):
            table = read_centered_benchmarks(page, lines, index, cells, captions)
        else:
            metric_at = next((i for i, cell in enumerate(cells) if cell[0].lower() in {"metric", "metrics"}), None)
            if metric_at is None or len(cells) - metric_at not in {3, 4}:
                continue
            table = read_metric_comparison(page, lines, index, cells[metric_at:], captions)
        if table and table.label.casefold() not in occupied:
            tables.append(table)
            occupied.add(table.label.casefold())
    return tables


def in_column(value: re.Match[str], heading: re.Match[str]) -> bool:
    return value.start() <= heading.end() + 4 and value.end() >= heading.start() - 4


def column_boundaries(centers: list[float]) -> list[float]:
    left_edge = centers[0] - (centers[1] - centers[0]) / 2
    right_edge = centers[-1] + (centers[-1] - centers[-2]) / 2
    return [left_edge, *((left + right) / 2 for left, right in pairwise(centers)), right_edge]


def aligned_values(line: str, boundaries: list[float], annotation: str = "") -> tuple[int, list[str]] | None:
    width = len(boundaries) - 1
    tokens = list(TOKEN.finditer(line))[-width:]
    if len(tokens) != width:
        return None
    values = []
    for token, (left, right) in zip(tokens, pairwise(boundaries), strict=True):
        center = (token.start() + token.end()) / 2
        value = token[0].removesuffix(annotation)
        if not left < center < right or GROUPED_NUMBER.fullmatch(value):
            return None
        if not NUMBER.fullmatch(value) and value not in MISSING:
            return None
        values.append(value)
    return tokens[0].start(), values


def aligned_label(line: str, label_at: int, value_at: int) -> str | None:
    match = TOKEN.search(line, max(0, label_at - 1), value_at)
    if not match or abs(match.start() - label_at) > 2:
        return None
    label = line[match.start():value_at].strip()
    if re.search(r"[^\W\d_]", label) and not re.search(r"\s{2,}", label):
        return label
    return None


def read_centered_benchmarks(page: Page, lines: list[str], index: int, cells: list[re.Match[str]],
                             captions: list[tuple[int, re.Match[str], str]]) -> Table | None:
    benchmark, model, *variants, _ = cells
    entries = []
    model_order = None
    body_end = index + 1
    # Each standalone benchmark is centered between the two rows it spans.
    while body_end + 2 < len(lines):
        above, middle, below = (list(HEADER_CELLS.finditer(line.rstrip())) for line in lines[body_end:body_end + 3])
        if len(middle) != 1 or abs(middle[0].start() - benchmark.start()) > 2:
            break
        dataset = middle[0][0]
        if middle[0].end() >= model.start() or not re.search(r"[^\W\d_]", dataset):
            break
        pair = []
        for row in (above, below):
            if len(row) != len(cells) - 1 or abs(row[0].start() - model.start()) > 2:
                return None
            if not all(in_column(value, heading) for value, heading in zip(row, cells[1:], strict=True)):
                return None
            if not re.search(r"[^\W\d_]", row[0][0]) or not NUMBER.fullmatch(row[-1][0]):
                return None
            values = [value[0] for value in row[1:-1]]
            if not all(value.endswith("%") and NUMBER.fullmatch(value) for value in values):
                return None
            pair.append((row[0][0], values))
        order = tuple(name for name, _ in pair)
        if order[0] == order[1] or (model_order is not None and order != model_order):
            return None
        if any(name == dataset for name, _, _ in entries):
            return None
        model_order = order
        entries.extend((dataset, name, values) for name, values in pair)
        body_end += 3

    footer = next(((at, match, caption) for at, match, caption in captions
                   if body_end <= at <= body_end + 2 and CAPTION.match(lines[at])
                   and all(not line.strip() for line in lines[body_end:at])), None)
    if len(entries) < 4 or footer is None:
        return None
    _, match, caption = footer
    if not distinct_labels([variant[0] for variant in variants]):
        return None
    metric = TRANSPOSED_METRIC.search(caption)
    header = Header(metric[1] if metric else "%", metric[1] if metric else "Unspecified score", 0, None, "%")
    columns = [Column(header, dataset, generator=name, original_row=f"{dataset} / {name}")
               for dataset, name, _ in entries]
    rows = [(variant[0], [values[i] for _, _, values in entries]) for i, variant in enumerate(variants)]
    return Table(page, match[1], caption, index, columns, rows)


def distinct_labels(labels: list[str]) -> bool:
    return all(re.search(r"[^\W\d_]", label) for label in labels) and len(set(map(str.casefold, labels))) == len(labels)


def read_metric_row(line: str, left: int, first_variant: int) -> tuple[re.Match[str], list[re.Match[str]]] | None:
    label = next((cell for cell in HEADER_CELLS.finditer(line.rstrip()) if abs(cell.start() - left) <= 2), None)
    if label is None or label.end() >= first_variant:
        return None
    tokens = list(TOKEN.finditer(line, label.end()))
    if len(tokens) != 3:
        return None
    scores, delta = tokens[:2], tokens[-1][0]
    if not all(NUMBER.fullmatch(score[0]) or GROUPED_NUMBER.fullmatch(score[0]) or score[0] in MISSING for score in scores):
        return None
    if delta not in MISSING and not NUMBER.fullmatch(delta.rstrip("*")):
        return None
    return label, tokens


def read_metric_comparison(page: Page, lines: list[str], index: int, cells: list[re.Match[str]],
                           captions: list[tuple[int, re.Match[str], str]]) -> Table | None:
    heading, *variants, delta = cells
    left = heading.start()
    first_variant = variants[0].start()
    nearby = [(at, match, caption) for at, match, caption in captions
              if index - 6 <= at and left - 6 <= match.start(1) < delta.end()]
    before = next((item for item in reversed(nearby) if item[0] < index and item[1].start(1) <= left + 4), None)
    if before:
        at = before[0]
        # A comparison grid directly above this caption owns it as a footer.
        if any(read_metric_row(line, left, first_variant) for line in lines[max(0, at - 3):at]):
            before = None
    after = next((item for item in nearby if item[0] > index), None)
    entries = []
    last = index
    stop = after[0] if after else len(lines)
    for at in range(index + 1, stop):
        row = read_metric_row(lines[at], left, first_variant)
        if row is None:
            if at - last > 1:
                break
            continue
        entries.append(row)
        last = at
    caption = before or (after if after and after[0] - last <= 3 else None)
    if len(entries) < 2 or not caption:
        return None
    if len(variants) == 1:
        # Poppler can reduce a column separator to one space. A unique break
        # past the first numeric column's center and near its right edge is required.
        merged = variants[0]
        left_center = median((tokens[0].start() + tokens[0].end()) / 2 for _, tokens in entries)
        right_edge = median(tokens[0].end() for _, tokens in entries)
        breaks = [gap for gap in re.compile(r" ").finditer(merged.string, merged.start(), merged.end())
                  if gap.end() > left_center and abs(gap.end() - right_edge) <= 2]
        if len(breaks) != 1:
            return None
        gap = breaks[0]
        variants = [
            HEADER_CELLS.search(merged.string, merged.start(), gap.start()),
            HEADER_CELLS.search(merged.string, gap.end(), merged.end()),
        ]
    variant_names = [variant[0] for variant in variants]
    if not distinct_labels(variant_names):
        return None
    if any(not in_column(token, heading) for _, tokens in entries
           for token, heading in zip(tokens, [*variants, delta], strict=True)):
        return None

    columns = []
    results = [[], []]
    for label, tokens in entries:
        values = [token[0] for token in tokens[:2]]
        # The value reader treats commas as decimal separators.
        if any(GROUPED_NUMBER.fullmatch(value) for value in values):
            continue
        unit = METRIC_UNIT.search(label[0])
        name = label[0][:unit.start()] if unit else label[0]
        header = Header(label[0], name, (label.start() + label.end()) / 2, None, unit[1] if unit else None)
        columns.append(Column(header, None, original_row=label[0]))
        for result, value in zip(results, values, strict=True):
            result.append(value)
    if not columns or not distinct_labels([column.header.label for column in columns]):
        return None
    _, match, text = caption
    rows = list(zip(variant_names, results, strict=True))
    return Table(page, match[1], text, index, columns, rows)


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
