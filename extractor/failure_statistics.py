"""Read native failure-rate tables with paired bootstrap confidence intervals."""

from dataclasses import dataclass
from functools import cache
import re

from pdf_pages import Page
from rag_tables import Column, Header, Table


PAIRED_METHODS = re.compile(r"^\s*(\S(?:.*?\S)?)\s{2,}(\S(?:.*?\S)?)\s{2,}[Δ∆]\s{2,}McNemar\s*$")
ROW_HEAD = re.compile(r"^\s*Dataset\s{2,}(Failure)\s+(Model)\s*$")
SCALES = re.compile(r"^\s*%\s*\[95%\s+CI]\s{2,}%\s*\[95%\s+CI]\s{2,}pp\s{2,}p\s*$")
CAPTION = re.compile(r"^\s*(Table\s+[A-Z]?\d+(?:\.\d+)*):\s", re.I)
BOOTSTRAP = re.compile(
    r"\b95%\s+bootstrap\s+confidence\s+intervals\s+over\s+(\d{1,3}(?:,\d{3})*|\d+)\s+resamples\b",
    re.I,
)
FAILURES = {"Leak": "Leakage Rate", "Stale": "Stale Citation Rate", "Unsup.": "Unsupported Answer Rate"}
FAILURE_CELL = re.compile(r"^(Leak|Stale|Unsup\.)(?=\s|$)")
FAILURE_LEGEND = re.compile(
    r"\bLeak,\s*Stale,\s*(?:and\s+)?Unsup\.\s+denote\s+unauthorized\s+leakage,\s*"
    r"stale\s+citation,\s*(?:and\s+)?unsupported[-\s]+answer\s+rates", re.I,
)
MODEL = re.compile(r"[A-Za-z][\w.-]*(?: [\w.()-]+)*")
MODEL_SUFFIX = re.compile(r"[A-Za-z0-9][\w.]*")
NUMBER = r"(?:\d+(?:\.\d+)?|\.\d+)"
INTERVAL = rf"({NUMBER}) +\[({NUMBER}), *({NUMBER})\]"
STATISTICS = re.compile(rf"(?<!\S){INTERVAL} +{INTERVAL} +([+−-]?{NUMBER}) +({NUMBER}(?:[eE][+−-]?\d+)?) *$")


@dataclass(frozen=True)
class StatisticRow:
    line: int
    model: str
    values: tuple[str, ...]


def read_failure_statistics(page: Page, lines: list[str], occupied: set[str]) -> list[Table]:
    if page.verification != "text_match":
        return []
    tables = []
    for index in range(1, len(lines) - 3):
        heads = ROW_HEAD.fullmatch(lines[index])
        names = PAIRED_METHODS.fullmatch(lines[index - 1])
        if heads is None or names is None or not SCALES.fullmatch(lines[index + 1]):
            continue
        start = index + 2
        end = next((i for i in range(start, len(lines)) if not lines[i].strip()), len(lines))
        caption_at = next((i for i in range(end, min(end + 4, len(lines))) if lines[i].strip()), len(lines))
        if caption_at == len(lines) or not (label := CAPTION.match(lines[caption_at])):
            continue
        if label[1].casefold() in occupied:
            continue
        caption_end = next((i for i in range(caption_at, len(lines)) if not lines[i].strip()), len(lines))
        caption = "\n".join(lines[caption_at:caption_end])
        if not (bootstrap := BOOTSTRAP.search(caption)) or not FAILURE_LEGEND.search(caption):
            continue
        baseline, treatment = (" ".join(name.split()) for name in names.groups())
        if not paired_caption(caption, baseline, treatment):
            continue
        rows = statistic_rows(lines[start:end], heads.start(1), heads.start(2))
        if rows:
            tables.extend(statistic_tables(page, label[1], index, caption, bootstrap, baseline, treatment, rows))
    return tables


def paired_caption(caption: str, baseline: str, treatment: str) -> bool:
    baseline, treatment = re.escape(baseline), re.escape(treatment)
    prose = " ".join(caption.split())
    return bool(
        re.search(rf"[Δ∆] pp denotes the absolute reduction from {baseline} to {treatment}\b", prose, re.I)
        and re.search(rf"\bpaired McNemar tests compare {baseline} and {treatment} on the same examples\b", prose, re.I)
    )


def dataset_name(label: str) -> str:
    # Poppler may insert a space between the initial capital and small capitals.
    return re.sub(r"(?<![A-Z])([A-Z])\s+([A-Z]{2,})(?=$|\s|[-–])", r"\1\2", label).replace(" -", "-")


def statistic_rows(lines: list[str], failure_at: int, model_at: int) -> list[tuple[str, str, StatisticRow]]:
    datasets = []
    failures = []
    model_cells = []
    value_cells = []
    for index, line in enumerate(lines):
        values = STATISTICS.search(line)
        prefix = line[:values.start()] if values else line
        dataset = prefix[:failure_at].strip()
        if dataset:
            if not dataset[0].isalpha():
                return []
            datasets.append((index, dataset_name(dataset)))
        rest = prefix[failure_at:]
        failure = FAILURE_CELL.match(rest.lstrip())
        if failure:
            failures.append((index, failure[1]))
            model = rest.lstrip()[failure.end():].strip()
        else:
            model = rest.strip()
            if model and failure_at + len(rest) - len(rest.lstrip()) < model_at - 1:
                return []
        model_cells.append(model)
        if values:
            value_cells.append((index, values.groups()))
        elif not (dataset or failure or model):
            return []

    model_lines = set()
    rows = []
    for index, values in value_cells:
        model = model_cells[index]
        if model:
            model_lines.add(index)
        elif 0 < index < len(model_cells) - 1:
            prefix, suffix = model_cells[index - 1], model_cells[index + 1]
            if prefix.endswith("-") and MODEL_SUFFIX.fullmatch(suffix):
                model = prefix + suffix
                model_lines.update((index - 1, index + 1))
        if not MODEL.fullmatch(model) or model.endswith("-") or not valid_statistics(values):
            return []
        rows.append(StatisticRow(index, model, values))
    if any(model and index not in model_lines for index, model in enumerate(model_cells)):
        return []

    # The order of generators repeats across failures. An omitted all-zero row
    # may leave a subset, but a second possible grouping makes the scope unsafe.
    model_order = dict.fromkeys(row.model for row in rows)
    if not 2 <= len(model_order) <= 6:
        return []
    rank = {model: index for index, model in enumerate(model_order)}
    scopes = centered_groups(failures, [(row.line, row.line) for row in rows],
                             [rank[row.model] for row in rows], ordered=True)
    if not scopes:
        return []
    dataset_groups = centered_groups(datasets, [(rows[scope.start].line, rows[scope.stop - 1].line) for scope in scopes],
                                     [failure for _, failure in failures])
    return [(dataset, failures[group][1], rows[row])
            for (_, dataset), groups in zip(datasets, dataset_groups, strict=True)
            for group in groups for row in scopes[group]] if dataset_groups else []


def valid_statistics(values: tuple[str, ...]) -> bool:
    first, first_low, first_high, second, second_low, second_high, delta, probability = (
        float(value.replace("−", "-")) for value in values
    )
    return (0 <= first_low <= first <= first_high <= 100 and 0 <= second_low <= second <= second_high <= 100
            and -100 <= delta <= 100 and 0 <= probability <= 1)


def centered_groups(markers: list[tuple[int, str]], spans: list[tuple[int, int]],
                    keys: list[str] | list[int], *, ordered: bool = False) -> list[range]:
    if not markers or not spans:
        return []

    @cache
    def partitions(index: int, start: int) -> list[tuple[int, ...]]:
        if index == len(markers):
            return [()] if start == len(spans) else []
        if start >= len(spans) or spans[start][0] > markers[index][0]:
            return []
        found = []
        used = set()
        for end in range(start, len(spans)):
            key = keys[end]
            if key in used or (ordered and end > start and key < keys[end - 1]):
                break
            used.add(key)
            if spans[end][1] < markers[index][0]:
                continue
            found.extend((end + 1, *rest) for rest in partitions(index + 1, end + 1))
            if len(found) > 1:
                break
        return found[:2]

    found = partitions(0, 0)
    if len(found) != 1:
        return []
    endpoints = found[0]
    return [range(start, end) for start, end in zip((0, *endpoints[:-1]), endpoints, strict=True)]


def statistic_tables(page: Page, label: str, index: int, caption: str, bootstrap: re.Match[str], baseline: str,
                     treatment: str, rows: list[tuple[str, str, StatisticRow]]) -> list[Table]:
    display_caption = " ".join(caption.split())
    bootstrap_aggregation = f"bootstrap {bootstrap[1]} resamples"
    tables = []
    for method, offset in ((baseline, 0), (treatment, 3), (f"{treatment} vs {baseline}", 6)):
        columns = []
        values = []
        for dataset, failure, row in rows:
            name = FAILURES[failure]
            row_locator = f"{dataset} / {failure} / {row.model}"
            if offset == 6:
                specifications = (
                    (f"{name} reduction vs {baseline}", "∆ pp", "pp", None, None),
                    (f"{name} paired McNemar p", "McNemar p", None, None, None),
                )
            else:
                column = f"{method} / % [95% CI]"
                specifications = (
                    (name, f"{column} / point estimate", "%", "per-model rate", "lower_is_better"),
                    (f"{name} 95% CI lower", f"{column} / lower bound", "%", bootstrap_aggregation, None),
                    (f"{name} 95% CI upper", f"{column} / upper bound", "%", bootstrap_aggregation, None),
                )
            for position, (metric, locator, unit, aggregation, direction) in enumerate(specifications, offset):
                evidence = (page, bootstrap[0]) if aggregation == bootstrap_aggregation else None
                columns.append(Column(Header(locator, metric, 0, None, unit), dataset, aggregation, row.model,
                                      original_row=row_locator, aggregation_evidence=evidence,
                                      direction=direction, source_label=locator))
                values.append(row.values[position])
        tables.append(Table(page, label, display_caption, index, columns, [(method, values)],
                            family="failure_statistics"))
    return tables
