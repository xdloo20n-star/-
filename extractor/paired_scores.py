"""Read native benchmark grids with source-grounded EM / F1 pairs."""

from dataclasses import replace
from decimal import Decimal
from itertools import pairwise
import re

from grouped_scores import DECIMAL, native_evidence
from pdf_pages import Page
from rag_tables import (
    Column, DATASET_HEADER, GROUPED_NUMBER, Header, MODEL_HEADER, NATIVE_CAPTION, Table, column_boundaries,
    distinct_labels,
)


PAIR = re.compile(rf"({DECIMAL.pattern})\s*/\s*({DECIMAL.pattern})")
MARKED_PAIR = re.compile(rf"({DECIMAL.pattern})(\*)?\s*/\s*({DECIMAL.pattern})(\*)?")
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
AVERAGE = {"average", "avg.", "avg"}
PAIR_PERCENT = re.compile(r"\b(?:EM|F1)\s*/\s*(?:EM|F1)\s*,\s*%", re.I)
PAIR_STAR_LEGEND = re.compile(r"[“\"‘']?\*[”\"’']?\s+(?:indicates?|denotes?)\s+p\s*<\s*\d+\.\d+", re.I)
ORDERED_HEADER = re.compile(r"\b(?:EM +F1|F1 +EM)\b")
PAIR_COUNT = re.compile(r"(?:preference\s+)?pairs", re.I)
REGEN_LABEL = re.compile(r"(outcome) (filter) \+ (\w+)", re.I)
MAIN_BACKBONE = re.compile(r"\bmain\s+experiments\s+use\s+([\w.-]+-Instruct)\s+as\s+the\s+policy\s+backbone\b", re.I)
MAIN_AVERAGE = re.compile(rf"({DECIMAL.pattern})\s*/\s*({DECIMAL.pattern})\s+for\s+the\s+main\s+instruction-tuned\s+backbone\b", re.I)
MAIN_FILTER = re.compile(r"\bthe same outcome-margin filter\b.{0,150}\bas in our main pipeline\b", re.I | re.S)
WITH_MARGIN_FILTER = re.compile(r"(.+)\s+\(with margin filter\)", re.I)
PRM_WEIGHT_SWEEP = re.compile(r"\bProcess Reward Model \(PRM\) weight \(α\)", re.I)
SELECTED_SETTING = re.compile(r"([⋆*]) denotes the selected configuration\b", re.I)
MAIN_TABLE = re.compile(r"\bthe main results comparing ([\w -]+?) with baseline methods across \w+ datasets are presented in\s+"
                        r"(Table \d+)\b", re.I)
DEFAULT_OPTIMIZER = re.compile(r"([\w-]+(?: [\w-]+)?) \(([A-Z]+)\) is our default approach\b", re.I)
PRM_ABLATION = re.compile(r"\bcomponent ablation study evaluating the impact of the Process Reward Model\b", re.I)
OPTIMIZER_ABLATION = re.compile(r"\bimpact of different optimization strategies\b", re.I)
BRANCH_ABLATION = re.compile(r"\bablation of the VTS branching configuration\b", re.I)
BRANCH_MAIN = re.compile(r"main configuration \(K\s*=\s*\d+\)", re.I)


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
        if not 4 <= len(header) <= 13 or header[-1][0].casefold() not in AVERAGE:
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


def read_additional_score_tables(pages: list[Page], known_tables: list[Table]) -> list[Table]:
    """Resolve layouts that need a second table as evidence or have different axes."""
    from classifier_tables import read_classifier_tables
    from recovery_scores import read_recovery_scores

    occupied = {(table.page.number, table.label.casefold()) for table in known_tables}
    tables = []
    for page, index, label, region in caption_regions(pages, occupied):
        grid = read_unlabeled_pairs(page, label, index, region, known_tables)
        transposed = read_transposed_pairs(page, label, index, region)
        tables.extend(grid or transposed)
    occupied.update((table.page.number, table.label.casefold()) for table in tables)
    tables += read_classifier_tables(pages, occupied)
    occupied.update((table.page.number, table.label.casefold()) for table in tables)
    return tables + read_recovery_scores(pages, occupied)


def caption_regions(pages: list[Page], occupied: set[tuple[int, str]]):
    for page in pages:
        if page.verification != "text_match":
            continue
        lines = page.text.splitlines()
        for index, line in enumerate(lines):
            matches = list(NATIVE_CAPTION.finditer(line))
            for offset, match in enumerate(matches):
                label = match[1]
                if (page.number, label.casefold()) in occupied:
                    continue
                # Adjacent column captions can be indented relative to their table body.
                end = matches[offset + 1].start(1) - 7 if offset + 1 < len(matches) else None
                left = max(0, match.start(1) - 6) if offset or match.start(1) >= 35 else 0
                region = [row[left:end].rstrip() for row in lines[index:index + 45]]
                region[0] = " " * (match.start(1) - left) + line[match.start(1):end].rstrip()
                yield page, index, label, region


def read_unlabeled_pairs(page: Page, label: str, start: int, lines: list[str],
                         known_tables: list[Table]) -> list[Table]:
    for index in range(1, min(8, len(lines))):
        if NATIVE_CAPTION.match(lines[index]):
            break
        headings = list(CELL.finditer(lines[index]))
        if not 4 <= len(headings) <= 13 or not ROW_HEADER.fullmatch(headings[0][0]):
            continue
        if headings[-1][0].casefold() not in AVERAGE:
            continue
        has_counts = bool(PAIR_COUNT.fullmatch(headings[1][0]))
        groups = [heading[0] for heading in headings[1 + has_counts:]]
        if len(groups) < 4 or not distinct_labels(groups):
            continue
        parsed = read_aligned_pairs(lines[index + 1:], headings, has_counts=has_counts)
        if len(parsed) < 2:
            return []
        rows = [(row, values) for row, values, _ in parsed if not DELTA_ROW.fullmatch(row)]
        order = reference_order(groups, rows, known_tables)
        if len(rows) < 2 or order is None:
            return []
        metrics, evidence = order
        caption = " ".join(" ".join(lines[:index]).split())
        columns = [
            make_score_column(group, metric, evidence=evidence, average=group == groups[-1])
            for group in groups for metric in metrics
        ]
        table = Table(page, label, caption, start + index, columns, rows, family="grouped_scores")
        tables = [table]
        if has_counts:
            header = Header(headings[1][0], "Preference pairs", 0, None, None)
            column = Column(header, None, grouped_digits=True)
            count_rows = [(row, [count]) for row, _, count in parsed if count is not None]
            tables.append(replace(table, columns=[column], rows=count_rows, family="preference_counts"))
        deltas = [(row, values) for row, values, _ in parsed if DELTA_ROW.fullmatch(row)]
        if delta := paired_delta(table, rows, deltas):
            tables.append(delta)
        return tables
    return []


def read_aligned_pairs(lines: list[str], headings: list[re.Match[str]], *, has_counts: bool = False,
                       marked: bool = False) -> list[tuple[str, list[str], str | None]]:
    label, *value_headers = headings
    centers = [cell_center(heading) for heading in value_headers]
    bounds = list(pairwise(column_boundaries(centers)))
    rows = []
    pattern = MARKED_PAIR if marked else PAIR
    for line in lines:
        if not line.strip() or NATIVE_CAPTION.match(line):
            break
        pairs = list(pattern.finditer(line))
        if not pairs:
            break
        if (len(pairs) != len(value_headers) - has_counts or line[pairs[-1].end():].strip()
                or any(line[a.end():b.start()].strip() for a, b in pairwise(pairs))):
            return []
        prefix = list(CELL.finditer(line[:pairs[0].start()].rstrip()))
        if len(prefix) != 1 + has_counts or abs(prefix[0].start() - label.start()) > 2:
            return []
        row = prefix[0]
        count = None
        if has_counts:
            count = prefix[1][0]
            if not (count.isdecimal() or GROUPED_NUMBER.fullmatch(count) and not count.endswith("%")):
                return []
        if any(not left < cell_center(value) < right
               for value, (left, right) in zip([*prefix[1:], *pairs], bounds, strict=True)):
            return []
        scores = [value for pair in pairs for value in (pair.groups()[::2] if marked else pair.groups())]
        rows.append((row[0], scores, count))
    return rows


def cell_center(match: re.Match[str]) -> float:
    return (match.start() + match.end()) / 2


def reference_order(groups: list[str], rows: list[tuple[str, list[str]]],
                    known_tables: list[Table]) -> tuple[tuple[str, str], tuple[Page, str]] | None:
    reference = {}
    target_groups = [group.casefold() for group in groups[:-1]] + [None]
    for table in known_tables:
        if table.family != "grouped_scores":
            continue
        sources = {(column.dataset.casefold() if column.dataset else None, column.header.name): index
                   for index, column in enumerate(table.columns)}
        if len(sources) != len(table.columns) or len(sources) != len(target_groups) * 2:
            continue
        source_lines = table.page.text.splitlines()
        header = ORDERED_HEADER.search(source_lines[table.index])
        evidence = (table.page, header[0]) if header else table.columns[0].name_evidence
        if not evidence:
            continue
        for metrics in (("EM", "F1"), ("F1", "EM")):
            keys = [(group, metric) for group in target_groups for metric in metrics]
            if not all(key in sources for key in keys):
                continue
            order = [sources[key] for key in keys]
            if any(values == [source[i] for i in order] for _, values in rows for _, source in table.rows):
                reference.setdefault(metrics, evidence)
    if len(reference) == 1:
        return next(iter(reference.items()))
    return None


def make_score_column(group: str, metric: str, *, evidence: tuple[Page, str], average: bool = False,
                      unit: str | None = None, transposed: bool = False) -> Column:
    label = f"{group} / {metric}" if average and not transposed else metric
    return Column(Header(label, metric, 0, None, unit), None if average else group,
                  "reported average" if average else None, original_row=group if transposed else None,
                  name_evidence=evidence)


def paired_delta(table: Table, rows: list[tuple[str, list[str]]],
                 deltas: list[tuple[str, list[str]]]) -> Table | None:
    if len(rows) != 2 or len(deltas) != 1:
        return None
    (base_name, base), (new_name, new) = rows
    match = REGEN_LABEL.fullmatch(new_name)
    if not match or base_name.casefold() != f"{match[1]} {match[2]} main".casefold():
        return None
    delta_name, values = deltas[0]
    # Independently rounded averages or scores can each differ by half the last printed digit.
    for raw_base, raw_new, raw_delta in zip(base, new, values, strict=True):
        operands = [Decimal(raw.replace("−", "-")) for raw in (raw_base, raw_new, raw_delta)]
        allowed = sum(Decimal("0.5").scaleb(value.as_tuple().exponent) for value in operands)
        if abs(operands[1] - operands[0] - operands[2]) > allowed:
            return None
    variant = f"{match[3]} minus {match[1]}-{match[2]} main"
    columns = [
        replace(column, header=replace(column.header, name=f"Delta {column.header.name}",
                                       label=f"Delta {column.header.name}"),
                aggregation=None, metric_variant=variant,
                source_label=f"{column.dataset or 'Average'} / Delta {column.header.name}")
        for column in table.columns
    ]
    return replace(table, columns=columns, rows=[(delta_name, values)], family="paired_deltas", config_name=new_name)


def read_transposed_pairs(page: Page, label: str, start: int, lines: list[str]) -> list[Table]:
    for index in range(1, min(8, len(lines))):
        headings = list(CELL.finditer(lines[index]))
        if NATIVE_CAPTION.match(lines[index]):
            break
        if len(headings) < 3 or not DATASET_HEADER.fullmatch(headings[0][0]):
            continue
        methods = [heading[0] for heading in headings[1:]]
        if not distinct_labels(methods):
            return []
        caption = "\n".join(lines[:index]).strip()
        if not METRIC_ORDER.search(caption):
            return []
        rows = read_aligned_pairs(lines[index + 1:], headings, marked=bool(PAIR_STAR_LEGEND.search(caption)))
        if len(rows) < 3 or not distinct_labels([row for row, _, _ in rows]):
            return []
        order = score_order(caption, [])
        if order is None:
            return []
        metrics, fragment = order
        unit = "%" if PAIR_PERCENT.search(caption) else None
        columns = [
            make_score_column(dataset, metric, evidence=(page, fragment), average=dataset.casefold() in AVERAGE,
                              unit=unit, transposed=True)
            for dataset, _, _ in rows for metric in metrics
        ]
        method_rows = [(method, [raw for _, values, _ in rows for raw in values[2 * i:2 * i + 2]])
                       for i, method in enumerate(methods)]
        return [Table(page, label, " ".join(caption.split()), start + index, columns, method_rows,
                      family="grouped_scores")]
    return []


def enrich_repeated_score_rows(pages: list[Page], tables: list[Table]) -> list[Table]:
    """Link exact copies of a benchmark row when the PDF identifies the same main setting."""
    tables = link_main_ablation_rows(pages, tables)
    backbone = native_evidence(pages, MAIN_BACKBONE)
    main_average = native_evidence(pages, MAIN_AVERAGE)
    if not (backbone and main_average):
        return tables
    model = MAIN_BACKBONE.fullmatch(backbone[1])[1]
    scores = MAIN_AVERAGE.fullmatch(main_average[1]).groups()
    models = [table for table in tables if table.family == "grouped_scores" and table.row_section
              and table.row_section.casefold() == f"{model.removesuffix('-Instruct')} Base/Instruct".casefold()]
    main = [(table, row, values) for table in models for row, values in table.rows
            if row.endswith("-Instruct") and tuple(values[-2:]) == scores]
    if len(main) != 1:
        return tables
    reference, method, raw = main[0]
    reference_scores = score_cells(reference.columns, raw)
    name = re.sub(r"(?<=\w)-S\s+TEP\b", "-STEP", method.removesuffix("-Instruct"))
    scope = reference.page.number, f"{reference.label} / {method}"
    # The same filter is named in the main and regeneration conditions.
    same_filter = native_evidence(pages, MAIN_FILTER)
    candidates = []
    for table in tables:
        if table.family != "grouped_scores" or table is reference:
            continue
        selected = SELECTED_SETTING.search(table.caption) if PRM_WEIGHT_SWEEP.search(table.caption) else None
        selected_rows = [row for row, _ in table.rows if selected and row.endswith(f" {selected[1]}")]
        for row, values in table.rows:
            margin = WITH_MARGIN_FILTER.fullmatch(row)
            condition = (margin and row.startswith(method.removesuffix("-Instruct"))
                         and "outcome-margin filter" in table.caption.casefold()) or (
                             same_filter and row.casefold() == "outcome filter main"
                             and "regen-augmented variant" in table.caption.casefold()) or selected_rows == [row]
            if condition and score_cells(table.columns, values) == reference_scores:
                candidates.append((table, row))
    if not candidates:
        return tables
    return tag_repeated_rows(tables, reference, method, candidates, scope, name, model, backbone)


def link_main_ablation_rows(pages: list[Page], tables: list[Table]) -> list[Table]:
    if not (statement := native_evidence(pages, MAIN_TABLE)):
        return tables
    method, label = MAIN_TABLE.fullmatch(statement[1]).groups()
    main = [(table, row, values) for table in tables if table.family == "grouped_scores" and table.label == label
            for row, values in table.rows if row == method]
    if len(main) != 1:
        return tables
    reference, main_row, raw = main[0]
    scores = score_cells(reference.columns, raw)
    optimizer = native_evidence(pages, DEFAULT_OPTIMIZER)
    default = DEFAULT_OPTIMIZER.fullmatch(optimizer[1]).groups() if optimizer else None
    default_row = f"{default[0]} ({default[1]})" if default and default[0] == method else None
    candidates = []
    for table in tables:
        if table.family != "grouped_scores" or table.page.number < reference.page.number or table is reference:
            continue
        for row, values in table.rows:
            condition = (row == method and PRM_ABLATION.search(table.caption)) or (
                row == default_row and OPTIMIZER_ABLATION.search(table.caption)) or (
                BRANCH_MAIN.fullmatch(row) and BRANCH_ABLATION.search(table.caption))
            if condition and score_cells(table.columns, values) == scores:
                candidates.append((table, row))
    if not candidates:
        return tables
    scope = reference.page.number, f"{reference.label} / {main_row}"
    return tag_repeated_rows(tables, reference, main_row, candidates, scope, method)


def tag_repeated_rows(tables: list[Table], reference: Table, method: str, candidates: list[tuple[Table, str]],
                      scope: tuple[int, str], name: str, model: str | None = None,
                      backbone: tuple[Page, str] | None = None) -> list[Table]:
    reference_datasets = {dataset_key(column.dataset): column.dataset for column in reference.columns}
    repeated = {id(table): row for table, row in [(reference, method), *candidates]}
    enriched = []
    for table in tables:
        if not (duplicate_row := repeated.get(id(table))):
            enriched.append(table)
            continue
        for index, (row, values) in enumerate(table.rows):
            bboxes = table.value_bboxes[index:index + 1] if table.value_bboxes else ()
            single_row = replace(table, rows=[(row, values)], value_bboxes=bboxes)
            if row == duplicate_row:
                columns = [replace(column, generator=model, generator_evidence=backbone, duplicate_scope=scope,
                                   dataset=reference_datasets[dataset_key(column.dataset)],
                                   source_label=score_locator(column)) for column in table.columns]
                single_row = replace(single_row, columns=columns, config_name=name)
            enriched.append(single_row)
    return enriched


def score_cells(columns: list[Column], values: list[str]) -> dict[tuple, str]:
    return {(dataset_key(column.dataset), column.header.name, column.header.unit, column.aggregation): value
            for column, value in zip(columns, values, strict=True)}


def dataset_key(dataset: str | None) -> str | None:
    if dataset and dataset.casefold() == "2wikimulti":
        return "2wikimultihopqa"
    return dataset.casefold() if dataset else None


def score_locator(column: Column) -> str:
    return " / ".join(item for item in (column.dataset, column.header.label, column.aggregation) if item)
