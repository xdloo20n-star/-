"""Read native failure-rate ablations, stress ranges, and judge summaries."""

from dataclasses import replace
import re

from failure_diagnostics import FAILURES, MODEL, SUFFIX, row_models
from failure_statistics import CAPTION, NUMBER, centered_groups, dataset_name
from pdf_pages import Page
from rag_tables import Column, Header, Table


RATE_LABELS = ("Leak", "Stale", "Unsup.")
RANGE = rf"{NUMBER}(?:[–-]{NUMBER})?"
ABLATION_HEADING = re.compile(
    r"^\s*(Block)\s{2,}(Setting)\s{2,}(Method / Condition)\s{2,}Report\s+Leak ↓\s+Stale ↓\s+Unsup\. ↓\s*$",
)
ABLATION_VALUES = re.compile(rf"(?<!\S)(Range|Avg\.)\s+({RANGE})\s+({RANGE})\s+({RANGE})\s*$")
ABLATION_SCOPE = re.compile(
    r"Range reports the minimum–maximum failure rate across the three evaluated LLMs, "
    r"while Avg\. reports their macro-average",
)
ROBUSTNESS_HEADING = re.compile(r"^\s*Eval\.\s{2,}(Metric)\s{2,}Naive\s{2,}TrustRAG\s*$")
ROBUSTNESS_VALUES = re.compile(
    rf"^\s*(Leak|Stale|Unsup\.|Valid|Over-refusal)\s*([↑↓])\s+({RANGE})\s+({NUMBER})\s*$",
)
ROBUSTNESS_KAPPA = re.compile(rf"(?<!\S)((?:[−-]?{NUMBER}\s+){{2}}[−-]?{NUMBER})\s*$")
ROBUSTNESS_SCALE = re.compile(
    r"For robustness and utility rows, values compare (Naive RAG) with (TrustRAG) and are percentages\. "
    r"Ranges are across ([\w.-]+), ([\w.-]+), and ([\w.-]+)\.",
)
ROBUSTNESS_DATASETS = re.compile(r"\bS\s*OURCE, SF, and F\s*RESH denote ([A-Z_]+), ([A-Z_]+), and ([A-Z_]+), respectively")
ROBUSTNESS_CAPTION_END = re.compile(ROBUSTNESS_DATASETS.pattern + r"\.")
PER_MODEL_HEADING = re.compile(r"^\s*Model\s{2,}(Variant)\s{2,}N\s+Leak ↓\s+Stale ↓\s+Unsup\. ↓\s*$")
PER_MODEL_VALUES = re.compile(rf"(?<!\S)(\d+)\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*$")
PER_MODEL_ABLATION = re.compile(r"^\s*(?:Access-filter|Supersession-filter|Conflict-reporting) ablation on (.+)\s*$")
PER_MODEL_SCOPE = re.compile(
    r"Rows labeled MacroAvg report the arithmetic mean over the three evaluated LLMs\. "
    r"(Rows labeled All summarize the matched TrustRAG runs across the same model set)\. "
    r"All values except N are failure rates \(%\)",
)
PER_MODEL_VARIANTS = re.compile(
    r"Abbreviations: (w/o Access), (w/o Supersession Filtering), and (w/o Conflict) denote removing "
    r"(access filtering), (supersession filtering), and (conflict reporting), respectively",
)
SIDE_CAPTION = re.compile(r"(?<= {6})Table [A-Z]?\d+:")
SIDE_GRID = re.compile(r"^\s{6,}(Dataset)\s{2,}Condition\s{2,}N\s*$")
ABLATION_SUMMARY_REF = re.compile(r"\b(\d+\.\d+) Ablations and Adversarial Stress Tests (Table \d+) analyzes\b")
ABLATION_BREAKDOWN_REF = re.compile(
    r"\b(Table [A-Z]?\d+) expands the ablation and adversarial stress-test results from Section (\d+\.\d+)\.",
)
MACRO_AVERAGE = "macro-average across 3 LLMs"


def read_failure_ablations(page: Page, lines: list[str], occupied: set[str]) -> list[Table]:
    tables = []
    for index, line in enumerate(lines):
        if line.lstrip().startswith("Eval."):
            left = left_panel(lines[index:])
            heading = ROBUSTNESS_HEADING.fullmatch(left[0])
            if heading is not None:
                tables.extend(read_robustness(page, left, index, heading.start(1), occupied))
        elif heading := ABLATION_HEADING.fullmatch(line):
            tables.extend(read_ablations(page, lines[index:], index, heading, occupied))
        elif heading := PER_MODEL_HEADING.fullmatch(line):
            left = left_panel(lines[index:])
            tables.extend(read_model_breakdown(page, left, index, heading.start(1) - 4, occupied))
    return tables


def left_panel(lines: list[str]) -> list[str]:
    positions = [match.start() - 2 for line in lines if (match := SIDE_CAPTION.search(line))]
    positions.extend(match.start(1) - 2 for line in lines if (match := SIDE_GRID.fullmatch(line)))
    return [line[:min(positions)].rstrip() for line in lines] if positions else lines


def caption_block(lines: list[str], occupied: set[str],
                  complete: re.Pattern[str] | None = None) -> tuple[str, str, list[str]] | None:
    for i, line in enumerate(lines[1:], 1):
        match = CAPTION.match(line)
        if match is None:
            continue
        label = match[1]
        if label.casefold() in occupied:
            return None
        # Layout text can insert right-column lines between lines of the caption.
        end = i + 16 if complete else next((j for j in range(i + 1, len(lines)) if not lines[j].strip()), len(lines))
        caption = re.sub(r"(?<=\w)-\n\s*(?=\w)", "", "\n".join(line for line in lines[i:end] if line.strip()))
        caption = " ".join(caption.split())
        if complete:
            stop = complete.search(caption)
            if stop is None:
                return None
            caption = caption[:stop.end()]
        return label, caption, lines[1:i]
    return None


def range_ends(raw: str) -> list[str] | None:
    parts = re.split(r"[–-]", raw)
    first, last = parts[0], parts[-1]
    return [first, last] if 0 <= float(first) <= float(last) <= 100 else None


def range_columns(metric: str, header: str, dataset: str, count: int, direction: str) -> list[Column]:
    return [Column(Header(header, f"{metric} {which}", 0, None, "%"), dataset,
                   f"range across {count} LLMs", direction=direction, source_label=f"{header} / {which} (Range)")
            for which in ("minimum", "maximum")]


def read_ablations(page: Page, lines: list[str], index: int, heading: re.Match[str],
                   occupied: set[str]) -> list[Table]:
    block = caption_block(lines, occupied)
    if block is None:
        return []
    label, caption, body = block
    if not ABLATION_SCOPE.search(caption) or "failure rates (%)" not in caption:
        return []
    tables = []
    condition_at = heading.start(3)
    settings = []
    scores = []
    for i, line in enumerate(body):
        match = ABLATION_VALUES.search(line)
        prefix = line[:match.start()] if match else line
        if prefix.strip() in {"Component ablations", "Adversarial inadmissible evidence"}:
            panel = ablation_panel(page, label, caption, index, settings, scores)
            if scores and not panel:
                return []
            tables.extend(panel)
            settings, scores = [], []
            continue
        # Poppler may start a centered setting just left of its header.
        setting = prefix[heading.start(2) - 3:condition_at].strip()
        if setting:
            settings.append((i, dataset_name(setting)))
        if match:
            condition = prefix[condition_at:].strip()
            if not condition:
                return []
            scores.append((i, condition, match[1], match.groups()[1:]))
        elif prefix[condition_at:].strip():
            return []
    panel = ablation_panel(page, label, caption, index, settings, scores)
    return tables + panel if panel else []


def ablation_panel(page: Page, label: str, caption: str, index: int, settings: list[tuple[int, str]],
                   scores: list[tuple[int, str, str, tuple[str, ...]]]) -> list[Table]:
    groups = centered_groups(settings, [(i, i) for i, *_ in scores], [name for _, name, *_ in scores])
    tables = []
    for (_, dataset), group in zip(settings, groups):
        for position in group:
            i, condition, report, values = scores[position]
            if report == "Range":
                endpoints = [range_ends(raw) for raw in values]
                if any(pair is None for pair in endpoints):
                    return []
                columns = [column for header, metric in zip(RATE_LABELS, FAILURES, strict=True)
                           for column in range_columns(metric, header, dataset, 3, "lower_is_better")]
                raw_values = [raw for pair in endpoints for raw in pair]
            else:
                if not all(re.fullmatch(NUMBER, raw) and float(raw) <= 100 for raw in values):
                    return []
                columns = failure_columns(dataset, MACRO_AVERAGE)
                raw_values = list(values)
            tables.append(Table(page, label, caption, index + i, columns, [(condition, raw_values)],
                                row_section=dataset, family="failure_rates"))
    return tables


def read_robustness(page: Page, lines: list[str], index: int, metric_at: int,
                    occupied: set[str]) -> list[Table]:
    block = caption_block(lines, occupied, ROBUSTNESS_CAPTION_END)
    if block is None:
        return []
    label, caption, body = block
    scale = ROBUSTNESS_SCALE.search(caption)
    aliases = ROBUSTNESS_DATASETS.search(caption)
    if scale is None or aliases is None or "Cohen’s κ between one automatic judge and human labels" not in caption:
        return []
    names = dict(zip(("SOURCE", "SF", "FRESH"), aliases.groups(), strict=True))
    trimmed = [line.strip() for line in body]
    try:
        robustness, utility, judge = (trimmed.index(name) for name in
                                     ("Robustness checks", "Utility diagnostics", "Judge reliability"))
    except ValueError:
        return []
    if not robustness < utility < judge:
        return []
    tables = []
    for first, last in ((robustness + 1, utility), (utility + 1, judge)):
        tables.extend(robustness_panel(page, label, caption, index + first, metric_at, body[first:last], names,
                                      scale.group(1, 2)))
    return tables + kappa_rows(page, label, caption, index + judge, body[judge + 1:])


def robustness_panel(page: Page, label: str, caption: str, index: int, metric_at: int, lines: list[str],
                     names: dict[str, str], methods: tuple[str, str]) -> list[Table]:
    metrics = dict(zip(RATE_LABELS + ("Valid", "Over-refusal"),
                       FAILURES + ("Valid answer rate", "Over-refusal rate"), strict=True))
    settings = []
    scores = []
    for i, line in enumerate(lines):
        setting = dataset_name(line[:metric_at].strip())
        if setting:
            settings.append((i, names.get(setting, setting)))
        match = ROBUSTNESS_VALUES.fullmatch(line[metric_at:])
        if match:
            scores.append((i, *match.groups()))
        elif line[metric_at:].strip():
            return []
    groups = centered_groups(settings, [(i, i) for i, *_ in scores], [metric for _, metric, *_ in scores])
    tables = []
    for (_, dataset), group in zip(settings, groups):
        for position in group:
            i, header, arrow, limits, score = scores[position]
            ends = range_ends(limits)
            if ends is None or not re.search(r"[–-]", limits) or float(score) > 100:
                return []
            metric = metrics[header]
            direction = "lower_is_better" if arrow == "↓" else "higher_is_better"
            columns = range_columns(metric, header, dataset, 3, direction)
            tables.append(Table(page, label, caption, index + i, columns, [(methods[0], ends)],
                                row_section=f"{dataset} / {header}", family="failure_rates"))
            column = Column(Header(header, f"{metric} all-model value", 0, None, "%"), dataset,
                            "same value across 3 LLMs", direction=direction, source_label=f"{methods[1]} / {header}")
            tables.append(Table(page, label, caption, index + i, [column], [(methods[1], [score])],
                                row_section=f"{dataset} / {header}", family="failure_rates"))
    return tables


def kappa_rows(page: Page, label: str, caption: str, index: int, lines: list[str]) -> list[Table]:
    headers = ("κ Leak", "κ Stale", "κ Unsup.")
    if not lines or not re.fullmatch(r"\s*Judge\s+κ Leak\s+κ Stale\s+κ Unsup\.\s*", lines[0]):
        return []
    cells, scores = [], {}
    for i, line in enumerate(line for line in lines[1:] if line.strip()):
        match = ROBUSTNESS_KAPPA.search(line)
        cells.append(line[:match.start()].strip() if match else line.strip())
        if match:
            values = match[1].split()
            if not all(-1 <= float(raw.replace("−", "-")) <= 1 for raw in values):
                return []
            scores[i] = values
    used = set()
    tables = []
    for i, values in scores.items():
        judge = cells[i]
        if not judge and i >= 2 and all(j not in used and j not in scores for j in (i - 2, i - 1)):
            if cells[i - 2].endswith("-") and SUFFIX.fullmatch(cells[i - 1]):
                judge = cells[i - 2] + cells[i - 1]
                used.update((i - 2, i - 1))
        elif judge.endswith("-") and i + 1 < len(cells) and i + 1 not in scores and SUFFIX.fullmatch(cells[i + 1]):
            judge += cells[i + 1]
            used.add(i + 1)
        if not MODEL.fullmatch(judge):
            return []
        used.add(i)
        columns = [Column(Header(header, "Cohen kappa", 0, None, None), None,
                          original_row=judge, source_label=header, metric_variant=failure,
                          metric_variant_evidence=(page, header))
                   for header, failure in zip(headers, FAILURES, strict=True)]
        tables.append(Table(page, label, caption, index + i, columns, [(judge, values)],
                            config_name=f"{judge} judge vs human labels", family="judge_summary"))
    return tables if len(used) == len(cells) else []


def read_model_breakdown(page: Page, lines: list[str], index: int, variant_at: int,
                         occupied: set[str]) -> list[Table]:
    block = caption_block(lines, occupied, PER_MODEL_SCOPE)
    if block is None:
        return []
    label, caption, body = block
    variants = PER_MODEL_VARIANTS.search(caption)
    scope = PER_MODEL_SCOPE.search(caption)
    if scope is None or variants is None:
        return []
    aliases = {short: f"w/o {long.title()}" for short, long in zip(variants.groups()[:3], variants.groups()[3:])}
    tables = []
    dataset = None
    start = 0
    for i, line in enumerate(body + [""]):
        heading = PER_MODEL_ABLATION.fullmatch(line)
        if not (heading or line.strip() == "Prompt-injection stress" or i == len(body)):
            continue
        if dataset:
            panel = model_breakdown_rows(page, label, caption, index + start, variant_at, body[start:i],
                                         dataset, aliases, (page.number, scope[1]))
            if not panel:
                return []
            tables.extend(panel)
        elif any(line.strip() for line in body[start:i]):
            return []
        dataset = dataset_name(heading[1].strip()) if heading else "Prompt-injection"
        start = i + 1
    return tables


def model_breakdown_rows(page: Page, label: str, caption: str, index: int, variant_at: int,
                         lines: list[str], dataset: str, aliases: dict[str, str], scope: tuple[int, str]) -> list[Table]:
    models = []
    variants = []
    scores = {}
    for i, line in enumerate(lines):
        match = PER_MODEL_VALUES.search(line)
        prefix = line[:match.start()] if match else line
        boundary = variant_at - 1 if prefix[variant_at - 1:].startswith("w/o ") else variant_at
        models.append(prefix[:boundary].strip())
        variants.append(prefix[boundary:].strip())
        if match:
            size = int(match[1])
            values = list(match.groups()[1:])
            if size <= 0 or any(float(raw) > 100 for raw in values):
                return []
            scores[i] = size, values
    aggregates = {"All": "pooled matched 3 LLMs", "MacroAvg": MACRO_AVERAGE}
    models = row_models(models, set(scores), set(aggregates))
    if models is None:
        return []
    used = set()
    tables = []
    for i, (size, values) in scores.items():
        name = variants[i]
        if not name and 0 < i < len(variants) - 1 and all(j not in scores and j not in used for j in (i - 1, i + 1)):
            name = f"{variants[i - 1]} {variants[i + 1]}"
            if name in aliases:
                used.update((i - 1, i + 1))
        if not name.strip():
            return []
        used.add(i)
        method = aliases.get(name, name)
        model = models[i]
        row = f"{dataset} / {model} / {name}"
        columns = failure_columns(dataset, aggregates.get(model, "per-model rate"),
                                  model if model not in aggregates else None, row, size)
        if model == "All":
            columns = [replace(column, duplicate_scope=scope) for column in columns]
        tables.append(Table(page, label, caption, index + i, columns, [(method, values)], family="failure_statistics"))
    return tables if all(not variant or i in used for i, variant in enumerate(variants)) else []


def failure_columns(dataset: str, aggregation: str, generator: str | None = None,
                    row: str | None = None, size: int | None = None) -> list[Column]:
    return [Column(Header(header, metric, 0, None, "%"), dataset, aggregation, generator,
                   original_row=row, direction="lower_is_better", source_label=header, sample_size=size)
            for header, metric in zip(RATE_LABELS, FAILURES, strict=True)]


def link_ablation_averages(pages: list[Page], tables: list[Table]) -> list[Table]:
    averages = [(i, table) for i, table in enumerate(tables) if table.family in {"failure_rates", "failure_statistics"}
                and table.columns[0].aggregation == MACRO_AVERAGE and table.columns[0].dataset
                and table.columns[0].dataset.casefold() == "prompt-injection"]
    if not averages:
        return tables
    summaries = [(match[1], match[2]) for page in pages if page.verification == "text_match"
                 for match in ABLATION_SUMMARY_REF.finditer(page.prose)]
    breakdowns = [(page, match) for page in pages if page.verification == "text_match"
                  for match in ABLATION_BREAKDOWN_REF.finditer(page.prose)]
    if len(summaries) != 1 or len(breakdowns) != 1:
        return tables
    section, summary_label = summaries[0]
    proof_page, reference = breakdowns[0]
    if reference[2] != section:
        return tables
    matching = [(i, table) for i, table in averages if table.label == summary_label and ABLATION_SCOPE.search(table.caption)]
    updated = list(tables)
    scope = proof_page.number, reference[0]
    for i, appendix in averages:
        if appendix.label != reference[1] or not PER_MODEL_SCOPE.search(appendix.caption):
            continue
        name, values = appendix.rows[0]
        size = appendix.columns[0].sample_size
        if size is None:
            continue
        candidates = [(prior_i, prior) for prior_i, prior in matching if prior.page.number < appendix.page.number
                      and method_abbreviation(name, prior.rows[0][0])]
        if len(candidates) != 1:
            continue
        prior_i, prior = candidates[0]
        if values != prior.rows[0][1] or updated[prior_i] is not prior:
            continue
        updated[prior_i] = replace(prior, columns=[replace(column, duplicate_scope=scope, sample_size=size,
                                                          sample_size_evidence=(appendix.page, str(size)))
                                                   for column in prior.columns])
        updated[i] = replace(appendix, config_name=prior.rows[0][0], columns=[
            replace(column, duplicate_scope=scope, dataset=prior.columns[0].dataset) for column in appendix.columns
        ])
    return updated


def method_abbreviation(candidate: str, full: str) -> bool:
    left, right = candidate.casefold().split(), full.casefold().split()
    return len(left) == len(right) and all(
        short == long or (short.endswith(".") and len(short) > 2 and long.startswith(short[:-1]))
        for short, long in zip(left, right)
    )
