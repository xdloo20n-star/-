"""Read watermark benchmark grids with separate metric and condition blocks."""

from dataclasses import replace
import re

from pdf_pages import Page
from qa_comparison import find_native
from rag_tables import Column, Header, NATIVE_CAPTION, NUMBER, Table


EFFECTIVENESS_CAPTION = re.compile(r"\bresults? of effectiveness evaluation across datasets, LLMs, and baselines\b", re.I)
GROUPS = (
    ("Validity Metrics", "LLM", ("WRR", "WTRR", "FHR"), 3),
    ("Quality Metrics", "Scheme", ("PPLavg", "KCR", "RSavg"), 4),
    ("Task Metrics", "Scheme", ("CDPA", "CIRA", "WDPV"), 4),
)
GROUP_HEADING = re.compile(r"\s*Validity\s+LLM\s+Quality\s+Scheme\s+Task\s+Scheme\s*")
METRICS_HEADING = re.compile(r"\s*Metrics\s+Metrics\s+Metrics\s*")
SCORED_ROW = re.compile(r"\s*(?:(?P<dataset>[\w-]+)\s+)?" + r"\s+".join(
    rf"(?P<metric_{i}>{'|'.join(labels)})\s*(?P<direction_{i}>[↑↓])?\s+"
    rf"(?P<values_{i}>(?:{NUMBER.pattern}\s+){{{width - 1}}}{NUMBER.pattern})"
    for i, (_, _, labels, width) in enumerate(GROUPS)
))
DIRECTIONS = {"↑": "higher_is_better", "↓": "lower_is_better"}
RETRIEVAL_PROTOCOL = re.compile(
    r"\bFor\s+RAG\s+systems,\s+unless\s+otherwise\s+specified,\s+top-(\d+)\s+text\s+"
    r"was\s+retrieved\s+from\s+the\s+knowledge\s+base\s+to\s+evaluate\s+the\s+retrievability\s+of\s+watermark\s+texts\b",
    re.I,
)
MODEL_PROTOCOL = re.compile(
    r"\bWe\s+utilize\s+three\s+mainstream\s+large\s+language\s+models\s+in\s+experiments,\s+"
    r"including\s+two\s+advanced\s+black-box\s+LLMs\s+\("
    r"(?P<GPT>GPT-[\d.]+)\s+\([^()]+\)\s+and\s+"
    r"(?P<Gemini>Gemini-[\d.]+(?:-\s*|(?=Flash))Flash)\s+\([^()]+\)\)\s+"
    r"and\s+a\s+white-box\s+LLM\s+\((?P<Qwen>Qwen[\w.-]+)\s+\([^()]+\)\)", re.I,
)
WATERMARK_BASELINE = re.compile(r"\bwe\s+compare\s+[\w-]+\s+with\s+two\s+formally\s+accepted\s+approaches,"
                                r"\s+[\w-]+\s+\([^()]{1,110}\)\s+and\s+RAG-WM\b", re.I)
PPL_CAPTION = re.compile(r"\baverage PPL of original datasets\b", re.I)
TIME_CAPTION = re.compile(r"\btime overhead of watermarking\b", re.I)
ORIGINAL_DATA = re.compile(r"\ba\s+non-watermarked\s+RAG\s+\((?P<name>[\w-]+)\)\s+as\s+baselines\b", re.I)
PREFIX_TEXT = re.compile(r"^\s*(?P<heading>Prefix/Text\s*\((?P<unit>hours?)\))\s+(?P<values>\S.*)$", re.I)
INJECTION_CAPTION = re.compile(r"\bknowledge injection attack\b", re.I)


def read_watermark_tables(pages: list[Page], existing: list[Table]) -> list[Table]:
    native = [page for page in pages if page.verification == "text_match"]
    tables = []
    occupied = {(table.page.number, table.label.casefold()) for table in existing}
    for page in native:
        lines = page.text.splitlines()
        panels = []
        for start, line in enumerate(lines):
            caption = NATIVE_CAPTION.match(line)
            if not caption or (page.number, caption[1].casefold()) in occupied or not EFFECTIVENESS_CAPTION.search(line):
                continue
            panels.extend(read_effectiveness(page, lines, start, caption[1], native))
        tables.extend(panels)
        if panels:
            tables.extend(read_summary_tables(page, lines, native, panels, occupied))
    datasets = {column.dataset for table in tables if table.family == "watermark_metrics"
                for column in table.columns if column.dataset}
    if datasets:
        from watermark_appendix import read_watermark_appendix

        tables.extend(read_injection_tables(native, datasets, occupied))
        occupied.update((table.page.number, table.label.casefold()) for table in tables)
        tables.extend(read_watermark_appendix(native, datasets, occupied))
    return tables


def read_effectiveness(page: Page, lines: list[str], start: int, label: str, native: list[Page]) -> list[Table]:
    header = next((i for i in range(start + 1, min(start + 4, len(lines))) if lines[i].strip()), len(lines))
    if header + 4 >= len(lines) or not GROUP_HEADING.fullmatch(lines[header]):
        return []
    if lines[header + 1].strip() != "Dataset" or not METRICS_HEADING.fullmatch(lines[header + 2]):
        return []
    conditions = lines[header + 3].split()
    if len(conditions) != sum(group[3] for group in GROUPS) or not all(
            re.fullmatch(r"[A-Za-z][\w.-]*(?:\(\d+\))?", name) for name in conditions):
        return []
    groups = dataset_rows(lines[header + 4:])
    if len(groups) < 2:
        return []

    method = find_native(re.compile(
        rf"\b{re.escape(label)}\s+demonstrates?\s+the\s+robust\s+watermark\s+verification\s+of\s+([\w-]+)", re.I,
    ), native)
    models = find_native(MODEL_PROTOCOL, native)
    cutoff = find_native(RETRIEVAL_PROTOCOL, native)
    baseline = find_native(WATERMARK_BASELINE, native)
    tables = []
    offset = 0
    for i, (category, heading, _, width) in enumerate(GROUPS):
        names = conditions[offset:offset + width]
        offset += width
        if len(set(names)) != width:
            return []
        if heading == "LLM" and not method:
            continue
        for position, condition in enumerate(names):
            generator = normalize_model(condition, models) if heading == "LLM" else None
            generator_evidence = (models[0], models[1][0]) if models and generator and generator != condition else None
            config = method[1][1] if heading == "LLM" else ("RAG-WM" if condition == "RAGWM" and baseline else condition)
            columns, values = [], []
            for dataset, rows in groups:
                for row in rows:
                    metric = row[f"metric_{i}"]
                    columns.append(Column(
                        Header(metric, metric, row.start(f"metric_{i}"), None, None), dataset, generator=generator,
                        original_row=f"{dataset} / {metric}", generator_locator=condition,
                        generator_evidence=generator_evidence,
                        direction=DIRECTIONS.get(row[f"direction_{i}"]),
                        source_label=f"{category} / {heading} / {condition}",
                    ))
                    values.append(row[f"values_{i}"].split()[position])
            top_k = int(cutoff[1][1]) if cutoff and heading == "LLM" else None
            top_k_evidence = (cutoff[0], cutoff[1][0]) if top_k is not None else None
            tables.append(Table(page, label, lines[start].strip(), header + 3, columns, [(condition, values)],
                                family="watermark_metrics", config_name=config, top_k=top_k,
                                top_k_evidence=top_k_evidence))
    return tables


def normalize_model(condition: str, models: tuple[Page, re.Match[str]] | None) -> str:
    if not models or condition not in models[1].groupdict():
        return condition
    model = "".join(models[1][condition].split())
    return re.sub(r"(?<=\d)Flash$", "-Flash", model) if condition == "Gemini" else model


def dataset_rows(lines: list[str]) -> list[tuple[str, list[re.Match[str]]]]:
    groups, rows, parts = [], [], []
    for line in lines:
        if not line.strip():
            break
        match = SCORED_ROW.fullmatch(line.rstrip())
        if match is None:
            part = re.fullmatch(r"\s*([\w-]+)", line.rstrip())
            if not rows or part is None or part.end() >= min(row.start("metric_0") for row in rows):
                return []
            parts.append(part[1])
            continue
        phase = len(rows)
        if any(match[f"metric_{i}"] != labels[phase] for i, (_, _, labels, _) in enumerate(GROUPS)):
            return []
        if match["dataset"]:
            parts.append(match["dataset"])
        rows.append(match)
        if len(rows) != len(GROUPS[0][2]):
            continue
        if len(parts) != 1 and not (len(parts) == 2 and parts[0].endswith("-")):
            return []
        dataset = "".join(parts)
        if any(dataset.casefold() == name.casefold() for name, _ in groups):
            return []
        groups.append((dataset, rows))
        rows, parts = [], []
    return [] if rows else groups


def read_summary_tables(page: Page, lines: list[str], native: list[Page], panels: list[Table],
                        occupied: set[tuple[int, str]]) -> list[Table]:
    datasets = {column.dataset for column in panels[0].columns if column.dataset}
    system = next((table.config_name for table in panels if table.columns[0].generator), None)
    original = find_native(ORIGINAL_DATA, native)
    tables = []
    for start, line in enumerate(lines):
        caption = NATIVE_CAPTION.match(line)
        if not caption or (page.number, caption[1].casefold()) in occupied:
            continue
        label = caption[1]
        if not (ppl := PPL_CAPTION.search(line)) and not TIME_CAPTION.search(line):
            continue
        header = next((i for i in range(start + 1, min(start + 4, len(lines))) if lines[i].strip()), len(lines))
        if header + 1 >= len(lines) or not lines[header + 1].strip():
            continue
        heading, *displayed = lines[header].split()
        if heading != "Metrics" or len(displayed) != len(datasets):
            continue
        resolved = [resolve_dataset(name, datasets) for name in displayed]
        if None in resolved or len(set(resolved)) != len(datasets):
            continue
        if ppl:
            if not original:
                continue
            row, *values = lines[header + 1].split()
            if row != "PPLavg" or len(values) != len(displayed) or not all(NUMBER.fullmatch(raw) for raw in values):
                continue
            columns = [Column(Header(row, row, 0, None, None), dataset, source_label=shown)
                       for shown, dataset in zip(displayed, resolved, strict=True)]
            tables.append(Table(page, label, line.strip(), header, columns, [(row, values)],
                                config_name=original[1]["name"]))
        elif system and (match := PREFIX_TEXT.fullmatch(lines[header + 1].rstrip())):
            pairs = [raw.split("/") for raw in match["values"].split()]
            if len(pairs) != len(displayed) or any(
                    len(pair) != 2 or not all(NUMBER.fullmatch(raw) for raw in pair) for pair in pairs):
                continue
            proof = find_native(re.compile(
                r"\bfor\s+watermark\s+prefix\s+construction\s+and\s+linearly\s+generating\s+"
                rf"(?:\d+\s+)?watermark\s+texts\s+is\s+shown\s+in\s+{re.escape(label)}\b", re.I,
            ), native)
            if not proof:
                continue
            metrics = (("Prefix", "Prefix construction time"), ("Text", "Watermark text generation time"))
            columns = [Column(Header(key, metric, 0, None, match["unit"]), dataset,
                              direction="lower_is_better", name_evidence=(proof[0], proof[1][0]),
                              source_label=f"{shown} / {key}")
                       for shown, dataset in zip(displayed, resolved, strict=True) for key, metric in metrics]
            values = [raw for pair in pairs for raw in pair]
            tables.append(Table(page, label, line.strip(), header, columns, [(match["heading"], values)],
                                config_name=system))
    return tables


def resolve_dataset(name: str, datasets: set[str]) -> str | None:
    exact = next((dataset for dataset in datasets if dataset.casefold() == name.casefold()), None)
    if exact:
        return exact
    suffixes = [dataset for dataset in datasets if dataset.casefold().endswith("-" + name.casefold())]
    return suffixes[0] if len(suffixes) == 1 else None


def read_injection_tables(pages: list[Page], datasets: set[str], occupied: set[tuple[int, str]]) -> list[Table]:
    tables = []
    cutoff = find_native(RETRIEVAL_PROTOCOL, pages)
    top_k = int(cutoff[1][1]) if cutoff else None
    top_k_evidence = (cutoff[0], cutoff[1][0]) if cutoff else None
    for page in pages:
        lines = page.text.splitlines()
        for start, line in enumerate(lines):
            for caption in NATIVE_CAPTION.finditer(line):
                label = caption[1]
                if (page.number, label.casefold()) in occupied or not INJECTION_CAPTION.match(line[caption.end():].lstrip()):
                    continue
                proof = find_native(re.compile(
                    rf"\b{re.escape(label)}\s+shows?\s+that\s+the\s+WRRs\s+of\s+([\w-]+)\s+across\b", re.I,
                ), [page])
                if not proof:
                    continue
                grid = read_injection_grid(lines, start, caption.start(1), datasets)
                if not grid:
                    continue
                header, columns, rows = grid
                columns = [replace(column, name_evidence=(proof[0], proof[1][0])) for column in columns]
                for rate, values in rows:
                    title = f"{line[caption.start(1):].strip()}\nInjection Rate {rate}"
                    tables.append(Table(page, label, title, header, columns, [(rate, values)],
                                        row_section="Injection Rate", config_name=proof[1][1],
                                        top_k=top_k, top_k_evidence=top_k_evidence))
    return tables


def read_injection_grid(lines: list[str], start: int, caption_at: int, datasets: set[str]
                        ) -> tuple[int, list[Column], list[tuple[str, list[str]]]] | None:
    for index in range(start + 1, min(start + 6, len(lines) - 3)):
        heading = re.search(r"\bInjection\s*$", lines[index])
        if heading and abs(heading.start() - caption_at) <= 24 and lines[index + 2][heading.start():].strip() == "Rate":
            break
    else:
        return None
    left = heading.start()
    shown = lines[index + 1][left:].split()
    resolved = [resolve_dataset(name, datasets) for name in shown]
    if len(shown) != len(datasets) or None in resolved or len(set(resolved)) != len(datasets):
        return None
    columns = [Column(Header("WRR", "WRR", 0, None, None), dataset, source_label=name)
               for name, dataset in zip(shown, resolved, strict=True)]
    rows, last_rate = [], -1
    for line in lines[index + 3:]:
        cells = line[left:].split()
        if not cells or not re.fullmatch(r"\d{1,3}%", cells[0]):
            break
        rate, *values = cells
        percent = int(rate[:-1])
        if percent <= last_rate or percent > 100 or len(values) != len(shown) or not all(
                NUMBER.fullmatch(value) for value in values):
            return None
        rows.append((rate, values))
        last_rate = percent
    return (index + 1, columns, rows) if len(rows) >= 2 else None
