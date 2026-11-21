"""Read native QA comparison grids with repeated effectiveness and cost columns."""

from dataclasses import replace
import re

from pdf_pages import Page
from rag_tables import (
    Column, HEADER_CELLS, Header, NATIVE_CAPTION, Table, TOKEN, aligned_label,
    aligned_values, column_boundaries,
)


PREFIX = re.compile(r"^\s*(?P<data>Data\s+)?Types\s+(?P<method>Methods?)\s{2,}", re.I)
METRICS = ("EM", "F1", "Acc", "Step", "Time")
CAPTION_MODEL = re.compile(r"\b([\w.-]+(?: \([^()]+\))?) as (?:the )?(?:LLM|model)\b", re.I)
MODELS_CAPTION = re.compile(r"\b(?:different|multiple) (?:LLMs?|models?)\b", re.I)
AVERAGED_CAPTION = re.compile(r"\baverage[ds]?\b", re.I)
MARKED_METHOD = re.compile(r"[*∗†‡]$")
COMPARISON_TITLE = r"[\w.-]+(?:\s+\([^()]+\))?"
CAPTION_COMPARATORS = re.compile(
    rf"\b(?P<marked>(?P<method>[\w.-]+)[*∗†‡])\s+is\s+trained\s+with\s+a\s+different\s+base\s+LLM,\s+"
    rf"namely\s+(?P<family>[\w.-]+)\s+\([^()]{{1,120}}\);\s+therefore,\s+we\s+compare\s+"
    rf"the\s+results\s+of\s+(?P<reference>{COMPARISON_TITLE})\s+with\s+the\s+results\s+from\s+"
    rf"(?P=method)\s+with\s+(?P<first>(?P=family)\s+\([^()]+\))\s+and\s+the\s+results\s+of\s+others\s+"
    rf"with\s+the\s+results\s+from\s+(?P=method)\s+with\s+(?P<others>(?P=family)\s+\([^()]+\))\.", re.I,
)
EFFICIENCY = re.compile(
    r"\bfor\s+efficiency,\s+we\s+measure\s+the\s+"
    r"(?P<Step>number\s+of\s+retrieval\s*-?\s*and\s*-\s*generate\s+steps)\s+and\s+the\s+"
    r"(?P<Time>average\s+time\s+for\s+answering\s+each\s+query\s+relative\s+to\s+the\s+one-step\s+approach)\b",
    re.I,
)
EFFECTIVENESS = re.compile(
    r"\bfor\s+effectiveness,\s+we\s+use\s+F1,\s+EM,\s+and\s+Accuracy\s*\(Acc\),\s+"
    r"following\s+the\s+standard\s+evaluation\s+protocol\b", re.I,
)
GENERATOR_METHOD = re.compile(
    r"\bregarding\s+the\s+LLMs\s+that\s+are\s+used\s+to\s+generate\s+answers,\s+we\s+use\s+the\s+"
    r"(?P<series>[\w.-]+)\s+series\s+models(?:\s+\([^()]{1,120}\))?\s+of\s+"
    r"(?P<first>\w+)\s+with\s+(?P<first_size>\d+[BM])\s+parameters\s+and\s+"
    r"(?P<second>\w+)\s+with\s+(?P<second_size>\d+[BM])\s+parameters,\s+and\s+the\s+"
    r"(?P<family>[\w.-]+)\s+model\s+\((?P<identifier>[\w.-]+)\)\.", re.I,
)
# The linked evaluator totals per-query scores and divides by the question count.
# https://github.com/starsuzi/Adaptive-RAG/blob/main/evaluate_final_acc.py#L93-L119
# https://github.com/starsuzi/Adaptive-RAG/blob/main/metrics/squad_answer_em_f1.py
# https://github.com/starsuzi/Adaptive-RAG/blob/main/metrics/drop_answer_em_f1.py
EVALUATOR = re.compile(r"\bcode\s+is\s+available\s+at:\s*https://\s*github\.com/starsuzi/Adaptive-RAG(?=[.\s]|$)", re.I)
AVERAGE_SCOPE = re.compile(
    r"\b(?P<aggregate>Table\s+\d+(?:\.\d+)*)\s+shows?\s+(?:our\s+)?(?:main\s+)?results\s+"
    r"averaged\s+over\s+all\s+considered\s+datasets\b[^.!?]{0,450}[.!?]\s+"
    rf"(?:in\s+addition,\s*)?we\s+report\s+the\s+more\s+granular\s+results\s+with\s+"
    rf"(?P<detail_model>{COMPARISON_TITLE})\s+"
    r"\bon\s+each\s+of\s+the\s+\S+\s+and\s+\S+\s+datasets\s+in\s+"
    r"(?P<detail>Table\s+\d+(?:\.\d+)*)\b\.?", re.I,
)
TABLE_REFERENCE = re.compile(r"\bTable\s+\d+(?:\.\d+)*\b", re.I)
APPENDIX_LINK = re.compile(
    rf"\s*\(and\s+more\s+with\s+different\s+LLMs?\s+in\s+"
    rf"(?P<tables>{TABLE_REFERENCE.pattern}(?:\s+and\s+{TABLE_REFERENCE.pattern})+)\s+of\s+Appendix\)", re.I,
)
DATASET_NAME = r"[\w./-]+(?:\s+[\w./-]+)*"
DATASET_CITATION = rf"\d+\)\s+{DATASET_NAME}\s+\([^()]{{1,120}}\)"
DATASET_LIST = rf"{DATASET_CITATION}(?:,\s*(?:and\s+)?{DATASET_CITATION})+"
DATASET_METHOD = re.compile(
    rf"\bnamely\s+(?P<single>{DATASET_LIST})\.\s+multi-hop\s+QA\b[^!?]{{0,400}}?"
    rf"\bnamely\s+(?P<multi>{DATASET_LIST})\.", re.I,
)
DATASET_ITEMS = re.compile(rf"\d+\)\s+({DATASET_NAME})\s+\(")
COUNT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")
EFFICIENCY_NAMES = {"Step": "Retrieval-and-generation steps per query", "Time": "Relative time per query"}
SINGLE_STEP = "Single-step Approach"
ONE = re.compile(r"1(?:[.,]0+)?")
FRACTIONAL_COUNT = re.compile(r"\d+[.,]\d*[1-9]\d*")


def read_qa_comparisons(page: Page, lines: list[str], occupied: set[str]) -> list[Table]:
    if page.verification != "text_match":
        return []
    tables = []
    for start, line in enumerate(lines):
        match = NATIVE_CAPTION.match(line)
        if match is None or match[1].casefold() in occupied:
            continue
        for index in range(start + 2, min(start + 9, len(lines))):
            if NATIVE_CAPTION.match(lines[index - 1]):
                break
            if not PREFIX.match(lines[index]):
                continue
            caption = " ".join(" ".join(lines[start:index - 1]).split())
            while index < len(lines):
                table, end = read_panel(page, lines, index, match[1], caption)
                if table is None:
                    break
                tables.append(table)
                # A second panel may follow with its own group and metric headings.
                for group_at in range(end, min(end + 3, len(lines))):
                    if lines[group_at].strip():
                        break
                else:
                    break
                index = group_at + 1
            break
    return tables


def read_panel(page: Page, lines: list[str], index: int, label: str, caption: str) -> tuple[Table | None, int]:
    prefix = PREFIX.match(lines[index])
    if prefix is None:
        return None, index
    metrics = list(TOKEN.finditer(lines[index], prefix.end()))
    if len(metrics) % len(METRICS):
        return None, index
    group_count = len(metrics) // len(METRICS)
    groups = list(HEADER_CELLS.finditer(lines[index - 1].rstrip()))
    if not 2 <= group_count <= 6 or len(groups) != group_count:
        return None, index
    if tuple(metric[0] for metric in metrics) != METRICS * group_count:
        return None, index
    if len({group[0].casefold() for group in groups}) != group_count:
        return None, index

    by_dataset = bool(prefix["data"])
    model = CAPTION_MODEL.search(caption)
    if (by_dataset and not model) or (not by_dataset and not MODELS_CAPTION.search(caption)):
        return None, index
    aggregation = "reported average" if AVERAGED_CAPTION.search(caption) else None
    centers = [(metric.start() + metric.end()) / 2 for metric in metrics]
    columns = []
    for i, group in enumerate(groups):
        start = i * len(METRICS)
        pivot = (group.start() + group.end()) / 2
        if not centers[start] < pivot < centers[start + len(METRICS) - 1]:
            return None, index
        dataset = group[0] if by_dataset else None
        generator = model[1] if by_dataset else group[0]
        columns.extend(Column(Header(metric[0], "Accuracy" if metric[0] == "Acc" else metric[0], center, None, None),
                              dataset, aggregation, generator)
                       for metric, center in zip(metrics[start:start + len(METRICS)],
                                                 centers[start:start + len(METRICS)], strict=True))

    rows, end = read_panel_rows(lines, index + 1, prefix.start("method"), column_boundaries(centers))
    if sum(not MARKED_METHOD.search(method) for method, _ in rows) < 2:
        return None, index
    return Table(page, label, caption, index, columns, rows, family="qa_comparison"), end


def enrich_qa_comparisons(pages: list[Page], tables: list[Table]) -> list[Table]:
    qa_tables = [table for table in tables if table.family == "qa_comparison"]
    if not qa_tables:
        return tables
    native = [page for page in pages if page.verification == "text_match"]
    efficiency = find_native(EFFICIENCY, native)
    effectiveness = find_native(EFFECTIVENESS, native)
    linked_scores = effectiveness if find_native(EVALUATOR, native) else None
    averages = benchmark_averages(qa_tables, native)
    exceptions = generator_exceptions(qa_tables, native)
    documented_models = documented_generators(native)
    enriched = []
    for table in tables:
        if table.family != "qa_comparison":
            enriched.append(table)
            continue
        average = averages.get(" ".join(table.label.casefold().split()))
        baselines = [values for row, values in table.rows if row.casefold() == SINGLE_STEP.casefold()]
        baseline = baselines[0] if len(baselines) == 1 else None
        columns = []
        for i, column in enumerate(table.columns):
            observed_steps = column.header.label == "Step" and any(
                FRACTIONAL_COUNT.fullmatch(values[i]) for _, values in table.rows
            )
            reference_time = baseline is not None and ONE.fullmatch(baseline[i]) is not None
            columns.append(enrich_qa_column(column, efficiency, linked_scores, average,
                                            observed_steps, reference_time))
        rows = [(row, values) for row, values in table.rows if not MARKED_METHOD.search(row)]
        enriched.append(replace(table, columns=[canonical_generator(column, documented_models) for column in columns],
                                rows=rows))
        for row, values in table.rows:
            models = exceptions.get((" ".join(table.label.casefold().split()), row.casefold()))
            if not models or any(column.generator not in models for column in columns):
                continue
            reference_page, reference = models[columns[0].generator][1]
            # The caption identifies which comparison columns reuse the same named model.
            scope = (reference_page.number, reference) if average or all(c.dataset for c in columns) else None
            marked_columns = [replace(column, generator=models[column.generator][0], generator_locator=column.generator,
                                      generator_evidence=models[column.generator][1],
                                      duplicate_scope=scope if column.header.label != "Time" else None)
                              for column in columns]
            enriched.append(replace(table, columns=marked_columns, rows=[(row, values)], config_name=row[:-1]))
    return enriched


def find_native(pattern: re.Pattern[str], pages: list[Page]) -> tuple[Page, re.Match[str]] | None:
    for page in pages:
        for text in (page.reading_text, page.text):
            if match := pattern.search(text):
                return page, match
    return None


def documented_generators(pages: list[Page]) -> dict[str, tuple[str, tuple[Page, str]]]:
    source = find_native(GENERATOR_METHOD, pages)
    if not source:
        return {}
    page, match = source
    evidence = page, match[0]
    models = {}
    for position in ("first", "second"):
        model = f"{match['series']}-{match[position]}"
        size = match[f"{position}_size"]
        models[f"{model} ({size})".casefold()] = f"{model} {size}", evidence
    family_prefix = f"{match['family'].casefold()}-"
    identifier = match["identifier"].casefold()
    if identifier.startswith(family_prefix):
        edition = identifier.removeprefix(family_prefix).split("-")[0]
        if edition:
            models[f"{match['family']} ({edition})".casefold()] = match["identifier"], evidence
    return models


def canonical_generator(column: Column, models: dict[str, tuple[str, tuple[Page, str]]]) -> Column:
    if column.generator and (documented := models.get(column.generator.casefold())):
        model, evidence = documented
        return replace(column, generator=model, generator_locator=column.generator, generator_evidence=evidence)
    return column


def generator_exceptions(tables: list[Table], pages: list[Page]) -> dict[
        tuple[str, str], dict[str, tuple[str, tuple[Page, str]]]]:
    exceptions = {}
    for table in tables:
        note = CAPTION_COMPARATORS.search(table.caption)
        if not note or any(column.dataset for column in table.columns):
            continue
        comparators = {column.generator for column in table.columns if column.generator}
        if note["reference"] not in comparators:
            continue
        literal = re.compile(r"\s+".join(re.escape(word) for word in note[0].split()))
        source = find_native(literal, [table.page])
        if not source:
            continue
        page, match = source
        marked = note["marked"].casefold()
        aggregate = " ".join(table.label.casefold().split())
        models = {name: (note["first"] if name == note["reference"] else note["others"], (page, match[0]))
                  for name in comparators}
        exceptions[aggregate, marked] = models
        for detail, allowed in linked_generators(pages, aggregate, comparators):
            exceptions[detail, marked] = {name: models[name] for name in allowed}
    return exceptions


def linked_generators(pages: list[Page], aggregate: str, comparators: set[str]):
    for page in pages:
        text = page.reading_text or page.text
        for match in AVERAGE_SCOPE.finditer(text):
            if " ".join(match["aggregate"].casefold().split()) != aggregate:
                continue
            detail_model = " ".join(match["detail_model"].casefold().split())
            detail_models = {name for name in comparators if detail_model in {name.casefold(), name.split(" (")[0].casefold()}}
            if len(detail_models) != 1:
                continue
            yield " ".join(match["detail"].casefold().split()), detail_models
            if appendix := APPENDIX_LINK.match(text, match.end()):
                for reference in TABLE_REFERENCE.findall(appendix["tables"]):
                    yield " ".join(reference.casefold().split()), comparators - detail_models


def benchmark_averages(tables: list[Table], pages: list[Page]) -> dict[str, tuple[str, tuple[Page, str]]]:
    methodology = find_native(DATASET_METHOD, pages)
    if not methodology:
        return {}
    _, lists = methodology
    named = {re.sub(r"\s+v\d+(?:\.\d+)*$", "", " ".join(name.casefold().split()))
             for group in ("single", "multi") for name in DATASET_ITEMS.findall(lists[group])}
    if not 2 <= len(named) < len(COUNT_WORDS):
        return {}

    averages = {}
    for page in pages:
        for match in AVERAGE_SCOPE.finditer(page.reading_text or page.text):
            detailed = " ".join(match["detail"].casefold().split())
            datasets = {" ".join(column.dataset.casefold().split()) for table in tables
                        if " ".join(table.label.casefold().split()) == detailed
                        for column in table.columns if column.dataset}
            if datasets == named:
                aggregate = " ".join(match["aggregate"].casefold().split())
                averages[aggregate] = f"mean across {COUNT_WORDS[len(datasets)]} datasets", (page, match[0])
    return averages


def enrich_qa_column(column: Column, efficiency: tuple[Page, re.Match[str]] | None,
                     linked_scores: tuple[Page, re.Match[str]] | None,
                     average: tuple[str, tuple[Page, str]] | None,
                     fractional_steps: bool, relative_reference: bool) -> Column:
    label = column.header.label
    if efficiency and label in EFFICIENCY_NAMES:
        page, match = efficiency
        column = replace(column, header=replace(column.header, name=EFFICIENCY_NAMES[label]),
                         direction="lower_is_better", name_evidence=(page, match[label]))
    if column.dataset is None and column.aggregation and average:
        aggregation, evidence = average
        column = replace(column, aggregation=aggregation, aggregation_evidence=evidence)
    if column.dataset and not column.aggregation:
        if label in {"EM", "F1", "Acc"} and linked_scores:
            page, match = linked_scores
            return replace(column, aggregation="mean across queries", aggregation_evidence=(page, match[0]))
        # Fractional entries under a per-query integer count require averaging.
        if label == "Step" and column.name_evidence and fractional_steps:
            return replace(column, aggregation="mean across queries", aggregation_evidence=column.name_evidence)
        if label == "Time" and column.name_evidence:
            column = replace(column, aggregation="mean across queries", aggregation_evidence=column.name_evidence)
    scope = column.dataset or (column.aggregation if average else None)
    if label == "Time" and column.name_evidence and relative_reference and column.generator and scope:
        variant = f"denominator {column.generator} / {scope} / {SINGLE_STEP}"
        column = replace(column, metric_variant=variant, metric_variant_evidence=column.name_evidence)
    return column


def read_panel_rows(lines: list[str], start: int, method_at: int,
                    boundaries: list[float]) -> tuple[list[tuple[str, list[str]]], int]:
    rows = []
    end = start
    for end in range(start, min(start + 32, len(lines))):
        line = lines[end].rstrip()
        if not line:
            break
        if len(line) < method_at and line[0].isspace():
            # The vertical center of a merged row label may have a line of its own.
            continue
        aligned = aligned_values(line, boundaries)
        if not aligned:
            break
        value_at, values = aligned
        method = aligned_label(line, method_at, value_at)
        if not method:
            break
        rows.append((method, values))
    else:
        end += 1
    return rows, end
