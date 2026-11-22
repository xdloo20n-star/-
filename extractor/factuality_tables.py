"""Read factuality breakdowns and ablations with their printed units."""

from dataclasses import replace
import re

from pdf_pages import Page
from rag_tables import CAPTION, Column, Header, Table, column_boundaries
from retrieval_tables import source_statement


BREAKDOWN = re.compile(r"\bBreakdown of factuality benchmark results\b", re.I)
ABLATION = re.compile(r"\bAblation study using the ([\w-]+) dataset\b", re.I)
PROMPTS = re.compile(r"\bExperiments to study the sensitivity of RAG\b", re.I)
CLOSED_BOOK = re.compile(r"\bClosed-book factuality benchmark results\b", re.I)
MAIN_COMPARISON = re.compile(r"\bComparison of RAG QA factuality scores\b", re.I)
SAME_BREAKDOWN = re.compile(r"\bTable\s+\d+\s+presents the breakdown results for the base-?\s*line and proposed method\b", re.I)
MACRO_AVERAGE = re.compile(r"\b(?:overall )?macro(?:-\s*|\s+)?average(?:d)? (?:gain|improvement)\b", re.I)
MAIN_MODEL = re.compile(r"\bWe used (Llama-[\w.-]+)\s*\([^)]*\)\s*as the base model across all our ex-?\s*periments\b", re.I)
PROMPT_MODEL = re.compile(r"\bused (llama-[\w.-]+)\s*\([^)]*\)\s*as the generative model\b", re.I)
PERCENT_VALUE = re.compile(r"[−+-]?\d+(?:\.\d+)?%")
DECIMAL = re.compile(r"[−+-]?\d+(?:\.\d+)?")
BREAKDOWN_HEADINGS = ("Correct", "Hallucination", "Refusal", "Unverified", "Factuality Score")
BREAKDOWN_METRICS = ("Accuracy", "Hallucination rate", "Missing response rate", "Unverified response rate", "Factuality")
CLOSED_HEADINGS = ("Acc (Rec)", "Miss.", "Hall.", "Fac.")
CLOSED_METRICS = ("Accuracy", "Missing response rate", "Hallucination rate", "Factuality")
MAIN_SCOPE = "factuality benchmark breakdown"


def read_factuality_tables(pages: list[Page], tables: list[Table]) -> list[Table]:
    occupied = {(table.page.number, table.label.casefold()) for table in tables}
    comparison = next((table for table in tables if MAIN_COMPARISON.search(table.caption)
                       and len(table.rows) >= 2 and all(c.header.name == "Factuality" for c in table.columns)), None)
    linked = comparison is not None and source_statement(pages, SAME_BREAKDOWN) is not None
    main_model = source_statement(pages, MAIN_MODEL) if comparison else None
    if comparison and linked:
        avg_source = source_statement([comparison.page], MACRO_AVERAGE)
        dataset_count = sum(column.dataset is not None for column in comparison.columns)
        columns = [replace(column, duplicate_scope=MAIN_SCOPE if column.dataset else None,
                           generator=main_model[1][1] if main_model else None,
                           generator_evidence=(main_model[0], main_model[1][1]) if main_model else None)
                   for column in comparison.columns]
        if avg_source and dataset_count:
            columns = [replace(column, aggregation=f"macro mean across {dataset_count} benchmarks",
                               aggregation_evidence=(avg_source[0], avg_source[1][0]),
                               dataset=f"Macro-average over {dataset_count} RAG benchmarks")
                       if column.dataset is None and column.aggregation == "mean" else column for column in columns]
        tables = [replace(table, columns=columns) if table is comparison else table for table in tables]
    methods = {"Baseline": "Baseline", "This Work": "This Work"}
    if comparison:
        proposed = next((name for name, _ in comparison.rows if "".join(name.casefold().split()) == "prismrag"), None)
        if proposed:
            methods["This Work"] = proposed
    detail = []
    ablations = []
    for page in pages:
        if page.verification != "text_match":
            continue
        lines = page.text.splitlines()
        for index, line in enumerate(lines):
            caption_match = CAPTION.match(line)
            if not caption_match or (page.number, caption_match[1].casefold()) in occupied:
                continue
            label = caption_match[1]
            caption = " ".join(line.split())
            if BREAKDOWN.search(caption):
                detail.extend(breakdown_rows(page, lines, index, label, caption, methods, main_model, linked))
            elif CLOSED_BOOK.search(caption):
                detail.extend(closed_book_rows(page, lines, index, label, caption, methods, main_model))
            elif match := ABLATION.search(caption):
                ablations.extend(ablation_rows(page, lines, index, label, caption, match[1], main_model,
                                               linked, methods["This Work"]))
            elif PROMPTS.search(caption) and (prompt_model := source_statement([page], PROMPT_MODEL)):
                ablations.extend(ablation_rows(page, lines, index, label, caption, "Web Pages (C)", prompt_model,
                                               False, methods["This Work"], prompts=True))
    # The breakdown is the detailed source for values also reported in the ablation summary.
    return [*tables, *detail, *ablations]


def headings_on(line: str, labels: tuple[str, ...]) -> list[float] | None:
    pattern = r"^\s*" + r"\s{2,}".join(f"({re.escape(label)})" for label in labels) + r"(?:\s{2,}.*)?$"
    match = re.fullmatch(pattern, line)
    if not match:
        return None
    return [(match.start(i) + match.end(i)) / 2 for i in range(1, len(labels) + 1)]


def result_columns(labels, names, centers, dataset, model, scope=None, scale=None):
    return [Column(Header(printed, name, center, None, None), dataset,
                   generator=model[1][1] if model else None,
                   generator_evidence=(model[0], model[1][1]) if model else None,
                   direction="higher_is_better" if name in {"Accuracy", "Factuality"} else "lower_is_better",
                   source_label=printed, duplicate_scope=scope, reported_scale=scale)
            for printed, name, center in zip(labels, names, centers, strict=True)]


def located_values(line: str, bounds: list[float], pattern: re.Pattern[str]) -> tuple[str, list[str]] | None:
    values = [token for token in re.finditer(r"\S+", line) if bounds[0] <= (token.start() + token.end()) / 2 < bounds[-1]]
    if len(values) != len(bounds) - 1 or any(not pattern.fullmatch(token[0]) or not left <= (token.start() + token.end()) / 2 < right
                                             for token, left, right in zip(values, bounds[:-1], bounds[1:], strict=True)):
        return None
    return line[:values[0].start()].strip(), [value[0] for value in values]


def breakdown_rows(page, lines, index, label, caption, methods, model, linked):
    for start in range(index - 1, max(-1, index - 35), -1):
        centers = headings_on(lines[start], BREAKDOWN_HEADINGS)
        if centers:
            break
    else:
        return []
    bounds = column_boundaries(centers)
    rows = []
    markers = []
    for row_index, line in enumerate(lines[start + 1:index], start=start + 1):
        if not line.strip():
            continue
        if line.strip() in methods:
            markers.append((len(rows), line.strip()))
            continue
        located = located_values(line, bounds, PERCENT_VALUE)
        if not located:
            return []
        row, values = located
        parts = re.split(r"\s{2,}", row)
        if len(parts) > 2 or (len(parts) == 2 and parts[0] not in methods):
            return []
        if len(parts) == 2:
            markers.append((len(rows), parts[0]))
        rows.append((row_index, parts[-1], values))
    width = len(rows) // 2
    if width < 2 or 2 * width != len(rows) or [r[1] for r in rows[:width]] != [r[1] for r in rows[width:]]:
        return []
    tables = []
    for offset, method in enumerate(methods):
        group = rows[offset * width:(offset + 1) * width]
        if [marker for position, marker in markers if offset * width <= position < (offset + 1) * width] != [method]:
            return []
        for row_index, dataset, values in group:
            columns = result_columns(BREAKDOWN_HEADINGS, BREAKDOWN_METRICS, centers, dataset, model)
            if linked:
                columns = [replace(column, duplicate_scope=MAIN_SCOPE) for column in columns]
            tables.append(Table(page, label, caption, row_index, columns, [(dataset, values)],
                                row_section=method, config_name=methods[method]))
    return tables


def closed_book_rows(page, lines, index, label, caption, methods, model):
    tables = []
    for start in range(max(0, index - 33), index - 3):
        dataset = re.fullmatch(r"\s*Model\s{2,}([\w-]+)a?\s+\(closed-book\)\s*", lines[start])
        if not dataset or not (centers := headings_on(lines[start + 1], CLOSED_HEADINGS)):
            continue
        bounds = column_boundaries(centers)
        dataset_name = "CRAG" if dataset[1] == "CRAGa" else dataset[1]
        columns = result_columns(CLOSED_HEADINGS, CLOSED_METRICS, centers, dataset_name, model, scale="percent")
        rows = [located_values(line, bounds, DECIMAL) for line in lines[start + 2:start + 4]]
        if any(row is None for row in rows) or [row[0] for row in rows] != list(methods):
            continue
        for offset, (method, values) in enumerate(rows, start=start + 2):
            tables.append(Table(page, label, caption, offset, columns, [(method, values)],
                                row_section=columns[0].dataset, config_name=f"{methods[method]} closed-book"))
    return tables


def ablation_rows(page, lines, index, label, caption, dataset, model, linked, proposed, *, prompts=False):
    labels = ("Accuracy", "Hallucination", "Missing", "Factuality") if prompts else (
        "Accurate", "Hallucinated", "Missing", "Factuality")
    for start in range(index - 1, max(-1, index - 11), -1):
        heading = re.match(r"\s*(?:Method|Instruction)\s{2,}", lines[start])
        if heading and (centers := headings_on(lines[start][heading.end():], labels)):
            centers = [center + heading.end() for center in centers]
            break
    else:
        return []
    bounds = column_boundaries(centers)
    names = ("Accuracy", "Hallucination rate", "Missing response rate", "Factuality")
    tables = []
    for offset, line in enumerate(lines[start + 1:index], start=start + 1):
        parsed = located_values(line, bounds, PERCENT_VALUE)
        if not parsed:
            continue
        row, values = parsed
        if prompts and re.fullmatch(r"Prompt-[A-Z]", row):
            name = f"{proposed} prompt {row[-1]}"
        elif not prompts and re.fullmatch(r"Baseline|P\s*RISM\s*RAG|-[\w-]+", row):
            name = row
        else:
            continue
        columns = result_columns(labels, names, centers, dataset, model,
                                 MAIN_SCOPE if linked and name in {"Baseline", proposed} else None)
        tables.append(Table(page, label, caption, offset, columns, [(row, values)], config_name=name))
    return tables
