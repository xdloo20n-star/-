"""Keep neighboring retrieval and reasoning-ablation tables on their own sides."""

from dataclasses import replace
import re

from pdf_pages import Page
from rag_tables import GROUP, NATIVE_CAPTION, Column, Table, assign_groups, groups_for, headers, scan_rows
from retrieval_tables import source_statement


MAIN_COMPARISON = re.compile(r"\bcomparison of results from different baselines on the NQ and Huatuo test set\b", re.I)
ABLATION_CAPTION = re.compile(r"\bAblation results on reasoning content\b", re.I)
CONTEXT_CAPTION = re.compile(r"\bEvaluation on the retrieved contexts utilizing\b", re.I)
BASE_ROW = re.compile(r"Wiki|QA Pairs|Wiki\+QA Pairs|QA Pairs\+Wiki|Merged KB", re.I)
ABLATION_METRICS = ("EM", "F1", "ROUGE_1", "BLEU_1")
ANSWER_NAMES = {"precision": "Answer precision", "recall": "Answer recall"}
VARIANTS = {
    "EM": (re.compile(r"\bnormalized exact\s+match\b", re.I), "normalized exact match"),
    "F1": (re.compile(r"\bword-level F1(?:-score)?\b", re.I), "word-level"),
}
DUPLICATE_SCOPE = "missing-knowledge reasoning ablation with reasoning"


def read_reasoning_tables(pages: list[Page], tables: list[Table]) -> list[Table]:
    main = next((table for table in tables if MAIN_COMPARISON.search(table.caption)), None)
    found = []
    replaced = set()
    has_duplicates = False
    for page in pages:
        if page.verification != "text_match":
            continue
        lines = page.text.splitlines()
        for index, line in enumerate(lines):
            for caption in NATIVE_CAPTION.finditer(line):
                if not ABLATION_CAPTION.match(line[caption.end():].lstrip()):
                    continue
                ablations, header_at = ablation_tables(page, lines, index, caption, main)
                if not ablations:
                    continue
                found.extend(ablations)
                replaced.add((page.number, header_at))
                has_duplicates |= any(column.duplicate_scope for table in ablations for column in table.columns)

    existing = []
    for table in tables:
        if table is main:
            columns = []
            for column in table.columns:
                name = ANSWER_NAMES.get(column.header.name.casefold())
                header = replace(column.header, name=name) if name else column.header
                columns.append(replace(column, header=header, duplicate_scope=DUPLICATE_SCOPE if has_duplicates else None))
            table = replace(table, columns=columns)
        elif (table.page.number, table.index) in replaced and CONTEXT_CAPTION.search(table.caption):
            continue
        existing.append(table)
    return [*existing, *found]


def ablation_tables(page: Page, lines: list[str], caption_at: int, locator: re.Match[str],
                    main: Table | None) -> tuple[list[Table], int]:
    left = max(0, locator.start(1) - 3)
    caption = " ".join(lines[caption_at][locator.start(1):].split())
    for header_at in range(caption_at - 1, max(-1, caption_at - 20), -1):
        score_columns = headers(lines[header_at][left:])
        if tuple(column.name for column in score_columns) == ABLATION_METRICS:
            break
    else:
        return [], -1
    groups = []
    for line in reversed(lines[max(0, header_at - 4):header_at]):
        groups = groups_for(line[left:], score_columns[0].center, GROUP)
        if len(groups) >= 2:
            break
    if len(groups) != 2:
        return [], -1
    datasets = assign_groups(score_columns, groups)
    sides = [line[left:] for line in lines[:caption_at]]
    rows, _ = scan_rows(sides, header_at + 1, score_columns[0].center, len(score_columns))
    if len(rows) < 2 or len(rows) % 2:
        return [], -1
    values_from_main = {}
    if main:
        for row, values in main.rows:
            if row.startswith("Misknow-RAG with "):
                values_from_main[row] = {(column.header.name, column.dataset): raw
                                         for column, raw in zip(main.columns, values, strict=True)}
    evidence = {}
    split_evidence = None
    if main:
        if split_statement := source_statement([main.page], MAIN_COMPARISON):
            split_evidence = split_statement[0], split_statement[1][0]
        for name, (pattern, variant) in VARIANTS.items():
            if match := source_statement([main.page], pattern):
                evidence[name] = variant, (match[0], match[1][0])
    found = []
    for index in range(0, len(rows), 2):
        (base, values), (comparison, other_values) = rows[index:index + 2]
        if not BASE_ROW.fullmatch(base) or comparison != "w/o reasoning":
            return [], -1
        name = f"Misknow-RAG with {base}"
        base_values = values_from_main.get(name, {})
        columns = [Column(header, dataset, source_label=f"{dataset} / {header.label}",
                          data_split="test" if split_evidence else None, data_split_evidence=split_evidence)
                   for header, dataset in zip(score_columns, datasets, strict=True)]
        base_columns = []
        for column, raw in zip(columns, values, strict=True):
            metric = column.header.name
            variant, citation = evidence.get(metric, (None, None))
            if base_values.get((metric, column.dataset)) == raw:
                column = replace(column, duplicate_scope=DUPLICATE_SCOPE,
                                 metric_variant=variant, metric_variant_evidence=citation)
            base_columns.append(column)
        found.extend((
            Table(page, locator[1], caption, header_at, base_columns, [(base, values)], config_name=name),
            Table(page, locator[1], caption, header_at, columns, [(comparison, other_values)], row_section=base,
                  config_name=f"{name} / w/o reasoning"),
        ))
    return found, header_at
