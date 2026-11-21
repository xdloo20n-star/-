"""Extract knowledge-base inventories and retrieval-source comparisons."""

import re

from grouped_scores import native_evidence
from pdf_pages import Page
from rag_tables import Column, Header, NATIVE_CAPTION, Table


SOURCE_HEADING = re.compile(r"\bDataset +(?:Knowledge Base +(# Text Chunks)|Method +(QA Pairs Recall) +(Wiki Recall)) *$")
DATASET = r"Natural Questions|Huatuo-26M"
DECIMAL = r"\d+(?:\.\d+)?"
COUNT = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
INDEX_ROW = re.compile(rf"^ *(?:({DATASET}) +)?(Wikipedia|History QA Pairs|Merged KB) +({COUNT}) *$")
SOURCE_ROW = re.compile(rf"^ *(?:({DATASET}) +)?(2-way retrieval|Mis Wiki\+QA pairs|Mis QA Pairs\+Wiki) +({DECIMAL}) +({DECIMAL}) *$")
SOURCE_CAPTION = re.compile(r"Text chunk statistics about the knowledge bases|Results of context recall for QA pairs and")
QUERY_DATASETS = re.compile(rf"^ +({DATASET}) +({DATASET})(?= {{4,}}| *$)")
QUERY_HEADING = re.compile(r"^ +OrigQ +MisQ +OrigQ +MisQ *$")
QUERY_ROW = re.compile(rf"^ +(Wiki|QA Pairs) +({DECIMAL}) +({DECIMAL}) +({DECIMAL}) +({DECIMAL})(?= {{4,}}| *$)")
QUERY_CAPTION = re.compile(rf"^ *(Table \d+): +Evaluation on the retrieved contexts utilizing\b", re.I)
SAMPLES = re.compile(r"\b(\d+)\s+data\s+from\s+NQ\s+validation\s+set\s+and\s+(\d+)\s+consultation\s+data\s+from\s+Huatuo-?\s*26M\b", re.I)


def read_context_recall_tables(pages: list[Page], known: list[Table]) -> list[Table]:
    occupied = {(table.page.number, table.label.casefold()) for table in known}
    size = native_evidence(pages, SAMPLES)
    counts = dict(zip(("Natural Questions", "Huatuo-26M"), map(int, SAMPLES.fullmatch(size[1]).groups()))) if size else {}
    tables = []
    for page in pages:
        if page.verification != "text_match":
            continue
        lines = page.text.splitlines()
        for index, line in enumerate(lines):
            if heading := SOURCE_HEADING.search(line):
                tables.extend(source_grid(page, lines, index, heading, occupied, counts, size))
            if datasets := QUERY_DATASETS.match(line):
                tables.extend(query_grid(page, lines, index, datasets, counts, size))
    return tables


def source_grid(page: Page, lines: list[str], index: int, heading: re.Match[str],
                occupied: set[tuple[int, str]], counts: dict[str, int], size: tuple[Page, str] | None) -> list[Table]:
    left = max(0, heading.start() - 6)
    rows = []
    caption = None
    is_inventory = bool(heading[1])
    row_pattern = INDEX_ROW if is_inventory else SOURCE_ROW
    for line in lines[index + 1:index + 15]:
        local = line[left:]
        if (title := NATIVE_CAPTION.search(local)) and SOURCE_CAPTION.search(local[title.end():]):
            caption = title[1], " ".join(local[title.start(1):].split())
            break
        if row := row_pattern.fullmatch(local):
            rows.append(row.groups())
    if not caption or (page.number, caption[0].casefold()) in occupied or len(rows) != 6:
        return []
    if [row[0] for row in rows] != [None, "Natural Questions", None, None, "Huatuo-26M", None]:
        return []
    if [row[1] for row in rows[:3]] != [row[1] for row in rows[3:]] or len({row[1] for row in rows}) != 3:
        return []
    tables = []
    for group in (rows[:3], rows[3:]):
        dataset = group[1][0]
        if is_inventory:
            columns = [Column(Header(heading[1], "Indexed text chunks", 0, None, None), dataset,
                              source_label=heading[1], grouped_digits=True)]
        else:
            columns = [recall_column(dataset, label, variant, page, counts, size)
                       for label, variant in ((heading[2], "source-specific QA Pairs"), (heading[3], "source-specific Wiki"))]
        tables.append(Table(page, *caption, index, columns, [(row[1], list(row[2:])) for row in group],
                            row_section=dataset, family="context_recall"))
    return tables


def query_grid(page: Page, lines: list[str], index: int, datasets: re.Match[str],
               counts: dict[str, int], size: tuple[Page, str] | None) -> list[Table]:
    if len(set(datasets.groups())) != 2:
        return []
    headings = [i for i in range(index + 1, min(len(lines), index + 5)) if QUERY_HEADING.fullmatch(lines[i])]
    if len(headings) != 1:
        return []
    rows = []
    for at in range(headings[0] + 1, min(len(lines), headings[0] + 10)):
        line = lines[at]
        if footer := QUERY_CAPTION.match(line):
            if [row for row, _ in rows] != ["Wiki", "QA Pairs"]:
                return []
            caption = " ".join(re.split(r" {4,}", text.strip(), maxsplit=1)[0] for text in lines[at:at + 3])
            if "original query and the missing knowledge queries" not in caption or "recall scores" not in caption:
                return []
            columns = [recall_column(d, f"{d} / {label}", variant, page, counts, size, label)
                       for d in datasets.groups() for label, variant in (("OrigQ", "original query"), ("MisQ", "missing query"))]
            return [Table(page, footer[1], caption, index, columns, rows, family="context_recall")]
        if (row := QUERY_ROW.match(line)) and row.start(1) < datasets.start(1):
            rows.append((row[1], list(row.groups()[1:])))
    return []


def recall_column(dataset: str, label: str, variant: str, page: Page, counts: dict[str, int],
                  size: tuple[Page, str] | None, fragment: str | None = None) -> Column:
    sample = counts.get(dataset)
    return Column(
        Header(label, "Context recall", 0, None, None), dataset, source_label=label,
        metric_variant=variant, metric_variant_evidence=(page, fragment or label), reported_scale="percent",
        direction="higher_is_better", sample_size=sample, sample_size_evidence=size if sample else None,
        data_split="validation" if dataset == "Natural Questions" and sample else None,
        data_split_evidence=size if dataset == "Natural Questions" and sample else None,
    )
