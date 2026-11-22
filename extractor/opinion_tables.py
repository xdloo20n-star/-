"""Recover opinion-diversity comparisons from their positioned grids."""

from dataclasses import replace
from decimal import Decimal
import re

from failure_rates import visual_rows
from native_benchmark_tables import aligned, normalized_boxes, positioned_caption, words_text
from pdf_pages import Page, Word
from rag_tables import Column, Header, Table


DEPTH_PREFIX = "Diversity metrics at different retrieval depths (k) using hybrid retrieval."
DEPTH_CAPTION = re.compile(r"^ *(Table \d+): " + re.escape(DEPTH_PREFIX), re.M)
REFERENCE = re.compile(r"^Table \d+: Diversity metrics at k=(\d+) \(hybrid retrieval,$")
REFERENCE_DATASET = re.compile(r"hybrid retrieval,\s*forums dataset\)")
DATASET = "E-commerce seller forums"
MAIN_HEADERS = ("Entity Match", "Sent-Int σ", "Biz Impact σ", "Emot. Markers", "Revenue Band", "Tenure", "Evidence")
MAIN_COLUMNS = (0, 1, 2, 6, 3, 4, 5)
METRICS = (
    ("Match %", "Entity Match Rate", "%"),
    ("Sent-Int", "Sentiment-Intensity sigma", None),
    ("Biz Impact", "Business Impact sigma", None),
    ("Revenue Band", "Revenue Band Coverage", "%"),
    ("Tenure", "Tenure Coverage", "%"),
    ("Evidence", "Evidence Type Coverage", "%"),
    ("Marker", "Emotional Markers", None),
)
DEPTH_HEADER = "k KB Config Match % Sent-Int Biz Impact Revenue Band Tenure Evidence Marker"
GROUP_HEADER = "Entity Scored Diversity (σ) Categorical Coverage (%) Emot."
DECIMAL = re.compile(r"\d+\.\d+")
QUESTION_PREFIX = re.compile(r"Sentiment-Intensity σ by question type\b")
ENTITY_PREFIX = re.compile(r"Diversity metrics on entity-matched docu-?")
QUESTION_CAPTION = re.compile(QUESTION_PREFIX.pattern + r" \((k=(\d+))\)\.")
ENTITY_CAPTION = re.compile(ENTITY_PREFIX.pattern + r" ?ments only \((k=(\d+)), forums dataset\)\.")
QUESTION_TYPES = ("Detailed Overview", "Experience", "Business Impact", "Specific Examples", "Process", "Emotional")
ENTITY_METRICS = (
    ("Sent-Int σ", METRICS[1][1], METRICS[1][2]),
    ("Biz Impact σ", METRICS[2][1], METRICS[2][2]),
    ("Revenue Band Cov. (%)", METRICS[3][1], METRICS[3][2]),
    ("Tenure Cov. (%)", METRICS[4][1], METRICS[4][2]),
    ("Evidence Cov. (%)", METRICS[5][1], METRICS[5][2]),
    ("Emot. Markers", METRICS[6][1], METRICS[6][2]),
)
MATCHED_DOCS = re.compile(r"Matched Docs \(of (\d+)\)")
SIGNED_CHANGE = re.compile(r"([+−-]\d+\.\d+%)\*?")


def read_opinion_tables(pages: list[Page], known: list[Table]) -> list[Table]:
    occupied = {(table.page.number, table.label.casefold()) for table in known}
    tables = list(known)
    comparisons = []
    replaced = set()
    for page in pages:
        if page.verification != "text_match" or not page.words or not page.size:
            continue
        rows = visual_rows(page.words)
        for prefix, is_entity in ((QUESTION_PREFIX, False), (ENTITY_PREFIX, True)):
            match = re.search(r"^ *(Table \d+): " + prefix.pattern, page.text, re.M)
            if not match or not (anchor := positioned_caption(rows, match[1], prefix)):
                continue
            if comparison := read_comparison(page, match[1], rows, anchor, is_entity):
                comparisons.extend(comparison)
                replaced.add((page.number, match[1].casefold()))
        match = DEPTH_CAPTION.search(page.text)
        if not match:
            continue
        label = match[1]
        if (page.number, label.casefold()) in occupied:
            continue
        caption = positioned_caption(rows, label, re.compile(re.escape(DEPTH_PREFIX)))
        if caption is None:
            continue
        depth_tables = read_depth_groups(page, label, match[0].strip(), [row for row in rows if row[0].y > caption.y + 3])
        if depth_tables:
            reference = find_reference(known, depth_tables)
            if reference:
                index, depth = reference
                base = tables[index]
                scope = (base.page.number, base.label)
                tables[index] = replace(base, top_k=depth,
                                        top_k_evidence=(base.page, f"k={depth}"), columns=[
                    replace(column, header=replace(column.header, name=METRICS[position][1]),
                            dataset=DATASET, duplicate_scope=scope)
                    for column, position in zip(base.columns, MAIN_COLUMNS, strict=True)
                ])
                depth_tables = [replace(table, columns=[
                    replace(column, dataset=DATASET, duplicate_scope=scope if table.top_k == depth else None)
                    for column in table.columns
                ]) for table in depth_tables]
            tables.extend(depth_tables)
    return [table for table in tables if (table.page.number, table.label.casefold()) not in replaced] + comparisons


def read_comparison(page: Page, label: str, rows: list[list[Word]], anchor: Word, entity: bool) -> list[Table]:
    # The caption and grid both occupy the left column. Right-column prose
    # can share a baseline with the caption or the neighboring numeric cells.
    # Use word centers so wider bold percentages still belong to their column.
    right = page.size[0] / 2
    section = [line for row in rows if (line := [word for word in row
               if anchor.x0 - 5 < word.x0 and word.x < right and anchor.y - 3 < word.y < anchor.y + 150])]
    heading = "Metric Raw Enr. ∆" if entity else "Question Type Raw Enriched ∆"
    header_indices = [i for i, row in enumerate(section) if words_text(row) == heading]
    if len(header_indices) != 1:
        return []
    index = header_indices[0]
    data = section[index + 1:]
    caption_pattern = ENTITY_CAPTION if entity else QUESTION_CAPTION
    caption = caption_pattern.search(" ".join(words_text(line) for line in section[:index]))
    if not caption or len(data) != (7 if entity else len(QUESTION_TYPES)):
        return []
    depth = int(caption[2])
    if depth <= 0:
        return []

    printed_rows = [words_text(row[:-3]) for row in data]
    if entity:
        count = MATCHED_DOCS.fullmatch(printed_rows[0])
        if not count or printed_rows[1:] != [printed for printed, _, _ in ENTITY_METRICS]:
            return []
        metrics = [("entity-matched retrieved documents", None),
                   *((metric, unit) for _, metric, unit in ENTITY_METRICS)]
    else:
        if tuple(printed_rows) != QUESTION_TYPES:
            return []
        metrics = [(METRICS[1][1], None)] * len(QUESTION_TYPES)

    headings = section[index][-3:]
    centers = [word.x for word in headings]
    tables = []
    for printed, row, (metric, unit) in zip(printed_rows, data, metrics, strict=True):
        cells = row[-3:]
        if not aligned(cells, centers, 5):
            return []
        raw, enriched, delta = (cell.text for cell in cells)
        if entity and printed == printed_rows[0]:
            if not all(value.isdecimal() and int(value) <= int(count[1]) for value in (raw, enriched)):
                return []
            delta_name = "entity-matched retrieved document count relative change"
        else:
            if not all(DECIMAL.fullmatch(value) for value in (raw, enriched)):
                return []
            delta_name = f"{metric} relative change"
        change = SIGNED_CHANGE.fullmatch(delta)
        if (not change or Decimal(enriched) == Decimal(raw)
                or (change[1].startswith("+") != (Decimal(enriched) > Decimal(raw)))):
            return []
        columns = [Column(Header(word.text, name, word.x, depth, printed_unit), DATASET if entity else None,
                          source_label=word.text)
                   for word, name, printed_unit in zip(headings, (metric, metric, delta_name), (unit, unit, "%"),
                                                       strict=True)]
        for name, start, end in (("Traditional RAG Raw KB", 0, 1), ("O-RAG Enriched", 1, 3)):
            values = [raw, enriched, change[1]][start:end]
            tables.append(Table(page, label, f"{label}: {caption[0]}", int(row[0].y), columns[start:end],
                                [(printed, values)], config_name=name, top_k=depth, top_k_evidence=(page, caption[1]),
                                value_bboxes=normalized_boxes(page, cells[start:end])))
    return tables


def read_depth_groups(page: Page, label: str, caption: str, rows: list[list[Word]]) -> list[Table]:
    if len(rows) < 5 or words_text(rows[0]) != GROUP_HEADER or words_text(rows[1]) != DEPTH_HEADER:
        return []
    heading = rows[1]
    centers = ((heading[3].x + heading[4].x) / 2, heading[5].x, (heading[6].x + heading[7].x) / 2,
               (heading[8].x + heading[9].x) / 2, heading[10].x, heading[11].x, heading[12].x)
    tables = []
    previous_depth = 0
    for i in range(2, len(rows) - 2, 3):
        raw, depth_row, enriched = rows[i:i + 3]
        if raw[0].text != "Raw" or abs(raw[0].x - heading[1].x) > 8:
            break
        if (len(depth_row) != 1 or not depth_row[0].text.isdecimal() or abs(depth_row[0].x - heading[0].x) > 5
                or enriched[0].text != "Enriched" or enriched[0].y - raw[0].y > 18):
            return []
        depth = int(depth_row[0].text)
        if depth <= previous_depth or any(
            not aligned(row[1:], list(centers), 6) or any(not DECIMAL.fullmatch(w.text) for w in row[1:])
            for row in (raw, enriched)
        ):
            return []
        previous_depth = depth
        # The table evaluates every metric on the set of k retrieved documents.
        columns = [Column(Header(title, metric, center, depth, unit), None, source_label=title)
                   for center, (title, metric, unit) in zip(centers, METRICS, strict=True)]
        for row in (raw, enriched):
            cells = row[1:]
            tables.append(Table(page, label, caption, i, columns, [(row[0].text, [w.text for w in cells])],
                                row_section=f"k={depth}", config_name="Raw KB" if row is raw else "Enr. KB",
                                top_k=depth, top_k_evidence=(page, depth_row[0].text),
                                value_bboxes=normalized_boxes(page, cells)))
    return tables


def find_reference(known: list[Table], depth_tables: list[Table]) -> tuple[int, int] | None:
    for index, table in enumerate(known):
        match = REFERENCE.fullmatch(table.caption)
        if not match or not REFERENCE_DATASET.search(table.page.reading_text or table.page.text):
            continue
        if tuple(c.header.name for c in table.columns) != MAIN_HEADERS or [name for name, _ in table.rows] != ["Raw KB", "Enr. KB"]:
            continue
        depth = int(match[1])
        repeated = [row for row in depth_tables if row.top_k == depth]
        if len(repeated) != len(table.rows) or any(c.header.unit != METRICS[i][2]
                                                  for i, c in zip(MAIN_COLUMNS, table.columns, strict=True)):
            continue
        repeated_values = [[row.rows[0][1][i] for i in MAIN_COLUMNS] for row in repeated]
        if repeated_values == [values for _, values in table.rows]:
            return index, depth
    return None
