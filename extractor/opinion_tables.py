"""Read the hybrid-retrieval depth sweep and verify repetitions of its main comparison."""

from dataclasses import replace
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


def read_opinion_tables(pages: list[Page], known: list[Table]) -> list[Table]:
    occupied = {(table.page.number, table.label.casefold()) for table in known}
    tables = list(known)
    for page in pages:
        match = DEPTH_CAPTION.search(page.text)
        if not match or page.verification != "text_match" or not page.words or not page.size:
            continue
        label = match[1]
        if (page.number, label.casefold()) in occupied:
            continue
        rows = visual_rows(page.words)
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
