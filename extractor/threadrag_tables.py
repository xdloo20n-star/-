"""Read positioned ablation and cost results from ThreadRAG layouts."""

from dataclasses import replace
from decimal import Decimal
from itertools import pairwise
import re

from failure_rates import visual_rows
from grouped_scores import native_evidence
from native_benchmark_tables import aligned, body_above, normalized_boxes, positioned_caption, words_text
from pdf_pages import Page, Word
from rag_tables import Column, Header, NATIVE_CAPTION, Table, column_boundaries


EM_DEFINITION = re.compile(r"Exact Match \(EM\) and F1 scores to measure abso-?\s*lute correctness and token-level "
                           r"overlap, applying\s+standard text normalization to both predictions and\s+ground-truth answers")
EM_VARIANT = "exact match after standard text normalization"
DATASETS = {"HotpotQA": "HotpotQA", "2Wiki": "2WikiMultihopQA", "MuSiQue": "MuSiQue"}
FRACTION = re.compile(r"(?:0?\.\d+|1(?:\.0+)?)")
TOKEN_COUNT = re.compile(r"\d+(?:\.\d+)?[kM]|\d+")
ABLATION_PREFIX = re.compile(r"Ablation study of the STM \(SN, STG\)")
COST_PREFIX = re.compile(r"Comparison of construction \(Const\.\), inference")
CASE_PREFIX = re.compile(r"Case on MuSiQue\.")
GRAPHFREE_PREFIX = re.compile(r"Controlled Graph-Free ablation in terms of")
SELECTION_PREFIX = re.compile(r"Comparison of evidence selection strategies on")
BOUNDARY_PREFIX = re.compile(r"Intrinsic evaluation of the semantic boundary")
ABLATION_SETTINGS = (
    ("SN+STG+EDP", "SN+STG+EDP"), ("none", "no SN, no STG, no EDP"), ("EDP", "EDP only; no SN or STG"),
    ("SN+STG", "SN+STG; no EDP"), ("STG+EDP", "STG+EDP; no SN"),
)


def read_threadrag_tables(pages: list[Page], known: list[Table]) -> list[Table]:
    occupied = {(table.page.number, table.label.casefold()) for table in known}
    definition = native_evidence(pages, EM_DEFINITION)
    tables = []
    for page in pages:
        if page.verification != "text_match" or not page.words or not page.size:
            continue
        rows = visual_rows(page.words)
        for line in page.text.splitlines():
            for locator in NATIVE_CAPTION.finditer(line):
                label = locator[1]
                if (page.number, label.casefold()) in occupied:
                    continue
                text = line[locator.end():].lstrip()
                for prefix, reader, uses_em in ((ABLATION_PREFIX, read_ablation, True), (COST_PREFIX, read_cost, False),
                                                 (CASE_PREFIX, read_case, False), (GRAPHFREE_PREFIX, read_graphfree, True),
                                                 (SELECTION_PREFIX, read_selection, True),
                                                 (BOUNDARY_PREFIX, read_boundary, False)):
                    if (not uses_em or definition) and prefix.match(text) and (anchor := positioned_caption(rows, label, prefix)):
                        parsed = reader(page, label, anchor, definition)
                        if parsed:
                            occupied.add((page.number, label.casefold()))
                            tables.extend(parsed)
    return tables


def section(page: Page, anchor: Word, height: float, caption_lines: int):
    width, _ = page.size
    left, right = (0, width / 2) if anchor.x < width / 2 else (width / 2, width)
    # The other column can divide one local baseline into two visual rows.
    rows = visual_rows(tuple(word for word in page.words if left < word.x < right))
    above = body_above(rows, anchor, left, right, height)
    caption_rows = [row for row in rows if anchor.y - 2 <= row[0].y < anchor.y + 50]
    return above, " ".join(words_text(row) for row in caption_rows[:caption_lines])


def after_header(grid: list[list[Word]], heading: str, count: int) -> list[list[Word]]:
    starts = [i for i, row in enumerate(grid) if words_text(row) == heading]
    if len(starts) != 1 or len(grid) - starts[0] != count + 1:
        return []
    return grid[starts[0]:]


def em_column(title: str, center: float, dataset: str, definition: tuple[Page, str]) -> Column:
    return Column(Header(title, "EM", center, None, None), dataset, direction="higher_is_better",
                  source_label=f"{title} / EM" if title in DATASETS else title,
                  metric_variant=EM_VARIANT, metric_variant_evidence=definition, reported_scale="fraction")


def joined_box(words: list[Word]) -> Word:
    return Word(words_text(words), min(w.x0 for w in words), min(w.y0 for w in words),
                max(w.x1 for w in words), max(w.y1 for w in words))


def read_ablation(page: Page, label: str, anchor: Word,
                  definition: tuple[Page, str]) -> list[Table]:
    above, caption = section(page, anchor, 115, 2)
    grid = after_header(above, "SN STG EDP HotpotQA 2Wiki MuSiQue", 5)
    if not grid or not caption.endswith("and EDP modules across three datasets."):
        return []
    heading, *body = grid
    if not any(words_text(row) == "Ablation Setting Datasets" for row in above):
        return []
    centers = [word.x for word in heading]
    boundaries = column_boundaries(centers)
    tables = []
    for row, (expected, config) in zip(body, ABLATION_SETTINGS, strict=True):
        bands = [[word for word in row if x0 <= word.x < x1] for x0, x1 in pairwise(boundaries)]
        if sum(map(len, bands)) != len(row) or any(band and words_text(band) != "✓" for band in bands[:3]):
            return []
        setting = "+".join(word.text for word, band in zip(heading[:3], bands[:3], strict=True) if band) or "none"
        if setting != expected:
            return []
        columns, cells, values = [], [], []
        for title, dataset, band, center in zip(DATASETS, DATASETS.values(), bands[3:], centers[3:], strict=True):
            if not band or not FRACTION.fullmatch(band[0].text):
                return []
            columns.append(em_column(title, center, dataset, definition))
            cells.append(band[0])
            values.append(band[0].text)
            if setting == "SN+STG+EDP":
                if len(band) != 1:
                    return []
                continue
            delta = re.fullmatch(r"\((↓\s*\d+(?:\.\d+)?%)\)", words_text(band[1:]))
            if delta is None:
                return []
            columns.append(Column(Header("↓", "EM decrease", center, None, "%"), dataset, source_label=f"{title} / ↓"))
            values.append(delta[1])
            cells.append(joined_box(band[1:]))
        tables.append(Table(page, label, caption, len(tables), columns, [(setting, values)],
                            config_name=f"ThreadRAG ablation: {config}", value_bboxes=normalized_boxes(page, cells)))
    return tables


def read_cost(page: Page, label: str, anchor: Word, _definition) -> list[Table]:
    above, caption = section(page, anchor, 95, 2)
    grid = after_header(above, "Const. Inf. Total Const. Inf. Total", 3)
    if not grid or not caption.endswith("(Inf.), and total token costs."):
        return []
    heading, *body = grid
    groups = next((row for row in above if words_text(row) == "HotpotQA MuSiQue"), [])
    if (not groups or not any(words_text(row) == "Method" for row in above)
            or not all(heading[i].x < word.x < heading[i + 2].x for i, word in zip((0, 3), groups, strict=True))):
        return []
    if [words_text(row[:-6]) for row in body] != ["LightRAG", "KnowTrace", "ThreadRAG"]:
        return []
    centers = [word.x for word in heading]
    tables = []
    for row in body:
        name = row[0].text
        cells = row[1:]
        if not aligned(cells, centers, 6) or any(not TOKEN_COUNT.fullmatch(word.text) for word in cells):
            return []
        columns = [Column(Header(word.text, "Token cost", word.x, None, "tokens"), dataset, direction="lower_is_better",
                          source_label=f"{dataset} / {word.text}", metric_variant=variant)
                   for word, dataset, variant in zip(heading, ["HotpotQA"] * 3 + ["MuSiQue"] * 3,
                                                      ("construction", "inference", "total") * 2, strict=True)]
        tables.append(Table(page, label, caption, len(tables), columns, [(name, [word.text for word in cells])],
                            value_bboxes=normalized_boxes(page, cells)))
    return tables


def read_case(page: Page, label: str, anchor: Word, _definition) -> list[Table]:
    rows = visual_rows(tuple(word for word in page.words if anchor.y - 400 < word.y < anchor.y - 3))
    lines = [words_text(row) for row in rows]
    if "# Semantic Topology Graph (STG) Construction:" not in lines or "# Graph Pruning & Topological Linearization:" not in lines:
        return []
    patterns = (
        r"- Graph initialized with \d+ context-intact nodes and \d+ structural \(narrative/semantic/lexical\) edges\.",
        r"- \d+ query-relevant anchor nodes identified based on the extracted entities\.",
        r"- \d+ highly activated nodes retained out of \d+ after energy threshold pruning\.",
    )
    matched = [[row for row, line in zip(rows, lines, strict=True) if re.fullmatch(pattern, line)] for pattern in patterns]
    if any(len(matches) != 1 for matches in matched):
        return []
    graph, anchors, retained = [[word for word in line[0] if word.text.isdecimal()] for line in matched]
    if not (graph[0].y < anchors[0].y < retained[0].y and 0 < int(retained[0].text) <= int(retained[1].text)
            and 0 < int(anchors[0].text) <= int(graph[0].text)):
        return []
    repeat = (page.number, label) if graph[0].text == retained[1].text else None
    descriptions = (
        ("Graph construction", graph, ("Context-intact nodes", "Structural edges"),
         ("Semantic node count", "Structural edge count")),
        ("Anchor identification", anchors, ("Query-relevant anchor nodes",), ("Anchor count",)),
        ("Graph pruning", retained, ("Retained nodes", "Original nodes"), ("Retained node count", "Semantic node count")),
    )
    tables = []
    for row, cells, titles, metrics in descriptions:
        columns = [Column(Header(title, metric, word.x, None, None), "MuSiQue", source_label=title,
                          duplicate_scope=repeat if metric == "Semantic node count" else None)
                   for word, title, metric in zip(cells, titles, metrics, strict=True)]
        tables.append(Table(page, label, f"{label}: Case on MuSiQue.", len(tables), columns,
                            [(row, [word.text for word in cells])], config_name=f"ThreadRAG {label} example",
                            value_bboxes=normalized_boxes(page, cells)))
    return tables


def read_graphfree(page: Page, label: str, anchor: Word,
                   definition: tuple[Page, str]) -> list[Table]:
    above, caption = section(page, anchor, 80, 4)
    grid = after_header(above, "Method HotpotQA 2Wiki MuSiQue", 2)
    if not grid or "Exact Match (EM)." not in caption:
        return []
    heading, *body = grid
    if [words_text(row[:-3]) for row in body] != ["Graph-Free", "ThreadRAG"]:
        return []
    centers = [word.x for word in heading[1:]]
    columns = [em_column(title, center, dataset, definition)
               for title, dataset, center in zip(DATASETS, DATASETS.values(), centers, strict=True)]
    tables = []
    for row in body:
        name, *cells = row
        if not aligned(cells, centers, 6) or not all(FRACTION.fullmatch(word.text) for word in cells):
            return []
        tables.append(Table(page, label, caption, len(tables), columns, [(name.text, [word.text for word in cells])],
                            config_name=f"{name.text} (controlled graph-free ablation)",
                            value_bboxes=normalized_boxes(page, cells)))
    return tables


def read_selection(page: Page, label: str, anchor: Word,
                   definition: tuple[Page, str]) -> list[Table]:
    grid, caption = section(page, anchor, 90, 3)
    match = re.search(r"MuSiQue using (LLaMA[\w.-]+)\. All methods use the same candidate graph and evidence budget\.", caption)
    if not match:
        return []
    starts = [i for i, row in enumerate(grid) if words_text(row) == "Subgraph Strategy EM ∆ vs. EDP"]
    if len(starts) != 1:
        return []
    heading, *body = grid[starts[0]:]
    em_x = heading[2].x
    delta_x = (heading[3].x0 + heading[-1].x1) / 2
    # The delta words can lie a few points below the baseline of their scores.
    rows_with_scores = [row for row in body if any(abs(word.x - em_x) <= 5 for word in row)]
    names = ("No subgraph", "BFS (depth = 3)", "DFS (depth = 3)", "EDP")
    if len(rows_with_scores) != len(names):
        return []
    columns = [em_column("EM", em_x, "MuSiQue", definition),
               Column(Header("∆ vs. EDP", "EM difference vs EDP", delta_x, None, None), "MuSiQue",
                      source_label="∆ vs. EDP")]
    columns = [replace(column, generator=match[1], generator_evidence=(page, match[1]))
               for column in columns]
    tables = []
    used_deltas = set()
    for row, name in zip(rows_with_scores, names, strict=True):
        score = next(word for word in row if abs(word.x - em_x) <= 5)
        if words_text([word for word in row if word.x < score.x]) != name or not FRACTION.fullmatch(score.text):
            return []
        delta_cells = [word for line in body for word in line if abs(word.y - score.y) <= 4.5 and abs(word.x - delta_x) <= 5]
        if len(delta_cells) != 1 or delta_cells[0] in used_deltas:
            return []
        delta, = delta_cells
        if (name == "EDP" and delta.text != "–") or (name != "EDP" and not re.fullmatch(r"[−-]0\.\d+", delta.text)):
            return []
        used_deltas.add(delta)
        config = re.sub(r" \(depth = (\d+)\)", r"(depth=\1)", name)
        tables.append(Table(page, label, caption, len(tables), columns, [(name, [score.text, delta.text])],
                            config_name=f"ThreadRAG selection: {config}", value_bboxes=normalized_boxes(page, [score, delta])))
    baseline = Decimal(tables[-1].rows[0][1][0])
    for table in tables[:-1]:
        score, delta = table.rows[0][1]
        if Decimal(score) - baseline != Decimal(delta.replace("−", "-")):
            return []
    return tables


def read_boundary(page: Page, label: str, anchor: Word, _definition) -> list[Table]:
    above, caption = section(page, anchor, 75, 4)
    grid = after_header(above, "Precision Recall F1", 1)
    context = re.search(r"predictor on ((\d+) held-out Wikipedia documents)\. Scores measure agreement with "
                        r"teacher-generated reference boundaries\.", caption)
    if not grid or not context or int(context[2]) <= 0:
        return []
    heading, cells = grid
    if not aligned(cells, [word.x for word in heading], 6) or not all(FRACTION.fullmatch(word.text) for word in cells):
        return []
    columns = [Column(Header(word.text, word.text, word.x, None, None), "held-out Wikipedia semantic-boundary evaluation",
                      direction="higher_is_better", source_label=word.text, reported_scale="fraction",
                      metric_variant="agreement with teacher-generated boundary references",
                      metric_variant_evidence=(page, "agreement with teacher-generated reference"),
                      sample_size=int(context[2]), sample_size_evidence=(page, context[1])) for word in heading]
    return [Table(page, label, caption, 0, columns, [("Semantic boundary predictor", [word.text for word in cells])],
                  config_name="ThreadRAG STM boundary predictor", value_bboxes=normalized_boxes(page, cells))]
