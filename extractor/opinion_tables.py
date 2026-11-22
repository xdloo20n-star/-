"""Recover opinion-diversity comparisons from their positioned grids."""

from dataclasses import replace
from decimal import Decimal
import re

from failure_rates import visual_rows
from grouped_scores import native_evidence
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
DISTANCE_PREFIX = re.compile(r"Empirical W(?=\s|1)")
HOTEL_PREFIX = re.compile(r"OpinioBank results \(public hotel reviews\)")
SPLIT_PREFIX = re.compile(r"Entity-Split-Only ablation at k=")
RETRIEVAL_PREFIX = re.compile(r"Semantic-only vs\. hybrid retrieval at k=")
REFERENCE_DELTA_PREFIX = re.compile(r"Diversity metrics at k=\d+ \(hybrid retrieval,")
EXTRACTION_MODEL_PREFIX = re.compile(r"Retrieval diversity improvements \(∆ vs\. Raw")
GENERATION = re.compile(r"Response generation from the retrieved context\s+used (Anthropic Claude Sonnet) (4\.5)\.")
WILCOXON = re.compile(r"Wilcoxon\s+signed-rank test")
P_VALUE = r"(<0\.\d+)"
HOTELS = {
    "Silverton": "OpinioBank: Silverton Las Vegas", "Silverton LV": "OpinioBank: Silverton Las Vegas",
    "Ambassador": "OpinioBank: Ambassador Bangkok", "Ambassador Bangkok": "OpinioBank: Ambassador Bangkok",
    "Ala Moana": "OpinioBank: Ala Moana Honolulu", "Ala Moana Honolulu": "OpinioBank: Ala Moana Honolulu",
}
RAW_KB = "Traditional RAG Raw KB"
ENRICHED_KB = "O-RAG Enriched"
SPLIT_METRICS = tuple(METRICS[i][1] for i in (1, 2, 6))
RETRIEVAL_METRICS = {
    "Sent-Int σ": METRICS[1], "Biz Impact σ": METRICS[2], "Revenue Band (%)": METRICS[3],
    "Tenure (%)": METRICS[4], "Evidence (%)": METRICS[5], "Emot. Markers": METRICS[6],
}
REFERENCE_METRICS = {"Entity Match (%)": METRICS[0], **RETRIEVAL_METRICS}
EXTRACTION_MODELS = {"Mistral": "Mistral Large", "Qwen": "Qwen 235B", "Sonnet": "Claude Sonnet 4.5"}
EXTRACTION_MODEL_EVIDENCE = re.compile(r"Mistral Large,\s+Qwen 235B,\s+and Claude Sonnet 4\.5")
MODEL_APPENDIX_EVIDENCE = re.compile(
    r"Cross-model consistency\.\s+The Full O-RAG improvement over Raw KB is consistent across extraction models:\s+"
    r"Mistral Large \(([+−-]\d+\.\d+%) Sent-Int\s+σ\),\s+Qwen 235B \(([+−-]\d+\.\d+%)\),\s+and Claude Sonnet 4\.5\s+"
    r"\(([+−-]\d+\.\d+%)\),[^()]{0,120}\(detailed in Appendix B\)"
)
EXTRACTION_MODEL_METRICS = {
    "Sent-Int σ": METRICS[1][1], "Biz Impact σ": METRICS[2][1], "Revenue Band Cov.": METRICS[3][1],
    "Tenure Cov.": METRICS[4][1], "Evidence Cov.": METRICS[5][1], "Emot. Markers": METRICS[6][1],
}


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
                tables[index] = replace(base, top_k=depth, retrieval_method="hybrid",
                                        top_k_evidence=(base.page, f"k={depth}"), columns=[
                    replace(column, header=replace(column.header, name=METRICS[position][1]),
                            dataset=DATASET, duplicate_scope=scope)
                    for column, position in zip(base.columns, MAIN_COLUMNS, strict=True)
                ])
                depth_tables = [replace(table, retrieval_method="hybrid", columns=[
                    replace(column, dataset=DATASET, duplicate_scope=scope if table.top_k == depth else None)
                    for column in table.columns
                ]) for table in depth_tables]
            tables.extend(depth_tables)
    tables = [table for table in tables if (table.page.number, table.label.casefold()) not in replaced] + comparisons
    return tables + read_opinion_supplement(pages, tables)


def read_opinion_supplement(pages: list[Page], known: list[Table]) -> list[Table]:
    occupied = {(table.page.number, table.label.casefold()) for table in known}
    generation = native_evidence(pages, GENERATION)
    reference = next((table for table in known if table.retrieval_method == "hybrid" and table.columns
                      and table.columns[0].duplicate_scope == (table.page.number, table.label)), None)
    tables = []
    for page in pages:
        if page.verification != "text_match" or not page.words or not page.size:
            continue
        rows = visual_rows(page.words)
        for prefix, reader in ((DISTANCE_PREFIX, read_distance), (HOTEL_PREFIX, read_hotels),
                               (SPLIT_PREFIX, read_split_ablation), (RETRIEVAL_PREFIX, read_retrieval_method),
                               (REFERENCE_DELTA_PREFIX, read_reference_deltas), (EXTRACTION_MODEL_PREFIX, read_model_changes)):
            for caption in re.finditer(r"(?:^ *|(?<= {2}))(Table \d+): " + prefix.pattern, page.text, re.M):
                if prefix is not REFERENCE_DELTA_PREFIX and (page.number, caption[1].casefold()) in occupied:
                    continue
                if anchor := positioned_caption(rows, caption[1], prefix):
                    tables.extend(reader(page, caption[1], anchor, reference))
    if generation:
        page, statement = generation
        model, version = GENERATION.fullmatch(statement).groups()
        tables = [replace(table, columns=[replace(column, generator=model, generator_version=version,
                                                  generator_evidence=(page, model), generator_version_evidence=(page, version))
                                          if not column.duplicate_scope else column
                                          for column in table.columns]) for table in tables]
    return link_model_changes(tables, native_evidence(pages, EXTRACTION_MODEL_EVIDENCE),
                              native_evidence(pages, MODEL_APPENDIX_EVIDENCE))


def opinion_grid(page: Page, anchor: Word, heading: str, count: int, *, full_width: bool = False):
    midpoint = page.size[0] / 2
    left, right = ((0, page.size[0]) if full_width else
                   (0, midpoint) if anchor.x < midpoint else (midpoint, page.size[0]))
    words = sorted((word for word in page.words if left < word.x < right and anchor.y - 2 <= word.y < anchor.y + 155),
                   key=lambda word: word.y)
    rows = []
    for word in words:
        if not rows or word.y - rows[-1][0].y > 3:
            rows.append([])
        rows[-1].append(word)
    for row in rows:
        row.sort(key=lambda word: word.x)
    headers = [i for i, row in enumerate(rows) if re.sub(r"\bW 1\b", "W1", words_text(row)) == heading]
    if len(headers) != 1:
        return None
    index, = headers
    data = rows[index + 1:index + count + 1]
    if len(data) != count:
        return None
    caption = re.sub(r"\bW 1\b", "W1", " ".join(words_text(row) for row in rows[:index]))
    return caption, rows[index], data


def read_distance(page: Page, label: str, anchor: Word, reference: Table | None) -> list[Table]:
    grid = opinion_grid(page, anchor, "Dataset Raw W1 Enriched W1 ∆", 4)
    if not grid:
        return []
    caption, heading, rows = grid
    match = re.search(r"Empirical W1 between corpus-level and re-? ?trieved sentiment distributions "
                      r"\((k=(\d+))\)\. Lower is better\.", caption)
    if not match or int(match[2]) <= 0:
        return []
    depth = int(match[2])
    enriched = next(i for i, word in enumerate(heading) if word.text == "Enriched")
    centers = ((heading[1].x0 + heading[enriched - 1].x1) / 2,
               (heading[enriched].x0 + heading[-2].x1) / 2, heading[-1].x)
    names = {"Forums": DATASET, **{key: HOTELS[key] for key in ("Silverton", "Ambassador", "Ala Moana")}}
    if [words_text(row[:-3]) for row in rows] != list(names):
        return []
    tables = []
    for row_name, row in zip(names, rows, strict=True):
        cells = row[-3:]
        raw, enriched, delta = (word.text for word in cells)
        if not (aligned(cells, centers, 9) and DECIMAL.fullmatch(raw) and DECIMAL.fullmatch(enriched)
                and re.fullmatch(r"[+−-]\d+(?:\.\d+)?%", delta)):
            return []
        if Decimal(raw) == Decimal(enriched) or (delta.startswith("+") != (Decimal(enriched) > Decimal(raw))):
            return []
        specs = (("Raw W1", "W1 sentiment distribution distance", None, "lower_is_better"),
                 ("Enriched W1", "W1 sentiment distribution distance", None, "lower_is_better"),
                 ("∆", "W1 relative change", "%", None))
        columns = [Column(Header(title, metric, center, depth, unit), names[row_name], direction=direction,
                          source_label=title)
                   for center, (title, metric, unit, direction) in zip(centers, specs, strict=True)]
        for name, start, end in ((RAW_KB, 0, 1), (ENRICHED_KB, 1, 3)):
            tables.append(Table(page, label, caption, len(tables), columns[start:end],
                                [(row_name, [word.text for word in cells[start:end]])], config_name=name,
                                top_k=depth, top_k_evidence=(page, match[1]), value_bboxes=normalized_boxes(page, cells[start:end])))
    return tables


def read_hotels(page: Page, label: str, anchor: Word, reference: Table | None) -> list[Table]:
    grid = opinion_grid(page, anchor, "Property Reviews Entities Entity Match ∆ Sent-Int σ ∆", 3, full_width=True)
    if not grid:
        return []
    caption, heading, rows = grid
    if not caption.endswith("Entity match improvement and sentiment-intensity σ change vs. Raw KB."):
        return []
    centers = (heading[1].x, heading[2].x, (heading[3].x0 + heading[5].x1) / 2, (heading[6].x0 + heading[8].x1) / 2)
    names = ("Silverton LV", "Ambassador Bangkok", "Ala Moana Honolulu")
    if [words_text(row[:-4]) for row in rows] != list(names):
        return []
    tables = []
    for row_name, row in zip(names, rows, strict=True):
        reviews, entities, match_gain, sentiment_gain = (word.text for word in row[-4:])
        if not (aligned(row[-4:], centers, 9) and re.fullmatch(r"\d{1,3}(?:,\d{3})*", reviews)
                and entities.isdecimal() and re.fullmatch(r"[+−-]\d+\.\d+pp", match_gain)
                and re.fullmatch(r"[+−-]\d+\.\d+", sentiment_gain)):
            return []
        specs = (("Entity Match ∆", "Entity Match Rate absolute gain", "pp"),
                 ("Sent-Int σ ∆", "Sentiment-Intensity sigma absolute gain", None))
        columns = [Column(Header(title, metric, word.x, None, unit), HOTELS[row_name], source_label=title)
                   for word, (title, metric, unit) in zip(row[-2:], specs, strict=True)]
        tables.append(Table(page, label, caption, len(tables), columns, [(row_name, [match_gain, sentiment_gain])],
                            config_name=ENRICHED_KB, value_bboxes=normalized_boxes(page, row[-2:])))
    return tables


def reference_scope(reference: Table | None, depth: int, metrics: tuple[str, ...],
                    values: list[list[str]]) -> tuple[int, str] | None:
    if reference is None or depth != reference.top_k:
        return None
    lookup = {column.header.name: i for i, column in enumerate(reference.columns)}
    if not set(metrics) <= lookup.keys():
        return None
    expected = [[row[lookup[metric]] for metric in metrics] for _, row in reference.rows]
    return (reference.page.number, reference.label) if values == expected else None


def p_bound_column(title: str, metric: str, comparison: str, marker: Word, depth: int,
                   method: tuple[Page, str], sample_size: int | None = None) -> Column:
    return Column(Header(title, "p-value bound", marker.x, depth, None), DATASET, source_label=f"{title} / *",
                  metric_variant=f"Wilcoxon signed-rank; {metric}; {comparison}",
                  metric_variant_evidence=method, sample_size=sample_size)


def read_reference_deltas(page: Page, label: str, anchor: Word, _reference: Table | None) -> list[Table]:
    grid = opinion_grid(page, anchor, "Metric Raw KB Enr. KB ∆", 7)
    if not grid:
        return []
    caption, header, rows = grid
    context = re.search(r"Diversity metrics at (k=(\d+)) \(hybrid retrieval, forums dataset\)\. "
                        r"Statistical significance via Wilcoxon signed-rank test \(n=(\d+)\): "
                        r"bold = p<0\.\d+, \* = p" + P_VALUE + r"\.", caption)
    method = native_evidence([page], WILCOXON)
    if not (context and method and int(context[2]) > 0 and int(context[3]) > 0):
        return []
    depth, sample_size = int(context[2]), int(context[3])
    centers = ((header[1].x0 + header[2].x1) / 2, (header[3].x0 + header[4].x1) / 2, header[-1].x)
    ordering = ("Entity Match (%)", "Sent-Int σ", "Biz Impact σ", "Emot. Markers", "Revenue Band (%)", "Tenure (%)",
                "Evidence (%)")
    if tuple(words_text(row[:-3]) for row in rows) != ordering:
        return []
    tables = []
    for printed, row in zip(ordering, rows, strict=True):
        raw, enriched, marked = row[-3:]
        change = SIGNED_CHANGE.fullmatch(marked.text)
        if (not aligned(row[-3:], centers, 6) or not change or not DECIMAL.fullmatch(raw.text)
                or not DECIMAL.fullmatch(enriched.text) or Decimal(raw.text) == Decimal(enriched.text)
                or (change[1].startswith("+") != (Decimal(enriched.text) > Decimal(raw.text)))):
            return []
        _, metric, _ = REFERENCE_METRICS[printed]
        columns = [Column(Header("∆", f"{metric} relative change", marked.x, depth, "%"), DATASET, source_label="∆")]
        values, cells = [marked.text], [marked]
        if marked.text.endswith("*"):
            columns.append(p_bound_column("∆", metric, "Enriched vs Raw", marked, depth, method, sample_size))
            values.append(context[4])
            cells.append(marked)
        tables.append(Table(page, label, caption, len(tables), columns, [(printed, values)], config_name=ENRICHED_KB,
                            family="opinion_reference_delta", top_k=depth, top_k_evidence=(page, context[1]), retrieval_method="hybrid",
                            value_bboxes=normalized_boxes(page, cells)))
    return tables


def read_model_changes(page: Page, label: str, anchor: Word, _reference: Table | None) -> list[Table]:
    grid = opinion_grid(page, anchor, "Metric Mistral Qwen Sonnet", 6)
    if not grid:
        return []
    caption, heading, rows = grid
    context = re.search(r"\(∆ vs\. Raw KB\) across opinion extraction LLMs \((k=(\d+)), hybrid\)\.", caption)
    if not context or int(context[2]) <= 0 or [words_text(row[:-3]) for row in rows] != list(EXTRACTION_MODEL_METRICS):
        return []
    depth = int(context[2])
    centers = [word.x for word in heading[1:]]
    tables = []
    for row, (printed, metric) in zip(rows, EXTRACTION_MODEL_METRICS.items(), strict=True):
        cells = row[-3:]
        if not aligned(cells, centers, 6) or not all(SIGNED_CHANGE.fullmatch(word.text) for word in cells):
            return []
        for word, model in zip(cells, EXTRACTION_MODELS, strict=True):
            column = Column(Header(model, f"{metric} relative change", word.x, depth, "%"), None, source_label=model)
            tables.append(Table(page, label, caption, len(tables), [column], [(printed, [word.text])],
                                family="opinion_model_change", config_name=ENRICHED_KB, retrieval_method="hybrid",
                                config_note=f"Opinion extraction: {model}", top_k=depth, top_k_evidence=(page, context[1]),
                                value_bboxes=normalized_boxes(page, [word])))
    return tables


def link_model_changes(tables: list[Table], names: tuple[Page, str] | None,
                       comparison: tuple[Page, str] | None) -> list[Table]:
    references = [(i, table) for i, table in enumerate(tables) if table.family == "opinion_reference_delta"]
    changes = [(i, table) for i, table in enumerate(tables) if table.family == "opinion_model_change"]
    if not changes:
        return tables
    for i, table in changes:
        model = table.columns[0].source_label
        if names:
            full_name = EXTRACTION_MODELS[model]
            tables[i] = replace(table, config_note=f"Opinion extraction: {full_name}",
                                config_note_evidence=(names[0], full_name))
    if comparison is None:
        return tables
    expected_changes = dict(zip(EXTRACTION_MODELS, MODEL_APPENDIX_EVIDENCE.fullmatch(comparison[1]).groups(), strict=True))
    named_changes = {table.columns[0].source_label: table.rows[0][1][0].removesuffix("*")
                     for _, table in changes if table.rows[0][0] == "Sent-Int σ"}
    if named_changes != expected_changes:
        return tables
    scope = {(table.page.number, table.label) for _, table in references}
    if len(scope) != 1:
        return tables
    scope, = scope
    original = {(table.top_k, table.columns[0].header.name, table.rows[0][1][0]): i for i, table in references}
    sonnet = [(i, table) for i, table in changes if table.columns[0].source_label == "Sonnet"]
    corresponding = [original.get((table.top_k, table.columns[0].header.name, table.rows[0][1][0])) for _, table in sonnet]
    if len(sonnet) != len(EXTRACTION_MODEL_METRICS) or None in corresponding or len(set(corresponding)) != len(corresponding):
        return tables
    for (i, _), j in zip(sonnet, corresponding, strict=True):
        reference, model = tables[j], tables[i]
        columns = [replace(reference.columns[0], duplicate_scope=scope), *reference.columns[1:]]
        tables[j] = replace(reference, columns=columns, config_note=model.config_note, config_note_evidence=comparison)
        tables[i] = replace(model, columns=[replace(model.columns[0], dataset=DATASET, duplicate_scope=scope)])
    for i, _ in changes:
        table = tables[i]
        if table.columns[0].dataset is None:
            tables[i] = replace(table, columns=[replace(table.columns[0], dataset=DATASET)])
    return tables


def read_split_ablation(page: Page, label: str, anchor: Word, reference: Table | None) -> list[Table]:
    grid = opinion_grid(page, anchor, "KB Config Sent-Int σ Biz Impact Emot Marker", 3)
    if not grid:
        return []
    caption, heading, rows = grid
    match = re.search(r"Entity-Split-Only ablation at (k=(\d+)) \((Sonnet 4\.5|Mistral Large)\)\.", caption)
    if not match or int(match[2]) <= 0:
        return []
    depth, model = int(match[2]), match[3]
    model_source = native_evidence([page], re.compile(re.escape(model).replace(r"\ ", r"\s+")))
    row_names = ("Raw", "Entity-Split Only", "Full O-RAG")
    if [words_text(row[:-3]) for row in rows] != list(row_names):
        return []
    centers = tuple((heading[i].x0 + heading[i + 1].x1) / 2 for i in (2, 4, 6))
    values = [[word.text.removesuffix("*") for word in row[-3:]] for row in rows]
    if not all(aligned(row[-3:], centers, 9) and all(DECIMAL.fullmatch(value) for value in row_values)
               for row, row_values in zip(rows, values, strict=True)):
        return []
    repeated = reference_scope(reference, depth, SPLIT_METRICS, [values[0], values[2]]) if model == "Sonnet 4.5" else None
    star = re.search(r"\* = p" + P_VALUE + r" vs\. Entity-Split\.", caption)
    method = native_evidence([page], WILCOXON) if star else None
    tables = []
    for i, (row_name, row, row_values, config) in enumerate(zip(
        row_names, rows, values, (RAW_KB, "O-RAG Entity-Split Only", ENRICHED_KB), strict=True,
    )):
        scope = repeated if i != 1 else None
        if scope:
            config = reference.rows[0 if i == 0 else 1][0]
        titles = ("Sent-Int σ", "Biz Impact", "Emot Marker")
        columns = [Column(Header(title, metric, center, depth, None), DATASET, source_label=title, duplicate_scope=scope)
                   for title, metric, center in zip(titles, SPLIT_METRICS, centers, strict=True)]
        cells = row[-3:]
        if i == 2 and method:
            for word, title, metric in zip(row[-3:], titles, SPLIT_METRICS, strict=True):
                if word.text.endswith("*"):
                    columns.append(p_bound_column(title, metric, "Enriched vs Entity-Split Only", word, depth, method))
                    row_values.append(star[1])
                    cells.append(word)
        tables.append(Table(page, label, caption, i, columns, [(row_name, row_values)], config_name=config,
                            top_k=depth, top_k_evidence=(page, match[1]), value_bboxes=normalized_boxes(page, cells),
                            retrieval_method="hybrid" if repeated else None,
                            config_note=f"Opinion extraction: {model}" if not scope and i else None,
                            config_note_evidence=model_source if not scope and i else None))
    return tables


def read_retrieval_method(page: Page, label: str, anchor: Word, reference: Table | None) -> list[Table]:
    grid = opinion_grid(page, anchor, "Metric Raw Enr. (∆) Raw Enr. (∆)", 6)
    if not grid:
        return []
    caption, heading, rows = grid
    match = re.search(r"Semantic-only vs\. hybrid retrieval at (k=(\d+))\.", caption)
    if not match or not caption.endswith(" Semantic Hybrid") or int(match[2]) <= 0:
        return []
    depth = int(match[2])
    groups = [(word.text, word.x) for word in sorted(page.words, key=lambda word: word.x)
              if 3 < heading[0].y - word.y < 20 and 0 < word.x < page.size[0] / 2]
    if (tuple(text for text, x in groups) != ("Semantic", "Hybrid")
            or not all(words[0].x < group_x < words[-1].x
                       for (_, group_x), words in zip(groups, (heading[1:4], heading[4:7]), strict=True))):
        return []
    if [words_text(row[:-6]) for row in rows] != list(RETRIEVAL_METRICS):
        return []
    centers = tuple(word.x for word in heading[1:])
    repeated = reference_scope(reference, depth, tuple(spec[1] for spec in RETRIEVAL_METRICS.values()),
                               [[row[index].text for row in rows] for index in (-3, -2)])
    tables = []
    for row, (printed, (_, metric, unit)) in zip(rows, RETRIEVAL_METRICS.items(), strict=True):
        cells = row[-6:]
        if not aligned(cells, centers, 13):
            return []
        for group, name, selected in (("Semantic", "semantic-only", cells[:3]), ("Hybrid", "hybrid", cells[3:])):
            raw, enriched, marked = (word.text for word in selected)
            change = SIGNED_CHANGE.fullmatch(marked[1:-1]) if marked.startswith("(") and marked.endswith(")") else None
            if (not DECIMAL.fullmatch(raw) or not DECIMAL.fullmatch(enriched) or not change
                    or Decimal(raw) == Decimal(enriched) or (change[1].startswith("+") != (Decimal(enriched) > Decimal(raw)))):
                return []
            scope = repeated if name == "hybrid" else None
            titles = (f"{group} / Raw", f"{group} / Enr.", f"{group} / ∆")
            columns = [Column(Header(title, metric_name, cell.x, depth, metric_unit), DATASET, source_label=title,
                              duplicate_scope=scope if metric_name == metric else None)
                       for title, metric_name, cell, metric_unit in zip(
                           titles, (metric, metric, f"{metric} relative change"), selected, (unit, unit, "%"), strict=True,
                       )]
            names = (reference.rows[0][0], reference.rows[1][0]) if scope else (RAW_KB, ENRICHED_KB)
            for config, start, end in ((names[0], 0, 1), (names[1], 1, 3)):
                tables.append(Table(page, label, caption.removesuffix(" Semantic Hybrid"), len(tables), columns[start:end],
                                    [(printed, [raw, enriched, change[1]][start:end])], row_section=group, config_name=config,
                                    top_k=depth, top_k_evidence=(page, match[1]), retrieval_method=name,
                                    value_bboxes=normalized_boxes(page, selected[start:end])))
    return tables


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
