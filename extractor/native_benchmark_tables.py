"""Read benchmark tables using positioned text and their printed captions."""

import re
from itertools import pairwise
from decimal import Decimal
from statistics import mean

from failure_rates import visual_rows
from pdf_pages import Page, Word
from rag_tables import Column, Header, NATIVE_CAPTION, Table, column_boundaries
from retrieval_tables import source_statement


TABLE_CAPTION = re.compile(r"Table\s+[A-Z]?\d+:\s+", re.I)
XRAG_CAPTION = re.compile(r"Statistics of XRAG question types in monolin", re.I)
XRAG_ACCURACY_CAPTIONS = {
    "bounds": re.compile(r"LLM QA accuracy in answering XRAG ques", re.I),
    "monolingual": re.compile(r"LLM QA accuracy in the XRAG monolingual", re.I),
    "multilingual": re.compile(r"LLM QA accuracy in the XRAG multilingual", re.I),
}
XRAG_ACCURACY_SCALE = re.compile(r"we report each model[’']s accuracy\s*\(%\)", re.I)
XRAG_CONTROL_PREFIX = re.compile(r"Controlled analysis of (GPT-4o|Claude Sonnet 3\.5|Mistral-large|Command-R\+) on\b")
XRAG_MULTI_BASE = "XRAG multilingual retrieval"
XRAG_EN_BASE = "English monolingual RAG baseline"
XRAG_MULTI_AVERAGE = "macro average over four language pairs"
XRAG_MODELS = {
    "GPT-4o", "Claude Sonnet 3.5", "Claude 3.5", "Mistral-Large", "Mistral-large", "Command-R+", "Nova-Pro",
}
JAPANESE_STATS = re.compile(r"Statistics of the datasets\.\s+#Doc,\s+#QA,", re.I)
JAPANESE_TYPES = re.compile(r"Number of examples and percentages by\b", re.I)
CHARACTER_UNIT = re.compile(r"\bdocument lengths\s*\(in char-?\s*acters\)", re.I)
JAPANESE_DATASETS = ("Wikipedia", "News", "ProductReview", "CompanyRules")
JAPANESE_STATS_HEADERS = {
    "#Doc": "Documents", "#QA": "QA pairs", "minL": "Minimum document length",
    "medL": "Median document length", "maxL": "Maximum document length",
}
JAPANESE_QUESTION_TYPES = ("Extracting", "Reasoning", "Computing", "Information Integrating")
GROUPED_INTEGER = re.compile(r"\d+|\d{1,3}(?:,\d{3})+")
PAREN_PERCENT = re.compile(r"\((\d+\.\d+)%\)")
SYNTHETIC_CONSTRUCTION = "Synthetic corpus construction"
RECALL_PREFIX = re.compile(r"Retrieval performance comparison on the\b", re.I)
RECALL_CAPTION = re.compile(r"Retrieval performance comparison on the\s+MMLongBench-Doc dataset\.", re.I)
ABLATION_CAPTION = re.compile(r"Ablation study on Bayesian components\.", re.I)
GENERATOR_CAPTIONS = {
    "MMLongBench-Doc": re.compile(r"Performance comparison with different generator models on MMLongBench-Doc\.", re.I),
    "DocBench": re.compile(r"Performance comparison on DocBench with different generators\.", re.I),
}
FINAL_GENERATOR = re.compile(r"\bwe employ\s+([\w.-]+)\s+as the final response generator across all experi-?\s*ments\b", re.I)
GENERATOR_NAME = re.compile(r"[A-Za-z][A-Za-z\d]*(?:[-.][A-Za-z\d]+)+")
DOCUMENT_DOMAINS = tuple("Reports Tutorials Academic Guidebooks Brochures Industry Financial Overall".split())
DOCBENCH_GROUPS = {
    "Aca.": "Academia", "Fin.": "Finance", "Gov.": "Government", "Law.": "Law", "News": "News",
    "Text": "Text", "Meta": "Meta", "MM.": "Multimodal", "Unans.": "Unanswerable", "Overall": "Overall",
}
ABLATION_SETTINGS = {
    "Embedding-RAG": ("-", "-"),
    "BayesRAG (Linear Fusion)": ("Linear Weighted", "Graph"),
    "BayesRAG (Layout Prior)": ("Dempster-Shafer", "Layout"),
    "BayesRAG (Full)": ("Dempster-Shafer", "Graph"),
}
REPEATED_BAYES_SCORES = "BayesRAG full configuration, MMLongBench-Doc"
XRAG_ROW_METRICS = {
    "Example Number": ("Number of examples", None),
    "Aggregation": ("Aggregation questions", "Question"),
    "Comparison": ("Comparison questions", "Question"),
    "Multi-hop": ("Multi-hop questions", "Question"),
    "Set": ("Set questions", "Question"),
    "Original": ("Original answers", "Answer"),
    "Corrected": ("Human-corrected answers", "Answer"),
}
LANGUAGES = "De Es Zh Ar"
UNSIGNED = re.compile(r"\d+")
PERCENT_VALUE = re.compile(r"(?:100(?:\.0+)?|[0-9]{1,2}(?:\.\d+)?)")
RECALL_HEADER = re.compile(r"Recall@(\d+)")


def words_text(words: list[Word]) -> str:
    return " ".join(word.text for word in words)


def normalized_boxes(page: Page, cells: list[Word]) -> tuple:
    if not page.size:
        return ()
    width, height = page.size
    return (tuple((word.x0 / width, word.y0 / height, word.x1 / width, word.y1 / height)
                  for word in cells),)


def positioned_caption(rows: list[list[Word]], label: str, prefix: re.Pattern[str]) -> Word | None:
    matches = []
    for row in rows:
        for start, word in enumerate(row):
            if word.text.casefold() != "table":
                continue
            line = words_text(row[start:])
            locator = TABLE_CAPTION.match(line)
            if not locator or locator[0].rstrip(" :").casefold() != label.casefold():
                continue
            if prefix.match(line[locator.end():]):
                matches.append(word)
    return matches[0] if len(matches) == 1 else None


def body_above(rows: list[list[Word]], caption: Word, left: float, right: float, height: float) -> list[list[Word]]:
    return [cell_row for row in rows
            if (cell_row := [word for word in row if caption.y - height < word.y < caption.y - 3
                             and left < word.x0 and word.x < right])]


def aligned(cells: list[Word], centers: list[float], tolerance: float) -> bool:
    return len(cells) == len(centers) and all(abs(word.x - center) <= tolerance
                                             for word, center in zip(cells, centers, strict=True))


def read_native_benchmark_tables(pages: list[Page], known: list[Table]) -> list[Table]:
    occupied = {(table.page.number, table.label.casefold()) for table in known}
    tables = []
    for page in pages:
        if page.verification != "text_match" or not page.words:
            continue
        rows = visual_rows(page.words)
        for line in page.text.splitlines():
            for locator in NATIVE_CAPTION.finditer(line):
                label = locator[1]
                key = page.number, label.casefold()
                if key in occupied:
                    continue
                text = line[locator.end():].lstrip()
                parsed = []
                if XRAG_CAPTION.match(text) and (anchor := positioned_caption(rows, label, XRAG_CAPTION)):
                    parsed = xrag_counts(page, label, rows, anchor)
                for mode, prefix in XRAG_ACCURACY_CAPTIONS.items():
                    if prefix.match(text) and (anchor := positioned_caption(rows, label, prefix)):
                        scale = source_statement([candidate for candidate in pages if candidate.number <= page.number],
                                                 XRAG_ACCURACY_SCALE)
                        if scale:
                            parsed = xrag_accuracy(page, label, rows, anchor, mode)
                if control := XRAG_CONTROL_PREFIX.match(text):
                    anchor = positioned_caption(rows, label, XRAG_CONTROL_PREFIX)
                    if anchor and source_statement([candidate for candidate in pages if candidate.number <= page.number],
                                                   XRAG_ACCURACY_SCALE):
                        parsed = xrag_controlled(page, label, rows, anchor, control[1])
                if JAPANESE_STATS.match(text) and (anchor := positioned_caption(rows, label, JAPANESE_STATS)):
                    parsed = japanese_stats(page, label, rows, anchor)
                if JAPANESE_TYPES.match(text) and (anchor := positioned_caption(rows, label, JAPANESE_TYPES)):
                    parsed = japanese_question_types(page, label, rows, anchor)
                if RECALL_PREFIX.match(text) and (anchor := positioned_caption(rows, label, RECALL_PREFIX)):
                    parsed = recall_tables(page, label, rows, anchor)
                if ABLATION_CAPTION.match(text) and (anchor := positioned_caption(rows, label, ABLATION_CAPTION)):
                    setup = source_statement([candidate for candidate in pages if candidate.number < page.number],
                                             FINAL_GENERATOR)
                    parsed = ablation_tables(page, label, rows, anchor, setup)
                for benchmark, prefix in GENERATOR_CAPTIONS.items():
                    if prefix.match(text) and (anchor := positioned_caption(rows, label, prefix)):
                        parsed = generator_tables(page, label, rows, anchor, benchmark, prefix.match(text)[0])
                if parsed:
                    tables.extend(parsed)
                    occupied.add(key)
    return tables


def xrag_counts(page: Page, label: str, rows: list[list[Word]], footer: Word) -> list[Table]:
    grid = body_above(rows, footer, footer.x0 - 4, float("inf"), 115)
    headings = [index for index, row in enumerate(grid)
                if words_text(row) == "Monolingual Retrieval Multilingual Retrieval"]
    if len(headings) != 1:
        return []
    start = headings[0] + 1
    if start >= len(grid):
        return []
    langs = grid[start]
    if words_text(langs) != "De / Es / Zh / Ar De Es Zh Ar":
        return []
    centers = [langs[3].x, *(word.x for word in langs[7:])]
    parsed = []
    seen_groups = []
    for row in grid[start + 1:]:
        cells = row[-len(centers):]
        name = words_text(row[:-len(centers)])
        if words_text(row) in {"Question", "Answer"}:
            seen_groups.append(words_text(row))
            continue
        if (name not in XRAG_ROW_METRICS or not aligned(cells, centers, 2)
                or any(not UNSIGNED.fullmatch(word.text) for word in cells)):
            return []
        parsed.append((name, cells))
    if [name for name, _ in parsed] != list(XRAG_ROW_METRICS) or seen_groups != ["Question", "Answer"]:
        return []
    for total, *groups in zip(*[[int(cell.text) for cell in row] for _, row in parsed], strict=True):
        if sum(groups[:4]) != total or sum(groups[4:]) != total:
            return []

    source_columns = ("Monolingual Retrieval / De / Es / Zh / Ar",
                      *(f"Multilingual Retrieval / {lang}" for lang in LANGUAGES.split()))
    caption = f"{label}: Statistics of XRAG question types"
    tables = []
    for name, cells in parsed:
        metric, section = XRAG_ROW_METRICS[name]
        columns = [Column(Header(source, metric, center, None, None), f"XRAG / {source}", source_label=source)
                   for source, center in zip(source_columns, centers, strict=True)]
        tables.append(Table(page, label, caption, int(cells[0].y), columns, [(name, [cell.text for cell in cells])],
                            row_section=section, config_name="XRAG benchmark construction",
                            value_bboxes=normalized_boxes(page, cells)))
    return tables


def recall_tables(page: Page, label: str, rows: list[list[Word]], footer: Word) -> list[Table]:
    grid = body_above(rows, footer, footer.x0 - 4, float("inf"), 85)
    candidates = [(index, row[1:6]) for index, row in enumerate(grid)
                  if len(row) >= 6 and row[0].text == "Method"
                  and all(RECALL_HEADER.fullmatch(word.text) for word in row[1:6])]
    if len(candidates) != 1:
        return []
    start, headings = candidates[0]
    centers = [word.x for word in headings]
    bounds = column_boundaries(centers)
    caption_lines = [[word for word in row if footer.x0 - 4 < word.x0 and word.x < bounds[-1]]
                     for row in rows if footer.y - 2 < row[0].y < footer.y + 20]
    caption_words = " ".join(words_text(line) for line in caption_lines)
    locator = TABLE_CAPTION.match(caption_words)
    if not locator or not (caption := RECALL_CAPTION.match(caption_words[locator.end():])):
        return []
    columns = [Column(Header(word.text, "Page-evidence Recall", word.x, int(word.text.split("@")[1]), None),
                      "MMLongBench-Doc / Overall", direction="higher_is_better", source_label=word.text,
                      reported_scale="percent") for word in headings]
    parsed = []
    for grid_row in grid[start + 1:]:
        row = [word for word in grid_row if word.x < bounds[-1]]
        if not row:
            continue
        cells = row[-len(headings):]
        method = words_text(row[:-len(headings)])
        if (method not in {"Vector Retrieval", "BayesRAG"} or not aligned(cells, centers, 4)
                or any(not PERCENT_VALUE.fullmatch(cell.text) for cell in cells)):
            return []
        parsed.append((method, cells))
    if [method for method, _ in parsed] != ["Vector Retrieval", "BayesRAG"]:
        return []
    return [Table(page, label, f"{label}: {caption[0]}", int(cells[0].y), columns,
                  [(method, [cell.text for cell in cells])], value_bboxes=normalized_boxes(page, cells))
            for method, cells in parsed]


def score_table(page: Page, label: str, caption: str, benchmark: str, headings: list[Word],
                method: str, cells: list[Word], generator: str | None,
                generator_evidence: tuple[Page, str] | None = None) -> Table:
    metric = "LLM accuracy" if benchmark == "DocBench" else "LLM Score"
    config = "BayesRAG" if method in {"BayesRAG (ours)", "BayesRAG (Full)"} else method
    scope = REPEATED_BAYES_SCORES if benchmark == "MMLongBench-Doc" and config == "BayesRAG" and generator else None
    columns = []
    for heading in headings:
        source = f"{generator} / {heading.text}" if generator and not generator_evidence else heading.text
        group = DOCBENCH_GROUPS[heading.text] if benchmark == "DocBench" else heading.text
        columns.append(Column(
            Header(heading.text, metric, heading.x, None, None), f"{benchmark} / {group}",
            generator=generator, generator_locator="Generator Model" if generator and not generator_evidence else None,
            generator_evidence=generator_evidence, direction="higher_is_better",
            source_label=source, reported_scale="percent", duplicate_scope=scope,
        ))
    return Table(page, label, f"{label}: {caption}", int(cells[0].y), columns,
                 [(method, [cell.text for cell in cells])], config_name=config, value_bboxes=normalized_boxes(page, cells))


def ablation_tables(page: Page, label: str, rows: list[list[Word]], footer: Word,
                    setup: tuple[Page, re.Match[str]] | None) -> list[Table]:
    grid = body_above(rows, footer, footer.x0 - 4, float("inf"), 100)
    expected = ("Method", "Likelihood", "Prior", *DOCUMENT_DOMAINS)
    head = [index for index, row in enumerate(grid) if tuple(word.text for word in row) == expected]
    if len(head) != 1:
        return []
    header = grid[head[0]]
    headings = header[3:]
    centers = [heading.x for heading in headings]
    prior_left = (header[1].x + header[2].x) / 2
    generator = setup[1][1] if setup else None
    evidence = (setup[0], setup[1][0]) if setup else None
    parsed = {}
    for row in grid[head[0] + 1:]:
        if words_text(row) == "Inference":
            continue
        scores = row[-len(headings):]
        context = row[:-len(headings)]
        mark = next((index for index, word in enumerate(context) if word.text in {"%", "!", "✓", "✗", "✔", "✘"}), None)
        if mark is None or not aligned(scores, centers, 4) or any(not PERCENT_VALUE.fullmatch(w.text) for w in scores):
            return []
        method = words_text(context[:mark])
        likelihood = words_text([word for word in context[mark + 1:] if word.x < prior_left])
        prior = words_text([word for word in context[mark + 1:] if word.x >= prior_left])
        if ABLATION_SETTINGS.get(method) != (likelihood, prior) or method in parsed:
            return []
        if (context[mark].text in {"%", "✗", "✘"}) != (method == "Embedding-RAG"):
            return []
        parsed[method] = scores
    if list(parsed) != list(ABLATION_SETTINGS):
        return []
    return [score_table(page, label, "Ablation study on Bayesian components.", "MMLongBench-Doc", headings,
                        method, scores, generator, evidence) for method, scores in parsed.items()]


def generator_tables(page: Page, label: str, rows: list[list[Word]], footer: Word,
                     benchmark: str, caption: str) -> list[Table]:
    grid = body_above(rows, footer, footer.x0 - 4, float("inf"), 120)
    docbench = benchmark == "DocBench"
    header_labels = ("Overall",) if docbench else DOCUMENT_DOMAINS
    expected = ("Method", *header_labels, "Generator", "Model")
    matching = [index for index, row in enumerate(grid) if tuple(word.text for word in row) == expected]
    if len(matching) != 1:
        return []
    header_at = matching[0]
    header = grid[header_at]
    headings = header[1:-2]
    if docbench:
        if header_at + 1 == len(grid):
            return []
        header_at += 1
        subslices = grid[header_at]
        if tuple(word.text for word in subslices) != tuple(DOCBENCH_GROUPS)[:-1]:
            return []
        headings = [*subslices, *headings]
    centers = [heading.x for heading in headings]
    generator_left = (header[-3].x1 + header[-2].x0) / 2
    body = grid[header_at + 1:]
    if len(body) < 6 or len(body) % 3:
        return []
    parsed = []
    seen_generators = set()
    for start in range(0, len(body), 3):
        first, model_row, second = body[start:start + 3]
        if len(model_row) != 1 or model_row[0].x < generator_left:
            return []
        generator = model_row[0].text
        if generator in seen_generators or not GENERATOR_NAME.fullmatch(generator):
            return []
        seen_generators.add(generator)
        for row, method in zip((first, second), ("RAGAnything", "BayesRAG (ours)"), strict=True):
            if row[-1].x >= generator_left:
                return []
            cells = row[-len(headings):]
            if (words_text(row[:-len(headings)]) != method or not aligned(cells, centers, 4)
                    or any(not PERCENT_VALUE.fullmatch(cell.text) for cell in cells)):
                return []
            parsed.append(score_table(page, label, caption, benchmark, headings, method, cells, generator))
    # Geometric headers, grouped model labels, and values must all fit the same columns.
    if any(left >= right for left, right in pairwise(centers)):
        return []
    return parsed


def xrag_accuracy(page: Page, label: str, rows: list[list[Word]], footer: Word, mode: str) -> list[Table]:
    grid = body_above(rows, footer, footer.x0 - 4, float("inf"), 120)
    if mode == "bounds":
        return xrag_bounds(page, label, grid, rows, footer)

    modes = {
        "monolingual": ("En De Es Zh Ar Avg.", "macro average over four non-English languages"),
        "multilingual": ("En De En Es En Zh En Ar (crossling.)", XRAG_MULTI_AVERAGE),
    }
    query_labels, aggregation = modes[mode]
    heads = [index for index, row in enumerate(grid)
             if words_text(row) == f"Query. Lang. {query_labels}"]
    if len(heads) != 1 or heads[0] == 0:
        return []
    header_at = heads[0]
    headings = grid[header_at][2:]
    languages = LANGUAGES.split()
    if mode == "monolingual":
        if words_text(grid[header_at - 1]) != "Doc. Lang. En":
            return []
        conditions = [
            (f"En documents / {heading.text} queries", f"XRAG / monolingual retrieval / {heading.text}",
             "XRAG monolingual retrieval", "mean over questions") for heading in headings[:-1]
        ]
        average_at = (1, 2, 3, 4)
    else:
        docs = " ".join(f"En+En En+{lang}" for lang in languages)
        if (header_at < 2 or words_text(grid[header_at - 2]) != f"Doc. Lang. {docs} Avg."
                or words_text(grid[header_at - 1]) != LANGUAGES):
            return []
        conditions = []
        for lang in languages:
            conditions.extend((
                (f"En+En{lang} / En queries", f"XRAG / multilingual retrieval / En counterpart of {lang}",
                 XRAG_EN_BASE, "mean over questions"),
                (f"En+{lang} / {lang} queries", f"XRAG / multilingual retrieval / {lang}",
                 XRAG_MULTI_BASE, "mean over questions"),
            ))
        average_at = (1, 3, 5, 7)
    mean_label = "Avg." if mode == "monolingual" else "Avg. (crossling.)"
    conditions.append((mean_label, f"XRAG / {mode} retrieval / average", f"XRAG {mode} retrieval", aggregation))
    parsed = xrag_score_rows(grid[header_at + 1:], [word.x for word in headings])
    if not parsed:
        return []
    for _, cells in parsed:
        # The paper publishes averaged scores independently of its rounded component scores.
        displayed_average = mean(float(cells[index].text) for index in average_at)
        if abs(displayed_average - float(cells[-1].text)) > 0.06:
            return []
    return xrag_accuracy_tables(page, label, f"LLM QA accuracy in the XRAG {mode} retrieval setting",
                                parsed, conditions)


def xrag_bounds(page: Page, label: str, grid: list[list[Word]], rows: list[list[Word]], footer: Word) -> list[Table]:
    headers = [index for index, row in enumerate(grid) if words_text(row) == "No Retrieval Oracle Retrieval"]
    if len(headers) != 1:
        return []
    after = [w for row in rows for w in row
             if footer.y <= w.y <= footer.y + 43 and w.x0 >= footer.x0 - 4]
    if "Questions and supporting articles are in English" not in words_text(after):
        return []
    head = headers[0]
    headings = grid[head]
    centers = [(headings[index].x0 + headings[index + 1].x1) / 2 for index in (0, 2)]
    parsed = xrag_score_rows(grid[head + 1:], centers)
    conditions = [(name, "XRAG / monolingual retrieval / En", name, None)
                  for name in ("No Retrieval", "Oracle Retrieval")]
    return xrag_accuracy_tables(page, label, "LLM QA accuracy in answering XRAG questions without or with supporting articles",
                                parsed, conditions)


def xrag_score_rows(grid: list[list[Word]], centers: list[float]) -> list[tuple[str, list[Word]]]:
    parsed = []
    seen = set()
    for line in grid:
        generator = words_text(line[:-len(centers)])
        cells = line[-len(centers):]
        if (generator not in XRAG_MODELS or generator in seen or not aligned(cells, centers, 5)
                or any(not PERCENT_VALUE.fullmatch(cell.text) for cell in cells)):
            return []
        seen.add(generator)
        parsed.append((generator, cells))
    return parsed if len(parsed) >= 2 else []


def xrag_accuracy_tables(page: Page, label: str, caption: str, parsed: list[tuple[str, list[Word]]],
                         conditions: list[tuple[str, str, str, str | None]], row: str | None = None) -> list[Table]:
    tables = []
    for generator, cells in parsed:
        groups = {}
        for cell, (source, dataset, config, aggregation) in zip(cells, conditions, strict=True):
            scope = "XRAG multilingual retrieval and English counterparts" if config in {XRAG_MULTI_BASE, XRAG_EN_BASE} else None
            column = Column(Header(source, "QA accuracy", cell.x, None, None),
                            dataset, aggregation=aggregation, generator=generator, source_label=source,
                            reported_scale="percent", direction="higher_is_better", duplicate_scope=scope,
                            generator_locator="Generator" if row else None)
            groups.setdefault(config, []).append((column, cell))
        for config, values in groups.items():
            columns, words = zip(*values)
            tables.append(Table(page, label, f"{label}: {caption}", int(words[0].y),
                                list(columns), [(row or generator, [w.text for w in words])],
                                config_name=config, value_bboxes=normalized_boxes(page, words)))
    return tables


def xrag_controlled(page: Page, label: str, rows: list[list[Word]], footer: Word, caption_model: str) -> list[Table]:
    language_labels = [f"En+{lang}" for lang in LANGUAGES.split()]
    heading = [*language_labels, "Avg."]
    generator = "Claude 3.5" if caption_model == "Claude Sonnet 3.5" else caption_model
    grid = body_above(rows, footer, footer.x0 - 4, float("inf"), 115)
    nwords = len(generator.split())
    headers = [index for index, line in enumerate(grid)
               if words_text(line[:nwords]) == generator and [word.text for word in line[nwords:nwords + 5]] == heading]
    if len(headers) != 1:
        return []
    header_at = headers[0]
    centers = [word.x for word in grid[header_at][nwords:nwords + 5]]
    right = column_boundaries(centers)[-1]
    after = " ".join(words_text([w for w in line if footer.x0 - 4 < w.x0 and w.x < right])
                     for line in rows if footer.y - 2 <= line[0].y <= footer.y + 29)
    if not re.search(r"\bmul(?:-\s*)?tilingual\s+retrieval\s+setting\s+of\s+XRAG\b", after):
        return []
    conditions = {
        "XRAG-MultiR": XRAG_MULTI_BASE,
        "+EQ": "XRAG-MultiR with English questions",
        "+EQ, +ES": "XRAG-MultiR with English questions and supporting articles",
        "MonoRAG": XRAG_EN_BASE,
    }
    parsed = {}
    for line in grid[header_at + 1:]:
        words = [w for w in line if w.x < right]
        if not words:
            continue
        cells = words[-5:]
        row = words_text(words[:-5])
        key = row.replace(", ES", ", +ES")
        if (key not in conditions or key in parsed or not aligned(cells, centers, 4)
                or any(not PERCENT_VALUE.fullmatch(cell.text) for cell in cells)):
            return []
        if abs(mean(float(cell.text) for cell in cells[:-1]) - float(cells[-1].text)) > .06:
            return []
        parsed[key] = row, cells
    if list(parsed) != list(conditions):
        return []
    tables = []
    for key, (row, cells) in parsed.items():
        config = conditions[key]
        pairs = []
        for source, lang in zip(language_labels, LANGUAGES.split(), strict=True):
            data = f"En counterpart of {lang}" if key == "MonoRAG" else lang
            pairs.append((source, f"XRAG / multilingual retrieval / {data}", config, "mean over questions"))
        average = "English counterparts average" if key == "MonoRAG" else "average"
        pairs.append(("Avg.", f"XRAG / multilingual retrieval / {average}", config, XRAG_MULTI_AVERAGE))
        tables.extend(xrag_accuracy_tables(page, label, f"Controlled analysis of {caption_model} on XRAG",
                                           [(generator, cells)], pairs, row=row))
    return tables


def japanese_stats(page: Page, label: str, rows: list[list[Word]], footer: Word) -> list[Table]:
    if not source_statement([page], CHARACTER_UNIT):
        return []
    grid = body_above(rows, footer, footer.x0 - 4, float("inf"), 90)
    heads = [index for index, row in enumerate(grid)
             if tuple(word.text for word in row[:5]) == tuple(JAPANESE_STATS_HEADERS)]
    if len(heads) != 1:
        return []
    header_at = heads[0]
    headings = grid[header_at][:5]
    centers = [word.x for word in headings]
    right = column_boundaries(centers)[-1]
    parsed = {}
    for line in grid[header_at + 1:]:
        words = [word for word in line if word.x < right]
        if not words:
            continue
        dataset, *cells = words
        if (dataset.text not in JAPANESE_DATASETS or dataset.text in parsed or not aligned(cells, centers, 5)
                or any(not GROUPED_INTEGER.fullmatch(cell.text) for cell in cells)):
            return []
        counts = [int(cell.text.replace(",", "")) for cell in cells]
        if not all(counts[:2]) or counts[2:] != sorted(counts[2:]):
            return []
        parsed[dataset.text] = cells
    if tuple(parsed) != JAPANESE_DATASETS:
        return []
    return [Table(
        page, label, f"{label}: Statistics of the datasets.", int(cells[0].y),
        [Column(
            Header(w.text, JAPANESE_STATS_HEADERS[w.text], w.x, None,
                   "characters" if w.text.endswith("L") else None),
            dataset, source_label=w.text, grouped_digits=True,
            duplicate_scope=(page.number, "synthetic benchmark QA pair totals") if w.text == "#QA" else None,
        ) for w in headings],
        [(dataset, [cell.text for cell in cells])], config_name=SYNTHETIC_CONSTRUCTION,
        value_bboxes=normalized_boxes(page, cells),
    ) for dataset, cells in parsed.items()]


def japanese_question_types(page: Page, label: str, rows: list[list[Word]], footer: Word) -> list[Table]:
    grid = body_above(rows, footer, footer.x0 - 4, float("inf"), 90)
    heads = [index for index, row in enumerate(grid)
             if len(row) >= 6 and words_text(row[:2]) == "#QA →"
             and all(UNSIGNED.fullmatch(cell.text) for cell in row[2:6])]
    if len(heads) != 1 or heads[0] < 3:
        return []
    header_at = heads[0]
    totals = grid[header_at][2:6]
    centers = [cell.x for cell in totals]
    bounds = column_boundaries(centers)
    names = [text for row in grid[max(0, header_at - 6):header_at]
             if (text := words_text([word for word in row if bounds[0] < word.x < bounds[-1]]))]
    if names != ["Product Company", "Wikipedia News", "Review Rules"]:
        return []
    counts = [int(total.text) for total in totals]
    if not all(counts):
        return []
    table_caption = f"{label}: Number of examples and percentages by question type."
    headings = ("Wikipedia", "News", "Product Review", "Company Rules")
    totals_columns = [
        Column(Header("#QA", "QA pairs", total.x, None, None), dataset, source_label=f"{heading} / #QA",
               duplicate_scope=(page.number, "synthetic benchmark QA pair totals"))
        for total, heading, dataset in zip(totals, headings, JAPANESE_DATASETS, strict=True)
    ]
    tables = [Table(page, label, table_caption, int(totals[0].y), totals_columns,
                    [("#QA →", [word.text for word in totals])], config_name=SYNTHETIC_CONSTRUCTION,
                    value_bboxes=normalized_boxes(page, totals))]
    categories = []
    previous = ""
    for line in grid[header_at + 1:]:
        words = [word for word in line if word.x < bounds[-1]]
        if not words:
            continue
        if words_text(words) == "Information":
            previous = "Information "
            continue
        category = previous + words[0].text
        previous = ""
        if category not in JAPANESE_QUESTION_TYPES or len(words) != 9:
            return []
        columns, values = [], []
        for index, (dataset, total, heading) in enumerate(zip(JAPANESE_DATASETS, counts, headings, strict=True)):
            count, share = words[2 * index + 1:2 * index + 3]
            if not bounds[index] < count.x < share.x < bounds[index + 1] or not UNSIGNED.fullmatch(count.text):
                return []
            ratio = PAREN_PERCENT.fullmatch(share.text)
            if not ratio or int(count.text) > total:
                return []
            decimal = Decimal(ratio[1])
            if (Decimal(count.text) * 100 / total).quantize(decimal) != decimal:
                return []
            columns.extend((
                Column(Header("Count", "Question count by type", count.x, None, None), dataset,
                       metric_variant=category, source_label=f"{heading} / Count"),
                Column(Header("Percentage", "Question proportion by type", share.x, None, "%"), dataset,
                       metric_variant=category, source_label=f"{heading} / Percentage",
                       sample_size=total, sample_size_label="#QA"),
            ))
            values.extend((count.text, share.text[1:-1]))
        categories.append(category)
        tables.append(Table(page, label, table_caption, int(words[0].y), columns, [(category, values)],
                            config_name=SYNTHETIC_CONSTRUCTION, value_bboxes=normalized_boxes(page, words[1:])))
    return tables if tuple(categories) == JAPANESE_QUESTION_TYPES and not previous else []
