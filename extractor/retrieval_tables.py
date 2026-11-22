"""Read retrieval comparisons with a caption metric and grouped dataset columns."""

from dataclasses import replace
import re

from pdf_pages import Page
from rag_tables import CAPTION, Column, Header, Table, column_boundaries


RANKING_CAPTION = re.compile(
    r"\bPerformance of (?P<retriever>[\w.-]+) with Various Rerankers and LCR Enhancement\. "
    r"(?P<metric>NDCG@(?P<cutoff>\d+)) scores for the (?P=retriever) retriever\b", re.I,
)
RANKING_HEADINGS = ("NQ", "TOUC", "SCID", "NFCO", "DBPE", "FEVE", "AVG", "DL19", "DL20", "AVG")
RANKING_DATASETS = (
    "NaturalQuestions", "Touché", "SciDocs", "NFCorpus", "DBpedia-Entity", "FEVER",
    "Mean over six BEIR test datasets", "TREC Deep Learning 2019", "TREC Deep Learning 2020",
    "Mean over TREC DL19 and DL20 test datasets",
)
RANKING_SECTION = re.compile(r"^(?:Retriever-Only|(?:Pre-trained LLM|Fine-tuned Transformer) Based Rerankers)$", re.I)
RANKING_METHOD = re.compile(r"[\w-]+(?:\s+[\w-]+){0,3}")
ENHANCEMENTS = {"+LCR": "+ LCR", "+LCR w/o QT": "+ LCR w/o query threshold"}
RETRIEVED_DOCUMENTS = re.compile(r"\bthe retriever fetches top[- ](\d+) documents\b", re.I)
RERANKER_MODEL = re.compile(r"\bLLM-based rerankers,\s*we use ([\w.-]+)\b", re.I)
TEST_SETS = re.compile(r"\btest sets and relevance annotations provided by these benchmarks\b", re.I)
BENCHMARK_NAME = re.compile(r"\bRAG-\s*Multi-Corpus\s+benchmark\b", re.I)
AVERAGE_HEADING = re.compile(r"Avg ((?:Recall|Precision|NDCG)@(\d+)|MRR)")
AVERAGE_LAYOUT = re.compile(r"^\s*(?:(Organization|Query Type)\s{2,})?Method\s{2,}.*\s{2,}Count\s*$")
AVERAGE_CAPTION = re.compile(r"\b(?:Overall Retrieval Performance|Retrieval Performance by (?:Organization|Query Type))\b", re.I)
FRACTION = re.compile(r"(?:0|0?\.\d+|1(?:\.0+)?)")
CHUNKING_METHODS = {"Baseline": "Agentic Chunking", "W-RAC": "W-RAC"}
EFFICIENCY_CAPTION = re.compile(r"\bToken and Runtime Comparison by Organization\b", re.I)
EFFICIENCY_HEADINGS = (
    ("Total Length", "Content length", None, "organization total"),
    ("Input Tokens", "Input tokens", "tokens", "organization total"),
    ("Output Tokens", "Output tokens", "tokens", "organization total"),
    ("Time (s)", "Processing time", "s", "organization total"),
    ("Total Files", "Processed files", "files", "organization total"),
    ("Avg Input Tokens", "Mean input tokens per file", "tokens/file", "per-file mean"),
    ("Avg Output Tokens", "Mean output tokens per file", "tokens/file", "per-file mean"),
    ("Avg Time (s)", "Mean time per file", "s", "per-file mean"),
    ("P90 Time (s)", "P90 time", "s", "per-file percentile"),
    ("P95 Time (s)", "P95 time", "s", "per-file percentile"),
)
EFFICIENCY_NUMBER = re.compile(r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
CHARACTER_LENGTH = re.compile(r"\btotal content length of [\d,]+ characters\b", re.I)


def source_statement(pages: list[Page], pattern: re.Pattern[str]) -> tuple[Page, re.Match[str]] | None:
    for page in pages:
        for text in (page.reading_text, page.text):
            if match := pattern.search(text):
                return page, match
    return None


def following_caption(lines: list[str], index: int, pattern: re.Pattern[str]) -> tuple[int, str, str] | None:
    for i in range(index + 1, min(index + 31, len(lines))):
        if match := CAPTION.match(lines[i]):
            return (i, match[1], " ".join(lines[i].split())) if pattern.search(lines[i]) else None
    return None


def read_retrieval_tables(pages: list[Page], tables: list[Table]) -> list[Table]:
    occupied = {(table.page.number, table.label.casefold()) for table in tables}
    found = []
    for page in pages:
        if page.verification != "text_match":
            continue
        lines = page.text.splitlines()
        for index, line in enumerate(lines):
            if average := AVERAGE_LAYOUT.fullmatch(line):
                found.extend(average_rows(page, lines, index, average, occupied, pages))
            if line.split() == ["Organization", "Method"]:
                found.extend(efficiency_rows(page, lines, index, occupied, pages))
            cells = list(re.finditer(r"\S+", line))
            if tuple(cell[0] for cell in cells) != ("Method", *RANKING_HEADINGS):
                continue
            if index < 2 or lines[index - 1].split() != ["BEIR", "TREC"]:
                continue
            previous = [i for i in range(max(0, index - 10), index) if CAPTION.match(lines[i])]
            if not previous:
                continue
            caption_at = previous[-1]
            label = CAPTION.match(lines[caption_at])[1]
            caption = " ".join(" ".join(lines[caption_at:index - 1]).split())
            metric = RANKING_CAPTION.search(caption)
            if not metric or (page.number, label.casefold()) in occupied:
                continue
            setup = [candidate for candidate in pages if candidate.number < page.number]
            found.extend(ranking_rows(page, index, lines, label, caption, metric, cells[1:], setup))
    return found


def ranking_rows(page: Page, index: int, lines: list[str], label: str, caption: str, metric: re.Match[str],
                 cells: list[re.Match[str]], setup: list[Page]) -> list[Table]:
    retrieved = source_statement(setup, RETRIEVED_DOCUMENTS)
    generator = source_statement(setup, RERANKER_MODEL)
    split = source_statement(setup, TEST_SETS)
    centers = [(cell.start() + cell.end()) / 2 for cell in cells]
    boundaries = column_boundaries(centers)
    columns = []
    for position, (heading, name, center) in enumerate(zip(RANKING_HEADINGS, RANKING_DATASETS, centers, strict=True)):
        group = "BEIR" if position < 7 else "TREC"
        header = Header(heading, metric["metric"], center, int(metric["cutoff"]), None)
        columns.append(Column(header, name, "mean across datasets" if heading == "AVG" else None,
                              direction="higher_is_better", source_label=f"{group} / {heading}",
                              data_split=("benchmark test set with supplied relevance judgments"
                                          if split and heading != "AVG" else None),
                              data_split_evidence=(split[0], split[1][0]) if split and heading != "AVG" else None))
    found = []
    method = None
    section = None
    llm_section = False
    for offset, line in enumerate(lines[index + 1:], start=index + 1):
        row = line.strip()
        if not row:
            break
        if RANKING_SECTION.fullmatch(row):
            method = None
            section = row
            llm_section = row.casefold().startswith("pre-trained llm")
            continue
        tokens = list(re.finditer(r"\S+", line))
        if len(tokens) <= len(columns):
            break
        values = tokens[-len(columns):]
        row = line[:values[0].start()].strip()
        valid = all(FRACTION.fullmatch(value[0])
                    and left <= (value.start() + value.end()) / 2 < right
                    for value, left, right in zip(values, boundaries[:-1], boundaries[1:], strict=True))
        if not valid:
            method = None
            continue
        if enhancement := ENHANCEMENTS.get(row):
            if method is None or (row.endswith("QT") and "without the query threshold" not in caption):
                continue
            name = f"{method} {enhancement}"
            row_section = method
        elif row == "-" and section == "Retriever-Only":
            method = name = row_section = section
        elif RANKING_METHOD.fullmatch(row):
            method = name = row
            row_section = None
        else:
            method = None
            continue
        row_columns = columns
        if generator and (enhancement or llm_section):
            row_columns = [replace(column, generator=generator[1][1],
                                   generator_evidence=(generator[0], generator[1][1])) for column in columns]
        found.append(Table(
            page, label, caption, offset, row_columns, [(row, [value[0] for value in values])],
            row_section=row_section, config_name=name,
            retriever_model=metric["retriever"], top_k=int(retrieved[1][1]) if retrieved else None,
            top_k_evidence=(retrieved[0], retrieved[1][0]) if retrieved else None,
        ))
    return found


def average_rows(page: Page, lines: list[str], index: int, layout: re.Match[str],
                 occupied: set[tuple[int, str]], pages: list[Page]) -> list[Table]:
    headings = list(AVERAGE_HEADING.finditer(layout[0]))
    if {h[1].partition("@")[0] for h in headings} != {"Recall", "Precision", "MRR", "NDCG"}:
        return []
    if len({h[1] for h in headings}) != len(headings):
        return []
    match = following_caption(lines, index, AVERAGE_CAPTION)
    if not match:
        return []
    caption_at, label, caption = match
    if (page.number, label.casefold()) in occupied:
        return []
    if not source_statement([candidate for candidate in pages if candidate.number <= page.number], BENCHMARK_NAME):
        return []
    count_heading = re.search(r"\bCount\s*$", layout[0])
    centers = [(heading.start() + heading.end()) / 2 for heading in [*headings, count_heading]]
    boundaries = column_boundaries(centers)
    found = []
    for offset, line in enumerate(lines[index + 1:caption_at], start=index + 1):
        if not line.strip():
            continue
        tokens = list(re.finditer(r"\S+", line))
        if len(tokens) <= len(centers):
            return []
        values = tokens[-len(centers):]
        if not (values[-1][0].isdecimal() and all(FRACTION.fullmatch(value[0]) for value in values[:-1])):
            return []
        if any(not left <= (value.start() + value.end()) / 2 < right
               for value, left, right in zip(values, boundaries[:-1], boundaries[1:], strict=True)):
            return []
        row = line[:values[0].start()].strip()
        sections = re.split(r"\s{2,}", row)
        if len(sections) != (2 if layout[1] else 1) or sections[-1] not in CHUNKING_METHODS:
            return []
        method = sections[-1]
        group = sections[0] if layout[1] else "Overall"
        dataset = f"RAG-Multi-Corpus / {group}"
        columns = [
            Column(Header(heading[0], heading[1], center, int(heading[2]) if heading[2] else None, None),
                   dataset, "mean over queries", direction="higher_is_better", source_label=heading[0],
                   reported_scale="fraction", sample_size=int(values[-1][0]), sample_size_label="Count")
            for heading, center in zip(headings, centers[:-1], strict=True)
        ]
        found.append(Table(page, label, caption, offset, columns, [(method, [v[0] for v in values[:-1]])],
                           row_section=group if layout[1] else None, config_name=CHUNKING_METHODS[method]))
    return found


def efficiency_rows(page: Page, lines: list[str], index: int, occupied: set[tuple[int, str]],
                    pages: list[Page]) -> list[Table]:
    if not 0 < index < len(lines) - 1:
        return []
    top = list(re.finditer(r"\S+(?: \S+)*(?=\s{2,}|$)", lines[index - 1]))
    bottom = list(re.finditer(r"\S+(?: \S+)*(?=\s{2,}|$)", lines[index + 1]))
    labels = [f"{first[0]} {last[0]}" for first, last in zip(top, bottom)]
    if len(top) != len(bottom) or labels != [label for label, *_ in EFFICIENCY_HEADINGS]:
        return []
    match = following_caption(lines, index, EFFICIENCY_CAPTION)
    if not match:
        return []
    caption_at, label, caption = match
    if (page.number, label.casefold()) in occupied:
        return []
    benchmark = source_statement([candidate for candidate in pages if candidate.number <= page.number], BENCHMARK_NAME)
    characters = source_statement([page], CHARACTER_LENGTH)
    centers = [(first.start() + first.end() + last.start() + last.end()) / 4
               for first, last in zip(top, bottom, strict=True)]
    boundaries = column_boundaries(centers)
    found = []
    for offset, line in enumerate(lines[index + 2:caption_at], start=index + 2):
        if not line.strip():
            continue
        tokens = list(re.finditer(r"\S+", line))
        if len(tokens) <= len(centers):
            return []
        values = tokens[-len(centers):]
        if any(not EFFICIENCY_NUMBER.fullmatch(value[0]) or not left <= (value.start() + value.end()) / 2 < right
               for value, left, right in zip(values, boundaries[:-1], boundaries[1:], strict=True)):
            return []
        sections = re.split(r"\s{2,}", line[:values[0].start()].strip())
        if len(sections) != 2 or sections[-1] not in {"Agentic Chunking", "W-RAC"} or not values[4][0].isdecimal():
            return []
        group, method = sections
        dataset = f"RAG-Multi-Corpus / {group}" if benchmark else group
        columns = [
            Column(Header(printed, name, center, None, "characters" if name == "Content length" and characters else unit),
                   dataset, aggregation, source_label=printed, grouped_digits=True,
                   sample_size=int(values[4][0]), sample_size_label="Total Files")
            for (printed, name, unit, aggregation), center in zip(EFFICIENCY_HEADINGS, centers, strict=True)
        ]
        found.append(Table(page, label, caption, offset, columns, [(method, [v[0] for v in values])], row_section=group))
    return found
