"""Read knowledge-selection benchmarks with mean scores and answer F1 SEM."""

import re

from grouped_scores import native_evidence
from pdf_pages import Page
from rag_tables import Column, Header, Table


HEADER = re.compile(r"^ *(Input Knowledge) +KP +KR +KF1 +(R-L|EM) +F1(?= {3,}| *$)")
FRACTION = r"(?:[01](?:\.\d+)?|\.\d+)"
ROW = re.compile(r"^ *(No|Full|Gold) knowledge +" + " +".join(f"({FRACTION})" for _ in range(5))
                 + rf" +\(± *({FRACTION})\)(?= {{3,}}| *$)", re.I)
GENERATOR_NAME = re.compile(r"(?=.*\d)[\w.-]+(?: [\w.-]+){0,4}")
CAPTION = re.compile(r"^(Table \d+): +([\w.-]+) (response|answer) generation performance", re.I)
KNOWLEDGE_METRICS = re.compile(r"knowledge precision \(KP\),\s*recall \(KR\),\s*and F1 \(KF1\)", re.I)
ANSWER_METRIC = re.compile(r"(?:response ROUGE-L F1 \(R-L\)|answer exact match \(EM\))", re.I)
STANDARD_ERROR = re.compile(r"standard error(?: of(?: the)?)? mean", re.I)
HOT_POT = re.compile(r"first\s+(\d+)\s+examples\s+in\s+the\s+HotpotQA\s+training\s+set", re.I)
WOW = re.compile(r"first\s+(\d+)\s+conversations\s*\((\d+)\s+wizard\s+utterances\)\s+from\s+the\s+[“\"]test\s+seen[”\"]\s+set\s+of\s+WoW", re.I)
VERSIONS = (
    (re.compile(r"^GPT-4o-mini$", re.I), re.compile(r"\b(gpt-4o-mini-\d{4}-\d{2})-?\s*(\d{2})\b", re.I)),
    (re.compile(r"^LLaMA 3\.1 8B$", re.I), re.compile(r"\b(meta-llama/Meta-Llama-3\.1-8B)-?\s*(Instruct-Turbo)\b")),
    (re.compile(r"^Mistral 7B[- ]Instruct$", re.I), re.compile(r"\b(mistralai/Mistral-7B-Instruct)-?\s*(v\d+\.\d+)\b")),
)


def read_knowledge_benchmarks(pages: list[Page], known: list[Table]) -> list[Table]:
    occupied = {(table.page.number, table.label.casefold()) for table in known}
    tables = []
    for page in pages:
        if page.verification != "text_match":
            continue
        lines = page.text.splitlines()
        for index, line in enumerate(lines):
            if header := HEADER.match(line):
                tables.extend(read_grid(pages, page, lines, index, header[2], occupied))
    return tables


def local_text(line: str) -> str:
    return re.split(r" {3,}", line.strip(), maxsplit=1)[0]


def read_grid(pages: list[Page], page: Page, lines: list[str], index: int, score: str,
              occupied: set[tuple[int, str]]) -> list[Table]:
    panels: dict[str, list[tuple[str, list[str]]]] = {}
    current = None
    footer = None
    for at in range(index + 1, min(len(lines), index + 34)):
        body = lines[at]
        if not body or len(body) - len(body.lstrip()) > 5:
            continue
        if row := ROW.match(body):
            values = list(row.groups()[1:])
            if not current or any(not 0 <= float(value) <= 1 for value in values):
                return []
            panels[current].append((f"{row[1]} knowledge", values))
            continue
        local = local_text(body)
        if CAPTION.match(local):
            footer = at
            break
        if not GENERATOR_NAME.fullmatch(local) or local in panels:
            return []
        current = local
        panels[current] = []
    if footer is None or len(panels) < 2 or any(
        [r.casefold() for r, _ in rows] != ["no knowledge", "full knowledge", "gold knowledge"]
        for rows in panels.values()
    ):
        return []
    caption_parts = []
    for line in lines[footer:footer + 7]:
        if not line or len(line) - len(line.lstrip()) > 2 or HEADER.match(line):
            break
        part = local_text(line)
        caption_parts.append(part)
        if part.endswith("mean)."):
            break
    caption = re.sub(r"(?<=\w)- (?=[a-z])", "", " ".join(caption_parts))
    match = CAPTION.match(caption)
    label, dataset, task = match.groups()
    if (page.number, label.casefold()) in occupied:
        return []
    if not (KNOWLEDGE_METRICS.search(caption) and (answer := ANSWER_METRIC.search(caption))
            and STANDARD_ERROR.search(caption) and f"({score})" in answer[0]):
        return []
    evidence = lookup_split(pages, dataset)
    definitions = [
        ("KP", "Knowledge precision"), ("KR", "Knowledge recall"), ("KF1", "Knowledge F1"),
        (score, "EM" if score == "EM" else "ROUGE-L F1"), ("F1", "Answer F1"), ("F1 / SEM", "Answer F1 SEM"),
    ]
    sem = native_evidence([page], STANDARD_ERROR)
    tables = []
    for model, rows in panels.items():
        version, version_source = lookup_version(pages, model)
        columns = []
        for heading, metric in definitions:
            standard_error = heading == "F1 / SEM"
            columns.append(Column(
                Header(heading, metric, 0, None, None), dataset,
                "standard error of mean" if standard_error else "mean over examples",
                generator=model, generator_version=version, generator_version_evidence=version_source,
                data_split=evidence[0] if evidence else None, data_split_evidence=evidence[2] if evidence else None,
                sample_size=evidence[1] if evidence else None, sample_size_evidence=evidence[2] if evidence else None,
                aggregation_evidence=sem if standard_error else None, reported_scale="fraction", source_label=heading,
                direction=None if standard_error else "higher_is_better",
            ))
        tables.append(Table(page, label, caption, index, columns, rows, row_section=model,
                            family="knowledge_benchmark", task_type="dialogue generation" if task == "response" else "question answering"))
    return tables


def lookup_split(pages: list[Page], dataset: str) -> tuple[str, int, tuple[Page, str]] | None:
    if dataset.casefold() == "hotpotqa" and (evidence := native_evidence(pages, HOT_POT)):
        amount = HOT_POT.fullmatch(evidence[1])[1]
        return f"first {amount} training examples", int(amount), evidence
    if dataset.casefold() == "wow" and (evidence := native_evidence(pages, WOW)):
        conversations, utterances = WOW.fullmatch(evidence[1]).groups()
        return f"first {conversations} conversations from test seen", int(utterances), evidence
    return None


def lookup_version(pages: list[Page], model: str) -> tuple[str | None, tuple[Page, str] | None]:
    for model_pattern, version_pattern in VERSIONS:
        if model_pattern.fullmatch(model) and (evidence := native_evidence(pages, version_pattern)):
            return "-".join(version_pattern.fullmatch(evidence[1]).groups()), evidence
    return None, None
