"""Read native utility and human-judge diagnostics for governed QA."""

from dataclasses import dataclass, replace
from itertools import groupby
import re

from failure_statistics import CAPTION, NUMBER, dataset_name
from pdf_pages import Page
from rag_tables import Column, Header, Table


AUDIT_HEADER = re.compile(
    r"^\s*(?:(?P<judge>Judge)\s{2,}(?P<failure>Failure Type)|Metric)\s{2,}"
    r"N\s+TP\s+FP\s+FN\s+TN\s+Agreement\s+Precision\s+Recall\s+κ\s*$",
)
UTILITY_HEADER = re.compile(r"^\s*(Dataset)\s{2,}(Model)\s{2,}(System)\s*$")
STRATA_HEADER = re.compile(r"^\s*Generator\s+N\s+Leak Agr\.\s+Stale Agr\.\s+Unsup\. Agr\.\s*$")
UTILITY_SCALES = re.compile(
    r"\bPercentages are shown for failure-free, valid-answer, over-refusal, refusal, and agreement metrics",
    re.I,
)
AUDIT_SCALES = re.compile(
    r"\bAgreement is reported as a percentage, while precision, recall, and Cohen’s κ are proportions",
    re.I,
)
PRIMARY_JUDGE = re.compile(r"\bprimary\s+([A-Za-z][\w.-]+)\s+judge\b", re.I)
STRATA_JUDGE = re.compile(r"\bprimary\s+([A-Za-z][\w.-]+)\s+judge’s agreement by generator\b", re.I)
SAME_AUDIT = re.compile(r"\bthe same (\d+) human-audited responses\b", re.I)
SUMMARY_AUDIT = re.compile(
    r"\bOn (?P<sample>the same (?P<size>\d+) human-audited responses), Cohen’s κ across "
    r"the three judges\b.{1,350}?\(Appendix (?P<appendix>Table [A-Z]?\d+)\)",
)
MODEL = re.compile(r"[A-Za-z][\w.]*(?:-[\w.]+)+")
SUFFIX = re.compile(r"[A-Z0-9][\w.]*")
RATE = re.compile(NUMBER)
RATE_OR_DASH = rf"(?:{NUMBER}|[–—-])"
UTILITY_VALUES = re.compile(rf"(?<!\S)(?:{RATE_OR_DASH} +){{4}}{RATE_OR_DASH} *$")
AUDIT_VALUES = re.compile(rf"(?<!\S)(?:\d+ +){{5}}{NUMBER} +{NUMBER} +{NUMBER} +[−-]?{NUMBER} *$")
STRATA_VALUES = re.compile(rf"(?<!\S)\d+ +{NUMBER} +{NUMBER} +{NUMBER} *$")
FAILURES = ("Leakage rate", "Stale citation rate", "Unsupported answer rate")
FAILURE_TYPE = {
    "Leakage": FAILURES[0],
    "Unauthorized Leakage": FAILURES[0],
    "Stale Citation": FAILURES[1],
    "Unsupported": FAILURES[2],
    "Unsupported Answer": FAILURES[2],
}


@dataclass(frozen=True)
class AuditRow:
    line: int
    failure: str
    size: int
    values: tuple[str, ...]


def read_failure_diagnostics(page: Page, lines: list[str], occupied: set[str]) -> list[Table]:
    tables = []
    previous = 0
    for index, line in enumerate(lines):
        match = CAPTION.match(line)
        if match is None:
            continue
        end = next((i for i in range(index + 1, len(lines)) if not lines[i].strip()), len(lines))
        caption = " ".join(" ".join(lines[index:end]).split())
        label = match[1]
        body_start, previous = previous, end
        if label.casefold() in occupied:
            continue
        if UTILITY_SCALES.search(caption) and (judge := PRIMARY_JUDGE.search(caption)):
            tables.extend(read_utility_and_audit(page, label, caption, lines[body_start:index], body_start, judge[1]))
        elif AUDIT_SCALES.search(caption) and (judge := STRATA_JUDGE.search(caption)):
            audit = read_judge_audit(page, label, caption, lines[body_start:index], body_start, judge[1])
            same = SAME_AUDIT.search(caption)
            if same:
                add_repeat_scope(tables, audit, judge[1], int(same[1]), (page.number, same[0]))
            tables.extend(audit)
    return tables


def add_repeat_scope(previous: list[Table], audit: list[Table], judge: str, size: int,
                     scope: tuple[int, str]) -> None:
    name = f"{judge} judge vs human labels"

    def sample(table: Table) -> bool:
        return all(
            column.sample_size == size and column.generator is None for column in table.columns
        )

    for i, table in enumerate(previous):
        if table.config_name == name and sample(table):
            previous[i] = with_audit_scope(table, scope)
    for i, table in enumerate(audit):
        if sample(table):
            audit[i] = with_audit_scope(table, scope)


def with_audit_scope(table: Table, scope: tuple[int, str]) -> Table:
    return replace(table, columns=[replace(column, duplicate_scope=scope) for column in table.columns])


def order_audit_summaries(pages: list[Page], tables: list[Table]) -> tuple[list[Table], list[Table]]:
    appendix = {}
    for table in tables:
        if table.family == "failure_statistics" and table.columns[0].duplicate_scope:
            column = table.columns[0]
            appendix[table.label, table.config_name, column.sample_size] = column.duplicate_scope
    if not appendix:
        return tables, []
    ordinary, linked = [], []
    for table in tables:
        summary = f"{table.label} summarizes robustness checks, utility diagnostics, and judge reliability"
        if table.family != "judge_summary" or summary not in table.page.prose:
            ordinary.append(table)
            continue
        references = [(page, match) for page in pages if table.page.number <= page.number <= table.page.number + 1
                      and page.verification == "text_match" for match in SUMMARY_AUDIT.finditer(page.prose)]
        if len(references) != 1:
            ordinary.append(table)
            continue
        page, evidence = references[0]
        size = int(evidence["size"])
        scope = appendix.get((evidence["appendix"], table.config_name, size))
        if scope is None:
            ordinary.append(table)
            continue
        columns = [replace(column, duplicate_scope=scope, sample_size=size, reported_scale="fraction",
                           sample_size_evidence=(page, evidence["sample"])) for column in table.columns]
        # Emit the confusion-matrix record first, even when its summary appears
        # earlier in the paper. Both records keep their own raw-value evidence.
        linked.append(replace(table, columns=columns))
    return ordinary, linked


def read_utility_and_audit(page: Page, label: str, caption: str, lines: list[str], start: int,
                           primary: str) -> list[Table]:
    utility = [(i, m) for i, line in enumerate(lines) if (m := UTILITY_HEADER.fullmatch(line))]
    matrices = [(i, m) for i, line in enumerate(lines) if (m := AUDIT_HEADER.fullmatch(line)) and not m["judge"]]
    if len(utility) != 1 or len(matrices) != 1:
        return []
    utility_at, headings = utility[0]
    matrix, _ = matrices[0]
    if not 0 < utility_at < matrix:
        return []
    tables = utility_rows(page, label, caption, start + utility_at,
                          lines[utility_at - 1:matrix], headings)
    rows, markers = audit_rows(lines[matrix + 1:], None)
    if markers or [FAILURE_TYPE[row.failure] for row in rows] != list(FAILURES):
        return tables
    return tables + [audit_table(page, label, caption, start + matrix, primary, row) for row in rows]


def read_judge_audit(page: Page, label: str, caption: str, lines: list[str], start: int,
                     primary: str) -> list[Table]:
    matrices = [(i, m) for i, line in enumerate(lines) if (m := AUDIT_HEADER.fullmatch(line)) and m["judge"]]
    strata = [i for i, line in enumerate(lines) if STRATA_HEADER.fullmatch(line)]
    if len(matrices) != 1 or len(strata) != 1:
        return []
    matrix, headings = matrices[0]
    stratum = strata[0]
    if not matrix + 1 < stratum or lines[stratum - 1].strip() != "Primary-judge agreement stratified by generator":
        return []
    rows, markers = audit_rows(lines[matrix + 1:stratum - 1], headings.start("failure"))
    if not markers or len(rows) != len(markers) * len(FAILURES):
        return []
    tables = []
    for offset, (line, judge) in enumerate(markers):
        group = rows[offset * len(FAILURES):(offset + 1) * len(FAILURES)]
        if [FAILURE_TYPE[row.failure] for row in group] != list(FAILURES) or not group[0].line <= line <= group[-1].line:
            return []
        tables.extend(audit_table(page, label, caption, start + matrix, judge, row) for row in group)
    return tables + stratum_rows(page, label, caption, start + stratum, lines[stratum + 1:], primary)


def utility_rows(page: Page, label: str, caption: str, index: int, lines: list[str],
                 headings: re.Match[str]) -> list[Table]:
    if not re.fullmatch(r"Failure +Valid +Over +Refusal +Refusal", lines[0].strip()):
        return []
    if not re.fullmatch(r"-free +Answer +-refusal +Recall +Precision", lines[2].strip()):
        return []
    model_at, system_at = headings.start(2) - 2, headings.start(3) - 2
    models = []
    scores = []
    for i, line in enumerate(lines[3:]):
        match = UTILITY_VALUES.search(line)
        prefix = line[:match.start()] if match else line
        models.append(prefix[model_at:system_at].strip())
        if match:
            dataset = dataset_name(prefix[:model_at].strip())
            system = prefix[system_at:].strip()
            values = match[0].split()
            if not dataset or not system or any(RATE.fullmatch(raw) and float(raw) > 100 for raw in values):
                return []
            scores.append((i, dataset, system, values))
        elif prefix[:model_at].strip() or prefix[system_at:].strip():
            return []
    models = row_models(models, {score[0] for score in scores})
    if models is None:
        return []
    tables = []
    captions = ("Failure-free", "Valid Answer", "Over-refusal", "Refusal Recall", "Refusal Precision")
    metrics = ("Failure-free rate", "Valid answer rate", "Over-refusal rate", "Refusal recall", "Refusal precision")
    for i, dataset, system, values in scores:
        model = models[i]
        columns = [Column(Header(header, name, 0, None, "%"), dataset, generator=model,
                          original_row=f"{dataset} / {model} / {system}", generator_locator="Model",
                          source_label=header, direction="lower_is_better" if name == "Over-refusal rate" else "higher_is_better")
                   for header, name in zip(captions, metrics, strict=True)]
        tables.append(Table(page, label, caption, index + i, columns, [(system, values)], family="failure_statistics"))
    return tables


def row_models(cells: list[str], scored: set[int], aggregates: set[str] = frozenset()) -> dict[int, str] | None:
    used = set()
    models = {}
    for i in sorted(scored):
        model = cells[i]
        if not model and 0 < i < len(cells) - 1 and i - 1 not in scored and i + 1 not in scored:
            if cells[i - 1].endswith("-") and SUFFIX.fullmatch(cells[i + 1]) and not {i - 1, i + 1} & used:
                model = cells[i - 1] + cells[i + 1]
                used.update((i - 1, i + 1))
        elif model:
            used.add(i)
        if model not in aggregates and not MODEL.fullmatch(model):
            return None
        models[i] = model
    return models if all(not cell or i in used for i, cell in enumerate(cells)) else None


def audit_rows(lines: list[str], failure_at: int | None) -> tuple[list[AuditRow], list[tuple[int, str]]]:
    rows = []
    markers = []
    pending = None
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        match = AUDIT_VALUES.search(line)
        prefix = line[:match.start()] if match else line
        judge = prefix[:failure_at - 1].strip() if failure_at is not None else ""
        failure = prefix[failure_at - 1:].strip() if failure_at is not None else prefix.strip()
        if judge:
            if pending is not None:
                if not SUFFIX.fullmatch(judge):
                    return [], []
                offset, name = pending
                markers.append(((offset + i) // 2, name + judge))
                pending = None
            elif MODEL.fullmatch(judge):
                markers.append((i, judge))
            elif judge.endswith("-"):
                pending = i, judge
            else:
                return [], []
        if match:
            size, *values = match[0].split()
            counts = list(map(int, values[:4]))
            agreement, precision, recall, kappa = (float(raw.replace("−", "-")) for raw in values[4:])
            if (failure not in FAILURE_TYPE or sum(counts) != int(size) or int(size) <= 0 or not
                    (0 <= agreement <= 100 and 0 <= precision <= 1 and 0 <= recall <= 1 and -1 <= kappa <= 1)):
                return [], []
            rows.append(AuditRow(i, failure, int(size), tuple(values)))
        elif failure or not judge:
            return [], []
    return (rows, markers) if pending is None else ([], [])


def audit_table(page: Page, label: str, caption: str, index: int, judge: str, row: AuditRow) -> Table:
    locator = f"{judge} / {row.failure}"
    columns = [Column(Header(header, "Cohen kappa" if header == "κ" else header, 0, None,
                             "%" if header == "Agreement" else None), None,
                      original_row=locator, source_label=header, metric_variant=FAILURE_TYPE[row.failure],
                      metric_variant_evidence=(page, row.failure),
                      reported_scale="fraction" if header in {"Precision", "Recall", "κ"} else None,
                      sample_size=row.size)
               for header in ("TP", "FP", "FN", "TN", "Agreement", "Precision", "Recall", "κ")]
    return Table(page, label, caption, index + row.line, columns, [(judge, list(row.values))],
                 config_name=f"{judge} judge vs human labels", family="failure_statistics")


def stratum_rows(page: Page, label: str, caption: str, index: int, lines: list[str], primary: str) -> list[Table]:
    cells = []
    scores = []
    for i, line in enumerate(lines):
        match = STRATA_VALUES.search(line)
        cells.append(line[:match.start()].strip() if match else line.strip())
        if match:
            size, *values = match[0].split()
            if int(size) <= 0 or any(float(raw) > 100 for raw in values):
                return []
            scores.append((i, int(size), values))
    models = row_models(cells, {score[0] for score in scores})
    if models is None:
        return []
    tables = []
    for i, size, values in scores:
        model = models[i]
        columns = [Column(Header(header, "Agreement", 0, None, "%"), None, generator=model,
                          original_row=model, generator_locator="Generator", source_label=header,
                          metric_variant=failure, metric_variant_evidence=(page, header), sample_size=size)
                   for header, failure in zip(("Leak Agr.", "Stale Agr.", "Unsup. Agr."), FAILURES, strict=True)]
        tables.append(Table(page, label, caption, index + i, columns, [(primary, values)],
                            config_name=f"{primary} judge vs human labels", family="failure_statistics"))
    return tables
