"""Read PRO-STEP audit statistics, significance markers, and test outcomes."""

from dataclasses import replace
from decimal import Decimal
import re

from failure_rates import visual_rows
from grouped_scores import native_evidence
from native_benchmark_tables import normalized_boxes, positioned_caption, words_text
from paired_scores import MAIN_BACKBONE
from pdf_pages import Page, Word
from prostep_context import RETRIEVAL
from rag_tables import Column, Header, NATIVE_CAPTION, Table, distinct_labels
from recovery_scores import compatible_size
from retrieval_tables import source_statement


AUDIT = re.compile(r"Agreement between QwQ-32B \(our\b", re.I)
AUDIT_CAPTION = re.compile(r"Agreement between QwQ-32B \(our annotator\) and Claude Opus 4\.7 "
                           r"on the (\d+)-sample audit\.", re.I)
STRATA = re.compile(r"Per-step-type agreement on the \d+-sample\b", re.I)
AUDIT_SCOPE = re.compile(r"\bQwQ-32B-labeled trajectories \(held-out from PRM training\)", re.I)
AUDIT_CONFIG = "PRM supervision statistics"
AUDIT_DATASET = "Frontier-label audit"
MARKERS = re.compile(r"Results at the (\d+B) scale \(EM / F1, %\)\.", re.I)
STAR_LEGEND = re.compile(r"[“\"]\*[”\"] denotes p (<)\s*(0\.\d+) against (Search-R1)\.", re.I)
MODEL_3B = re.compile(r"\bat a smaller model\s+scale \((Qwen2\.5-3B-Instruct)\)", re.I)
IFBC = re.compile(r"Intermediate-Flawed-But-Correct \(IFBC\) rate\b", re.I)
IFBC_CAPTION = re.compile(r"\brate among correct trajectories\. Statistical tests compare each method with "
                          r"(PRO-S\s*TEP)\s*\. Lower is better\.", re.I)
PAIRED = re.compile(r"Per-dataset paired t-test on EM\.", re.I)
PAIRED_CAPTION = re.compile(r"Per-dataset paired t-test on EM\. n = test split size\.", re.I)
PAIRED_SCOPE = re.compile(r"\bWe supplement Table \d+ with bootstrap confidence intervals and per-dataset "
                          r"paired t-tests,\s+computed on per-question results against PRO-S\s*TEP\.", re.I)
PAIRED_HEADER = "Dataset n vs. Search-R1 vs. ReasonRAG vs. StepSearch"
DELTA_CELL = re.compile(r"[+−-]\d+\.\d+,")
SIGNIFICANCE = re.compile(r"(?:NS|[⋆*]+)")
INTEGER = re.compile(r"\d+")
COUNT = re.compile(r"\d+|\d{1,3}(?:,\d{3})+")
PERCENT = re.compile(r"\d+(?:\.\d+)?%")
KAPPA = re.compile(r"[−-]?(?:0\.\d+|1(?:\.0+)?|0)")
SCORE = re.compile(r"\d{1,2}\.\d+\*?|100\.0+\*?")
EXACT_P = re.compile(r"(?:0\.\d+|1(?:\.0+)?|0)")
BOUND_P = re.compile(r"<10[−-][1-9]\d*")


def read_prostep_appendix(pages: list[Page], known: list[Table]) -> list[Table]:
    occupied = {(table.page.number, table.label.casefold()) for table in known}
    native = [page for page in pages if page.verification == "text_match"]
    scope = source_statement(native, AUDIT_SCOPE)
    model = source_statement(native, MODEL_3B)
    paired = source_statement(native, PAIRED_SCOPE)
    paired_setup = (paired, native_evidence(native, MAIN_BACKBONE), source_statement(native, RETRIEVAL))
    tables = []
    for page in native:
        if not page.words or not page.size:
            continue
        rows = visual_rows(page.words)
        locators = (match for line in page.text.splitlines() for match in NATIVE_CAPTION.finditer(line))
        for locator in locators:
            label = locator[1]
            prefix = locator.string[locator.end():].lstrip()
            for pattern, header, height, read, context in (
                (AUDIT, "Metric Value", 124, audit_summary, scope),
                (STRATA, "Step type N Agreement Cohen’s κ", 105, audit_strata, scope),
                (MARKERS, "Dataset Base 3B Search-R1 PRO-Step", 143, significance_markers, model),
                (IFBC, "Method IFBC p vs. Ours # Correct", 113, ifbc_tests, None),
                (PAIRED, PAIRED_HEADER, 80, paired_tests, paired_setup),
            ):
                if not pattern.match(prefix) or (pattern is not MARKERS and (page.number, label.casefold()) in occupied):
                    continue
                anchor = positioned_caption(rows, label, pattern)
                if anchor is None:
                    continue
                grid = caption_grid(page, rows, anchor, height, header)
                if grid is not None:
                    tables.extend(read(page, label, *grid, context))
    return tables


def caption_grid(page, rows, anchor, height, header):
    center = page.size[0] / 2
    left, right = (0, center) if anchor.x < center else (center, page.size[0])
    region = [cells for row in rows if (cells := [word for word in row
              if left <= word.x < right and anchor.y - 2 <= word.y < anchor.y + height])]
    headings = [i for i, row in enumerate(region) if words_text(row) == header]
    if len(headings) != 1:
        return None
    index = headings[0]
    return " ".join(words_text(row) for row in region[:index]), region[index], region[index + 1:]


def cell(words, heading, metric, dataset, *, raw=None, **conditions):
    word = Word(words_text(words), min(w.x0 for w in words), min(w.y0 for w in words),
                max(w.x1 for w in words), max(w.y1 for w in words))
    column = Column(Header(heading, metric, word.x, None, None), dataset, source_label=heading, **conditions)
    return column, words_text(words) if raw is None else raw, word


def cell_row(page, label, caption, row, cells, **conditions):
    columns, raw, words = zip(*cells, strict=True)
    return Table(page, label, caption, int(words[0].y), list(columns), [(row, list(raw))],
                 value_bboxes=normalized_boxes(page, words), **conditions)


def audit_split(cells, scope):
    if scope is None:
        return cells
    return [(replace(column, data_split="held out from PRM training", data_split_evidence=(scope[0], scope[1][0])),
             value, word) for column, value, word in cells]


def audit_summary(page, label, caption, _header, body, scope):
    profile = AUDIT_CAPTION.search(caption)
    if profile is None or len(body) != 6:
        return []
    samples, agreement, wrap, kappa, qwq, frontier = body
    names = [words_text(row[:-1]) for row in (samples, qwq, frontier)]
    if names != ["Audited samples", "QwQ-labeled GOOD", "Frontier-labeled GOOD"]:
        return []
    if (words_text(wrap) != "valid labels)" or words_text(agreement[:1]) != "Agreement"
            or words_text(kappa[:2]) != "Cohen’s κ"):
        return []
    ratio = re.fullmatch(r"(\d+(?:\.\d+)?)% \((\d+) / (\d+) with", words_text(agreement[1:]))
    if (ratio is None or len(agreement) != 6 or len(kappa) < 3 or not KAPPA.fullmatch(kappa[2].text)
            or any(not INTEGER.fullmatch(row[-1].text) for row in (samples, qwq, frontier))):
        return []
    audited = int(samples[-1].text)
    matching, valid = map(int, ratio.groups()[1:])
    rate = Decimal(ratio[1])
    if (not 0 < matching <= valid <= audited == int(profile[1])
            or not all(0 <= int(row[-1].text) <= audited for row in (qwq, frontier))
            or abs(rate - 100 * Decimal(matching) / valid) > Decimal("0.5").scaleb(rate.as_tuple().exponent)):
        return []
    # Opening parentheses surround the two count cells and do not belong to either count.
    cells = [
        cell(agreement[4:5], "Agreement / valid labels", "Valid audited labels", AUDIT_DATASET,
             sample_size=audited, sample_size_evidence=(page, samples[-1].text)),
        cell(agreement[2:3], "Agreement / matching labels", "Matching labels", AUDIT_DATASET, raw=ratio[2],
             sample_size=valid, sample_size_evidence=(page, ratio[0])),
        cell(agreement[1:2], "Agreement", "Agreement", AUDIT_DATASET,
             sample_size=valid, sample_size_evidence=(page, ratio[0])),
        cell(kappa[2:3], "Cohen’s κ", "Cohen's kappa", AUDIT_DATASET,
             sample_size=valid, sample_size_evidence=(page, ratio[0])),
        cell(qwq[-1:], "QwQ-labeled GOOD", "QwQ GOOD labels", AUDIT_DATASET),
        cell(frontier[-1:], "Frontier-labeled GOOD", "Claude GOOD labels", AUDIT_DATASET),
    ]
    return [cell_row(page, label, caption, "Overall audit", audit_split(cells, scope), config_name=AUDIT_CONFIG)]


def audit_strata(page, label, caption, _header, body, scope):
    tables = []
    for row in body:
        if len(row) != 4:
            break
        action, count, agreement, kappa = row
        if (action.text not in {"answer", "search", "reason"} or not INTEGER.fullmatch(count.text)
                or not PERCENT.fullmatch(agreement.text) or not KAPPA.fullmatch(kappa.text)):
            break
        size = int(count.text)
        if not compatible_size(agreement.text.removesuffix("%"), size):
            return []
        conditions = {"metric_variant": action.text, "sample_size": size, "sample_size_evidence": (page, count.text),
                      "metric_variant_evidence": (page, action.text)}
        cells = [cell([agreement], "Agreement", "Agreement", AUDIT_DATASET, **conditions),
                 cell([kappa], "Cohen’s κ", "Cohen's kappa", AUDIT_DATASET, **conditions)]
        tables.append(cell_row(page, label, caption, action.text, audit_split(cells, scope), config_name=AUDIT_CONFIG))
    return tables if {table.rows[0][0] for table in tables} == {"answer", "search", "reason"} and len(tables) == 3 else []


def significance_markers(page, label, caption, header, body, model):
    scale = MARKERS.search(caption)
    legend = STAR_LEGEND.search(caption)
    if scale is None or legend is None or scale[1] != "3B":
        return []
    operator, threshold, baseline = legend.groups()
    tables = []
    datasets = []
    for row in body:
        if len(row) != 10 or not all(word.text == "/" for word in row[2::3]):
            break
        dataset = row[0].text
        scores = [word for word in row[1:] if word.text != "/"]
        if (not all(SCORE.fullmatch(word.text) for word in scores)
                or any(word.text.endswith("*") for word in scores[:-2])):
            return []
        if not all(abs(word.x - center) < 12 for word, center in zip(
            row[2::3], ((header[1].x + header[2].x) / 2, header[3].x, header[4].x), strict=True
        )):
            return []
        datasets.append(dataset)
        cells = [cell([word], f"PRO-Step / {metric} / *", "Significance p upper bound",
                      "2WikiMultiHopQA" if dataset == "2WikiMulti" else dataset, raw=operator + threshold,
                      original_row=dataset,
                      metric_variant=f"{metric} vs {baseline} {scale[1]}", metric_variant_evidence=(page, legend[0]),
                      generator=model[1][1] if model else None, generator_evidence=(model[0], model[1][0]) if model else None)
                 for metric, word in zip(("EM", "F1"), scores[-2:], strict=True) if word.text.endswith("*")]
        if cells:
            tables.append(cell_row(page, label, caption, "PRO-Step", cells, config_name="PRO-STEP"))
    return tables if len(datasets) >= 3 and datasets[-1] == "Average" and distinct_labels(datasets) else []


def ifbc_tests(page, label, caption, header, body, _model):
    reference = IFBC_CAPTION.search(caption)
    if reference is None:
        return []
    baseline = reference[1].replace(" ", "")
    dataset = "IFBC paired evaluation subset"
    tables = []
    for row in body:
        if words_text(row) == "↓":
            continue
        tokens = [i for i, word in enumerate(row) if PERCENT.fullmatch(word.text)]
        if len(tokens) != 1:
            break
        percentage_at = tokens[0]
        method = words_text(row[:percentage_at])
        count = row[-1]
        rate = row[percentage_at]
        probability = row[percentage_at + 1:-1]
        if (not method or not probability or not INTEGER.fullmatch(count.text)
                or abs(rate.x - header[1].x) > 10 or abs(count.x - (header[-2].x + header[-1].x) / 2) > 10):
            return []
        if not compatible_size(rate.text.removesuffix("%"), int(count.text)):
            return []
        cells = [cell([count], "# Correct", "Correct trajectories", dataset)]
        is_baseline = method.replace(" ", "") == baseline
        bound = "".join(word.text for word in probability)
        if is_baseline:
            if bound not in {"–", "-", "—"}:
                return []
        elif BOUND_P.fullmatch(bound):
            cells.append(cell(probability, "p vs. Ours", "IFBC difference p", dataset, raw=bound,
                              metric_variant=f"vs {baseline}; upper exclusive bound", metric_variant_evidence=(page, reference[0])))
        elif EXACT_P.fullmatch(probability[0].text) and words_text(probability[1:]) in {"", "( NS )", "(NS)"}:
            cells.append(cell(probability[:1], "p vs. Ours", "IFBC difference p", dataset,
                              metric_variant=f"vs {baseline}", metric_variant_evidence=(page, reference[0])))
        else:
            return []
        tables.append(cell_row(page, label, caption, method, cells, config_name=baseline if is_baseline else method))
    methods = [table.config_name for table in tables]
    return tables if len(methods) >= 3 and methods.count(baseline) == 1 and distinct_labels(methods) else []


def paired_tests(page, label, caption, header, body, setup):
    protocol = PAIRED_CAPTION.search(caption)
    comparison, backbone, retrieval = setup
    # The grid names comparators. Appendix prose identifies the other member of the pair.
    if protocol is None or comparison is None:
        return []
    baselines = header[3::2]
    centers = [(heading.x + baseline.x) / 2 for heading, baseline in zip(header[2::2], baselines, strict=True)]
    generator = MAIN_BACKBONE.fullmatch(backbone[1])[1] if backbone else None
    tables = []
    for row in body:
        if len(row) < 3 or row[2].text[-1:] != ",":
            break
        dataset, count = row[:2]
        if not COUNT.fullmatch(count.text):
            return []
        size = int(count.text.replace(",", ""))
        starts = [i for i, word in enumerate(row) if DELTA_CELL.fullmatch(word.text)]
        if len(starts) != len(baselines) or starts[0] != 2 or size <= 0:
            return []
        if any(abs(row[start].x - center) > 16 for start, center in zip(starts, centers, strict=True)):
            return []
        cells = []
        for start, end, baseline in zip(starts, [*starts[1:], len(row)], baselines, strict=True):
            probability = paired_probability(row[start + 1:end])
            if probability is None:
                return []
            raw, words = probability
            variant = f"vs {baseline.text}"
            if raw.startswith("<"):
                variant += " upper exclusive bound"
            cells.append(cell(words, f"vs. {baseline.text} / p", "Paired t-test p", dataset.text, raw=raw,
                              metric_variant=variant, metric_variant_evidence=(comparison[0], comparison[1][0]),
                              data_split="test", data_split_evidence=(page, protocol[0]),
                              sample_size=size, sample_size_evidence=(page, count.text),
                              generator=generator, generator_evidence=backbone))
        tables.append(cell_row(page, label, caption, dataset.text, cells, config_name="PRO-STEP",
                               retriever_model=retrieval[1][1] if retrieval else None,
                               top_k=int(retrieval[1][2]) if retrieval else None,
                               top_k_evidence=(retrieval[0], retrieval[1][0]) if retrieval else None))
    datasets = [table.rows[0][0] for table in tables]
    return tables if len(datasets) >= 3 and distinct_labels(datasets) else []


def paired_probability(words):
    while words and SIGNIFICANCE.fullmatch(words[-1].text):
        words = words[:-1]
    if len(words) == 1 and words[0].text.startswith("p="):
        raw = words[0].text[2:]
        return (raw, words) if EXACT_P.fullmatch(raw) else None
    if len(words) == 4 and words[0].text == "p":
        raw = "".join(word.text for word in words[1:])
        return (raw, words[1:]) if BOUND_P.fullmatch(raw) else None
    return None
