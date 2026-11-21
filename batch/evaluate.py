"""Measure local, one-to-one CSV agreement with the organizer's dev gold."""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import io
import json
from pathlib import Path
import re
import sys
import unicodedata
import zipfile


SCOPES = ("all_results", "table_only")
STATUSES = ("extracted", "absent", "present_not_extractable")
COUNTS = ("rows", *STATUSES, "marked_duplicate", "excluded_rows", "eligible",
          "source_keyed", "context_keyed")
CONFIG_FIELDS = ("rag_variant", "generator_model", "generator_version", "top_k")
DATASET_FIELDS = ("dataset_name", "dataset_version", "data_split")
RESULT_FIELDS = (
    "result_id", "config_id", "dataset_id", "extraction_status", "duplicate_of",
    "raw_value", "raw_unit", "metric_name", "metric_variant", "metric_k", "aggregation",
    "source_page", "source_table", "source_figure",
)
SOURCE_FIELDS = ("raw_value", "raw_unit", "metric_name", "metric_k", "metric_variant", "aggregation",
                 "source_page", "source_table", "source_figure")
DIAGNOSTIC_FIELDS = SOURCE_FIELDS[1:]
DECIMAL = re.compile(r"([+-]?)([0-9]*)(?:\.([0-9]+))? *(%?)")
INTEGER = re.compile(r"[0-9]+")
INLINE_CUTOFF = re.compile(r"\s*@\s*([0-9]+)(?![\w.])")
TABLE_PREFIX = re.compile(r"^(?:table|tab\.?)(?:\s+(.+))?$")
FIGURE_PREFIX = re.compile(r"^(?:figure|fig\.?)(?:\s+(.+))?$")
MISSING = {"", "-", "–", "—", "null", "n/a", "none", "unknown"}
AGREEMENT_LABEL = "Precision, recall and F1 measure agreement on these local matching keys only. They do not establish scientific validity."


def literal(value):
    return " ".join(unicodedata.normalize("NFC", value).split())


def label(value):
    return literal(value).lower()


def raw_key(value):
    value = literal(value).replace("−", "-")
    if value.lower() in MISSING:
        return ""
    decimal = DECIMAL.fullmatch(value)
    if decimal and (decimal[2] or decimal[3]):
        sign, whole, fraction, percent = decimal.groups()
        whole = whole.lstrip("0") or "0"
        fraction = (fraction or "").rstrip("0")
        return sign + whole + (f".{fraction}" if fraction else "") + percent
    return value


def unsigned_integer(value):
    return (value.lstrip("0") or "0") if INTEGER.fullmatch(value) else None


def metric_key(row):
    name = label(row["metric_name"])
    cutoff_field = literal(row["metric_k"])
    cutoff = unsigned_integer(cutoff_field)
    if name in MISSING or (cutoff_field and cutoff is None):
        return None
    inline = {unsigned_integer(match[1]) for match in INLINE_CUTOFF.finditer(name)}
    if len(inline) > 1 or (cutoff is not None and inline and cutoff not in inline):
        return None
    if inline:
        cutoff, = inline
        name = INLINE_CUTOFF.sub(f"@{cutoff}", name)
        # A terminal @k and the dedicated column express the same cutoff.
        name = name.removesuffix(f"@{cutoff}")
    if name in MISSING:
        return None
    name = re.sub(r"^(bleu|rouge)[ _-]([0-9]+|l)$", r"\1-\2", name)
    return name, cutoff, label(row["metric_variant"]), label(row["aggregation"])


def locator_key(value, prefix):
    value = label(value)
    match = prefix.fullmatch(value)
    if match:
        value = match[1] or ""
    return "" if value in MISSING else value


def source_key(row):
    value = raw_key(row["raw_value"])
    unit = literal(row["raw_unit"])
    if value.endswith("%") and label(unit) in {"%", "percent"}:
        value = value[:-1]
    metric = metric_key(row)
    page = unsigned_integer(literal(row["source_page"]))
    table = locator_key(row["source_table"], TABLE_PREFIX)
    figure = locator_key(row["source_figure"], FIGURE_PREFIX)
    if not value or metric is None or page in (None, "0") or not (table or figure):
        return None
    return value, unit, metric, page, table, figure


def indexed_csv(open_csv, table, id_field, fields, *, optional=False):
    try:
        stream = open_csv(table)
    except (FileNotFoundError, KeyError):
        if optional:
            return None
        raise ValueError(f"missing {table}.csv") from None

    with stream:
        reader = csv.DictReader(stream, strict=True)
        header = reader.fieldnames or []
        required = {"publication_id", id_field, *fields}
        if missing := required.difference(header):
            raise ValueError(f"{table}.csv missing columns: {', '.join(sorted(missing))}")
        if len(header) != len(set(header)):
            raise ValueError(f"{table}.csv has duplicate column names")
        rows = {}
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f"{table}.csv line {reader.line_num} has the wrong number of cells")
            publication_id = row["publication_id"]
            record_id = row[id_field]
            key = publication_id if id_field == "publication_id" else (publication_id, record_id)
            if not publication_id.strip() or not record_id.strip() or key in rows:
                raise ValueError(f"{table}.csv line {reader.line_num} has a blank or duplicate ID: {key!r}")
            rows[key] = row
        return rows


@dataclass
class Corpus:
    publications: dict
    results: dict
    configurations: dict | None
    datasets: dict | None

    @property
    def has_context(self):
        return self.configurations is not None and self.datasets is not None

    def context_key(self, row):
        if not self.has_context:
            return None
        publication = row["publication_id"]
        config = self.configurations.get((publication, row["config_id"]))
        dataset = self.datasets.get((publication, row["dataset_id"]))
        if config is None or dataset is None:
            return None
        config_fields = tuple(label(config[field]) for field in CONFIG_FIELDS)
        dataset_fields = tuple(label(dataset[field]) for field in DATASET_FIELDS)
        if config_fields[0] not in MISSING and dataset_fields[0] not in MISSING:
            return config_fields, dataset_fields
        return None


def read_corpus(open_csv):
    publications = indexed_csv(open_csv, "publications", "publication_id", ("in_scope", "exclusion_reason"))
    results = indexed_csv(open_csv, "results", "result_id", RESULT_FIELDS)
    configurations = indexed_csv(open_csv, "configurations", "config_id", CONFIG_FIELDS, optional=True)
    datasets = indexed_csv(open_csv, "datasets", "dataset_id", DATASET_FIELDS, optional=True)
    by_publication = defaultdict(list)
    for publication_id, row in publications.items():
        row["in_scope"] = label(row["in_scope"])
        if row["in_scope"] not in ("true", "false"):
            raise ValueError(f"{publication_id}: expected in_scope=true or false")
    for (publication_id, result_id), row in results.items():
        if publication_id not in publications:
            raise ValueError(f"{publication_id}/{result_id}: results row has no publication")
        status = label(row["extraction_status"])
        if status not in STATUSES:
            raise ValueError(f"{publication_id}/{result_id}: invalid extraction_status {status!r}")
        row["extraction_status"] = status
        duplicate_of = row["duplicate_of"].strip()
        if duplicate_of:
            target = results.get((publication_id, duplicate_of))
            if (status != "extracted" or target is None or target["duplicate_of"].strip()
                    or label(target["extraction_status"]) != "extracted"):
                raise ValueError(f"{publication_id}/{result_id}: duplicate_of must point to a canonical extracted row")
        row["duplicate_of"] = duplicate_of
        by_publication[publication_id].append(row)
    return Corpus(publications, by_publication, configurations, datasets)


def read_directory(directory):
    return read_corpus(lambda name: (directory / f"{name}.csv").open(encoding="utf-8-sig", newline=""))


def read_gold_zip(path):
    with zipfile.ZipFile(path) as archive:
        return read_corpus(lambda name: io.TextIOWrapper(
            archive.open(f"dev_gold/{name}.csv"), encoding="utf-8-sig", newline=""))


def publication_report(corpus, publication_id):
    publication = corpus.publications.get(publication_id)
    decision = "missing"
    reason = ""
    if publication is not None:
        decision = "included" if publication["in_scope"] == "true" else "excluded"
        reason = publication["exclusion_reason"]
    return {"decision": decision, "exclusion_reason": reason}


def scoped_records(corpus, publication_id, decision, scope, in_gold):
    counts = dict.fromkeys(COUNTS, 0)
    source_keys = Counter()
    context_keys = Counter()
    for row in corpus.results.get(publication_id, ()):
        if scope == "table_only" and not locator_key(row["source_table"], TABLE_PREFIX):
            continue
        status = row["extraction_status"]
        counts["rows"] += 1
        counts[status] += 1
        counts["marked_duplicate"] += bool(row["duplicate_of"])
        counts["excluded_rows"] += decision == "excluded"
        if not in_gold or decision != "included" or status != "extracted" or row["duplicate_of"]:
            continue
        counts["eligible"] += 1
        key = source_key(row)
        if key is not None:
            source_keys[key] += 1
            if (context := corpus.context_key(row)) is not None:
                context_keys[key, context] += 1
    counts["source_keyed"] = source_keys.total()
    counts["context_keyed"] = context_keys.total()
    return counts, source_keys, context_keys


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def agreement(matches, gold, candidate):
    return {
        "matches": matches,
        "gold_unmatched": gold - matches,
        "candidate_unmatched": candidate - matches,
        "precision": ratio(matches, candidate),
        "recall": ratio(matches, gold),
        "f1": ratio(2 * matches, candidate + gold),
    }


def flatten_source_keys(counter):
    return {(value, unit, *metric, page, table, figure): n
            for (value, unit, metric, page, table, figure), n in counter.items()}


def project_source_keys(counter, offset):
    projected = Counter()
    for key, count in counter.items():
        projected[key[:offset] + key[offset + 1:]] += count
    return projected


def additional_matches_by_field(gold, candidate, strict):
    # Keep exact matches fixed. Each projection only compares the unmatched rows.
    gold_rest = flatten_source_keys(gold - strict)
    candidate_rest = flatten_source_keys(candidate - strict)
    counts = dict.fromkeys(DIAGNOSTIC_FIELDS, 0)
    if gold_rest and candidate_rest:
        for offset, field in enumerate(DIAGNOSTIC_FIELDS, start=1):
            gold_projected = project_source_keys(gold_rest, offset)
            candidate_projected = project_source_keys(candidate_rest, offset)
            counts[field] = (gold_projected & candidate_projected).total()
    return counts


def scope_report(gold, candidate, publication_id, decisions, scope, diagnostics):
    in_gold = publication_id in gold.publications
    gold_counts, gold_source, gold_context = scoped_records(gold, publication_id, decisions["gold"], scope, in_gold)
    candidate_counts, candidate_source, candidate_context = scoped_records(
        candidate, publication_id, decisions["candidate"], scope, in_gold)
    denominators = gold_counts["eligible"], candidate_counts["eligible"]
    strict = gold_source & candidate_source
    report = {
        "gold": gold_counts,
        "candidate": candidate_counts,
        "source_agreement": agreement(strict.total(), *denominators) if in_gold else None,
        "context_agreement": agreement((gold_context & candidate_context).total(), *denominators)
        if in_gold and gold.has_context and candidate.has_context else None,
    }
    if diagnostics:
        report["additional_matches_by_field"] = additional_matches_by_field(gold_source, candidate_source, strict) if in_gold else None
    return report


def sum_counts(papers, side):
    return {field: sum(paper[side][field] for paper in papers) for field in COUNTS}


def corpus_agreement(papers, key, gold_count, candidate_count):
    scores = [paper[key] for paper in papers]
    eligible = [paper[key] for paper in papers if paper["gold"]["eligible"]]
    return {
        "micro": agreement(sum(score["matches"] for score in scores), gold_count, candidate_count),
        "macro": {"papers": len(eligible), **{
            metric: ratio(sum(score[metric] or 0.0 for score in eligible), len(eligible))
            for metric in ("precision", "recall", "f1")
        }},
    }


def corpus_scope(documents, expected, unexpected, scope, has_context, diagnostics):
    papers = [documents[publication_id]["scopes"][scope] for publication_id in expected]
    extras = [documents[publication_id]["scopes"][scope] for publication_id in unexpected]
    gold = sum_counts(papers, "gold")
    candidate = sum_counts(papers, "candidate")
    denominators = gold["eligible"], candidate["eligible"]
    report = {
        "gold": gold,
        "candidate": candidate,
        "candidate_outside_gold": sum_counts(extras, "candidate"),
        "gold_papers_with_eligible_records": sum(bool(paper["gold"]["eligible"]) for paper in papers),
        "candidate_papers_with_eligible_records": sum(bool(paper["candidate"]["eligible"]) for paper in papers),
        "source_agreement": corpus_agreement(papers, "source_agreement", *denominators),
        "context_agreement": corpus_agreement(papers, "context_agreement", *denominators) if has_context else None,
    }
    if diagnostics:
        report["additional_matches_by_field"] = {
            field: sum(paper["additional_matches_by_field"][field] for paper in papers)
            for field in DIAGNOSTIC_FIELDS
        }
    return report


def evaluate(gold, candidate, *, diagnostics=False):
    expected = sorted(gold.publications)
    present = sorted(gold.publications.keys() & candidate.publications.keys())
    missing = sorted(gold.publications.keys() - candidate.publications.keys())
    unexpected = sorted(candidate.publications.keys() - gold.publications.keys())
    documents = {}
    for publication_id in sorted(expected + unexpected):
        sides = {"gold": publication_report(gold, publication_id),
                 "candidate": publication_report(candidate, publication_id)}
        decisions = {side: info["decision"] for side, info in sides.items()}
        documents[publication_id] = {**sides, "in_gold": publication_id in gold.publications, "scopes": {
            scope: scope_report(gold, candidate, publication_id, decisions, scope, diagnostics) for scope in SCOPES
        }}
    report = {
        "report_version": 1,
        "interpretation": AGREEMENT_LABEL,
        "scoring": {
            "eligible": "Canonical extracted rows from included publications within the gold PDF set.",
            "table_only": "Eligible rows with a nonempty table locator.",
            "source": "Exact normalized raw value, raw unit, metric, cutoff, variant, aggregation, page and table/figure keys.",
            "context": "Source key plus publication-local configuration and dataset field keys, with the same denominators.",
            "macro": "Arithmetic mean across papers with eligible gold rows in each scope, using zero precision for papers with no eligible candidate rows.",
        },
        "corpus": {
            "coverage": {
                "gold_pdfs": len(expected),
                "candidate_pdfs": len(candidate.publications),
                "present_gold_pdfs": len(present),
                "pdf_coverage": ratio(len(present), len(expected)),
                "missing_pdfs": missing,
                "outside_gold_pdfs": unexpected,
                "scope_disagreement_pdfs": [publication_id for publication_id in present if
                                            documents[publication_id]["gold"]["decision"] !=
                                            documents[publication_id]["candidate"]["decision"]],
            },
            "context_tables_available": {"gold": gold.has_context, "candidate": candidate.has_context},
            "scopes": {scope: corpus_scope(documents, expected, unexpected, scope,
                                         gold.has_context and candidate.has_context, diagnostics)
                       for scope in SCOPES},
        },
        "documents": documents,
    }
    if diagnostics:
        report["scoring"]["additional_matches_by_field"] = (
            "One-to-one matches gained by ignoring one field at a time after exact matching. "
            "Counts can overlap across fields and do not affect precision, recall or F1."
        )
    return report


def percent(value):
    return "n/a" if value is None else f"{value:.2%}"


def ratios_text(score):
    if score is None:
        return "unavailable"
    return "/".join(percent(score[field]) for field in ("precision", "recall", "f1"))


def counts_text(scope):
    return ", ".join(f"{field} {scope['gold'][field]}/{scope['candidate'][field]}" for field in COUNTS)


def paper_score_text(name, score):
    return f"{name} unavailable" if score is None else f"{name} {score['matches']} matches, P/R/F1 {ratios_text(score)}"


def print_diagnostics(scope, indent):
    if (counts := scope.get("additional_matches_by_field")) is not None:
        entries = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        positive = ", ".join(f"{field} +{count}" for field, count in entries if count)
        print(f"{indent}Additional matches when ignoring one field: {positive or 'none'}")


def print_report(report):
    print(report["interpretation"])
    corpus = report["corpus"]
    coverage = corpus["coverage"]
    print(f"PDF coverage {coverage['present_gold_pdfs']}/{coverage['gold_pdfs']} "
          f"({percent(coverage['pdf_coverage'])}), {coverage['candidate_pdfs']} candidate PDFs")
    for key, title in (("missing_pdfs", "Missing PDFs"), ("outside_gold_pdfs", "Candidate PDFs outside gold"),
                       ("scope_disagreement_pdfs", "Inclusion disagreements")):
        print(f"{title} {', '.join(coverage[key]) or 'none'}")
    print("Counts are gold/candidate. Macro averages eligible gold papers and uses zero precision when a paper has no candidate records.")
    for name, scope in corpus["scopes"].items():
        print(f"\n{name}: {counts_text(scope)}")
        print(f"  Papers with eligible records {scope['gold_papers_with_eligible_records']}/"
              f"{scope['candidate_papers_with_eligible_records']}")
        if outside := scope["candidate_outside_gold"]["extracted"]:
            print(f"  Extracted candidate rows outside gold {outside}")
        for score_label, key in (("Source", "source_agreement"), ("Context", "context_agreement")):
            scores = scope[key]
            if scores is None:
                print(f"  {score_label} unavailable (configurations.csv or datasets.csv missing)")
                continue
            print(f"  {paper_score_text(score_label + ' micro', scores['micro'])}")
            print(f"  {score_label} macro ({scores['macro']['papers']} papers), P/R/F1 {ratios_text(scores['macro'])}")
        print_diagnostics(scope, "  ")
    for publication_id, paper in report["documents"].items():
        gold = paper["gold"]["decision"] if paper["in_gold"] else "outside gold"
        print(f"\n{publication_id} gold {gold}, candidate {paper['candidate']['decision']}")
        for side in ("gold", "candidate"):
            reason = paper[side]["exclusion_reason"]
            if paper[side]["decision"] == "excluded" and reason.strip():
                print(f"  {side} exclusion reason {literal(reason)}")
        for name, scope in paper["scopes"].items():
            print(f"  {name}: {counts_text(scope)}")
            if paper["in_gold"]:
                print(f"    {paper_score_text('Source', scope['source_agreement'])}. "
                      f"{paper_score_text('Context', scope['context_agreement'])}.")
                print_diagnostics(scope, "    ")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="candidate CSV folder")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--gold", type=Path, help="extracted dev_gold CSV folder")
    source.add_argument("--corpus-zip", type=Path, help="organizer archive containing dev_gold/")
    parser.add_argument("--json-report", metavar="PATH", help="write JSON to PATH, or use '-' for JSON on stdout")
    parser.add_argument("--diagnostics", action="store_true",
                        help="count additional matches when ignoring one source field at a time")
    args = parser.parse_args(argv)
    try:
        gold = read_directory(args.gold) if args.gold else read_gold_zip(args.corpus_zip)
        if not gold.publications:
            raise ValueError("gold publications.csv has no PDFs")
        report = evaluate(gold, read_directory(args.candidate), diagnostics=args.diagnostics)
        if args.json_report:
            payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            if args.json_report == "-":
                sys.stdout.write(payload)
                return 0
            Path(args.json_report).write_text(payload, encoding="utf-8")
        print_report(report)
    except (OSError, ValueError, csv.Error, zipfile.BadZipFile) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
