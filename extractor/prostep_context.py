"""Attach the main PRO-Step comparison protocol to its benchmark cells."""

from dataclasses import replace
import re

from grouped_scores import native_evidence
from paired_scores import MAIN_BACKBONE, MAIN_TABLE, dataset_key, score_locator
from pdf_pages import Page
from rag_tables import Table
from retrieval_tables import source_statement


TITLE = re.compile(r"\b(PRO-Step):\s+Step-level Process Reward Optimization\b", re.I)
CAPTION = re.compile(r"\bComparison of different RAG methods across (?:5|five) benchmarks using EM and F1 metrics\b", re.I)
PRIMARY = re.compile(
    r"\bwe employ\s+Qwen2\.5-?\s*7B-Instruct\b.{0,100}\bas the primary\s+backbone for "
    r"PRO-?S\s*TEP and most baselines\b", re.I | re.S,
)
EXCEPTIONS = {
    "Self-RAG": (re.compile(r"\bSelf-RAG\s+employs its specialized Llama-2-13B variant\b", re.I), "Llama-2-13B"),
    "AutoRAG": (re.compile(r"\bAutoRAG utilizes Llama-3-8B-?\s*Instruct\b", re.I), "Llama-3-8B-Instruct"),
}
BASELINES = {"Direct", "Standard RAG", "FLARE", "Self-RAG", "IRCoT", "Search-o1", "AutoRAG",
             "Search-R1", "ReasonRAG", "StepSearch"}
RETRIEVAL = re.compile(
    r"\bindexed with the\s+(BGE-base-en-v1\.5)\b.{0,100}\bretriever, and set the number of "
    r"re(?:-\s*)?trieved passages to k\s*=\s*(\d+) across all methods\b", re.I | re.S,
)
APPENDIX_LINK = re.compile(r"\bfull significance tests\b.{0,110}\bAppendix F\b", re.I)
APPENDIX_SCOPE = re.compile(r"\bWe supplement\s+(Table \d+)\s+with bootstrap confi-?\s*dence "
                            r"intervals and per-dataset paired t-tests\b", re.I)
TEST_SPLIT = re.compile(r"\bTable \d+:\s*Per-dataset paired t-test on EM\.\s*n\s*=\s*test split\s+size\.", re.I)
DATASETS = ("PopQA", "HotpotQA", "2WikiMultiHopQA", "Bamboogle", "MuSiQue")
CANONICAL_DATASET = {name.casefold(): name for name in DATASETS}


def test_split_evidence(pages: list[Page], label: str) -> tuple[Page, str] | None:
    section = source_statement(pages, APPENDIX_SCOPE)
    if not section or section[1][1] != label:
        return None
    for page in pages:
        if not section[0].number <= page.number <= section[0].number + 2:
            continue
        text = page.reading_text or page.text
        if match := TEST_SPLIT.search(text):
            body = text[match.end():match.end() + 1100]
            if all(re.search(rf"\b{re.escape(dataset)}\b", body) for dataset in DATASETS):
                return page, match[0]
    return None


def enrich_prostep_context(pages: list[Page], tables: list[Table]) -> list[Table]:
    native = [page for page in pages if page.verification == "text_match"]
    title = source_statement(native, TITLE)
    main_statement = source_statement(native, MAIN_TABLE)
    backbone = native_evidence(native, MAIN_BACKBONE)
    if not (title and main_statement and backbone):
        return tables
    setup_page, statement = main_statement
    main_method, label = statement.groups()
    primary = source_statement([setup_page], PRIMARY)
    if primary is None:
        return tables
    method_name = title[1][1]
    model = MAIN_BACKBONE.fullmatch(backbone[1])[1]
    retrieval = source_statement([setup_page], RETRIEVAL)
    split = test_split_evidence(native, label)

    main = [table for table in tables if table.page.number == setup_page.number and table.label == label
            and table.family == "grouped_scores" and CAPTION.search(table.caption)]
    if not main:
        return tables
    if not all(APPENDIX_LINK.search(table.caption) for table in main):
        split = None
    repeats = {column.duplicate_scope for table in main for column in table.columns if column.duplicate_scope}
    result = []
    for table in tables:
        linked = bool(repeats) and all(column.duplicate_scope in repeats for column in table.columns)
        if table not in main and not linked:
            result.append(table)
            continue
        for index, (row, values) in enumerate(table.rows):
            method = main_method if linked else row
            generator, evidence = model, primary
            if method in EXCEPTIONS:
                pattern, generator = EXCEPTIONS[method]
                evidence = source_statement([setup_page], pattern)
                if not evidence:
                    generator = None
            if method not in BASELINES and method != main_method:
                generator, evidence = None, None
            retrieved = retrieval if method != "Direct" and (method in BASELINES or method == main_method) else None
            columns = []
            for column in table.columns:
                dataset = CANONICAL_DATASET.get(dataset_key(column.dataset), column.dataset)
                test = split if dataset in DATASETS else None
                columns.append(replace(column, generator=generator or column.generator,
                                       generator_evidence=(evidence[0], evidence[1][0]) if evidence else column.generator_evidence,
                                       dataset=dataset, source_label=column.source_label or score_locator(column),
                                       data_split="test" if test else column.data_split,
                                       data_split_evidence=test or column.data_split_evidence))
            result.append(replace(
                table, rows=[(row, values)], columns=columns,
                config_name=method_name if method == main_method else table.config_name,
                retriever_model=retrieved[1][1] if retrieved else table.retriever_model,
                top_k=int(retrieved[1][2]) if retrieved else table.top_k,
                top_k_evidence=(retrieved[0], retrieved[1][0]) if retrieved else table.top_k_evidence,
                value_bboxes=table.value_bboxes[index:index + 1] if table.value_bboxes else (),
            ))
    return result
