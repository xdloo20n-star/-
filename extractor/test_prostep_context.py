import unittest

from pdf_pages import PDF, Page
from publication import extract_publication
from test_paired_scores import ROUNDED_ABLATION, ROUNDED_MAIN_GRID


# PRO-Step, PDF page 5 (Table 1 and §4.1) and page 18 (Table 16).
MAIN = ROUNDED_MAIN_GRID.replace(
    "using EM and F1 metrics.", "using EM and F1 metrics. Full significance tests are detailed in Appendix F.",
).replace(
    "PRO-S TEP         40.5",
    "Standard RAG      37.4     44.1    31.6    41.9          28.3    34.6    16.8    25.1    4.8     12.3   23.8   31.6\n"
    "Self-RAG          30.0     38.6    18.2    29.6          13.2    25.0     5.6    18.0    4.7     11.8   14.3   24.6\n"
    "AutoRAG           24.1     31.2    14.8    24.1          11.0    20.4    23.2    34.3    6.5     13.1   15.9   24.6\n"
    "PRO-S TEP         40.5",
)
SETUP = """4.1 Experimental Setup
For single-hop QA, we use PopQA. For multi-hop QA, we use HotpotQA,
2WikiMultiHopQA, Bamboogle, and MuSiQue. We report Exact Match (EM) and F1 score using the
FlashRAG evaluation framework.
Implementation Details. We employ Qwen2.57B-Instruct (Qwen Team, 2024) as the primary
backbone for PRO-S TEP and most baselines. For specific baselines, we utilize their official model
checkpoints to ensure a fair comparison: Self-RAG
employs its specialized Llama-2-13B variant (Asai et al., 2024),
and AutoRAG utilizes Llama-3-8BInstruct (Yu et al., 2024). For retrieval, we use the
2018 Wikipedia dump consisting of 5.9M documents, indexed with the BGE-base-en-v1.5 (Xiao
et al., 2024) retriever, and set the number of retrieved passages to k = 3 across all methods to
ensure consistency.
The main results comparing PRO-S TEP with baseline methods across five datasets are presented in Table 1.
PRO-S TEP (DPO) is our default approach.
"""
SIGNIFICANCE = """F Statistical Significance of Main Results
We supplement Table 1 with bootstrap confidence intervals and per-dataset paired t-tests,
computed on per-question results against PRO-S TEP.
Table 16: Per-dataset paired t-test on EM. n = test split
size.
Dataset     n
HotpotQA  7,405
PopQA    14,267
2WikiMultiHopQA  12,576
Bamboogle  125
MuSiQue   2,417
"""
TITLE = "PRO-Step: Step-level Process Reward Optimization for Retrieval-Augmented Generation"


def payload(setup=SETUP, significance=SIGNIFICANCE):
    pages = [Page(1, TITLE, "text_match"), Page(5, MAIN, "text_match", setup),
             Page(6, ROUNDED_ABLATION, "text_match"),
             Page(8, "Our main experiments use Qwen2.5-7B-Instruct as the policy backbone.", "text_match")]
    if significance:
        pages.append(Page(18, significance, "text_match"))
    return extract_publication(PDF({"Title": TITLE}, pages))


def cell(payload, table, row, column):
    source = next(e for e in payload["evidence"] if e["field"] == "raw_value"
                  and (e.get("table"), e.get("table_row"), e.get("table_column"))
                  == (table, row, f"{column} / mean per question"))
    result = next(r for r in payload["results"] if r["result_id"] == source["entity_id"])
    config = next(c for c in payload["configurations"] if c["config_id"] == result["config_id"])
    dataset = next(d for d in payload["datasets"] if d["dataset_id"] == result["dataset_id"])
    return result, config, dataset, source


class ProstepContextTest(unittest.TestCase):
    def test_main_backbone_and_split_use_linked_evaluation_statements(self):
        actual = payload()
        result, config, dataset, source = cell(actual, "Table 1", "Standard RAG", "2WikiMulti / F1")
        self.assertEqual((result["metric_name"], result["raw_value"], config["generator_model"],
                          config["top_k"], config.get("retriever_model"), dataset["dataset_name"], dataset["data_split"],
                          source["page"], source["verification"]),
                         ("F1", "34.6", "Qwen2.5-7B-Instruct", 3, "BGE-base-en-v1.5", "2WikiMultiHopQA", "test",
                          5, "text_match"))
        split = next(e for e in actual["evidence"] if e["field"] == "data_split" and e["entity_id"] == dataset["dataset_id"])
        self.assertEqual((split["page"], split["verification"]), (18, "text_match"))

    def test_primary_backbone_does_not_overwrite_exceptions_or_assert_direct_retrieval(self):
        actual = payload()
        self_rag = cell(actual, "Table 1", "Self-RAG", "PopQA / EM")[1]
        auto_rag = cell(actual, "Table 1", "AutoRAG", "PopQA / EM")[1]
        direct = cell(actual, "Table 1", "Direct", "PopQA / EM")[1]
        self.assertEqual((self_rag["generator_model"], auto_rag["generator_model"]), ("Llama-2-13B", "Llama-3-8B-Instruct"))
        self.assertIsNone(direct["top_k"])
        self.assertIsNone(direct.get("retriever_model"))

    def test_split_requires_appendix_link_and_confirmed_repeats_keep_the_canonical_context(self):
        actual = payload()
        result, config, dataset, _ = cell(actual, "Table 1", "PRO-S TEP", "HotpotQA / F1")
        repeat, repeat_config, repeat_dataset, _ = cell(actual, "Table 2", "PRO-S TEP", "HotpotQA / F1")
        self.assertEqual((config["rag_variant"], repeat.get("duplicate_of"), dataset["data_split"]),
                         ("PRO-Step", result["result_id"], "test"))
        self.assertEqual((repeat_config, repeat_dataset), (config, dataset))
        unknown = payload(significance=SIGNIFICANCE.replace("test split", "development split"))
        absent_split = cell(unknown, "Table 1", "Standard RAG", "2WikiMulti / F1")[2]
        self.assertIsNone(absent_split["data_split"])
