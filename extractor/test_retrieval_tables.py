import unittest

from pdf_pages import PDF, Page
from publication import extract_publication


SETUP = """Retrieval-augmented generation
We utilize the test sets and relevance annotations provided by these benchmarks.
4.1.2. Retrievers and Rerankers
For each query, the retriever fetches top-10 documents, reranked by the specified method, including LCR.
For LLM-based rerankers, we use Qwen2.5-7B-Instruct.
4.1.3. Confidence Quantification Setup
"""
RANKING = """Table 3
Comparative Performance of Contriever with Various Rerankers and LCR Enhancement. NDCG@5 scores for the
Contriever retriever combined with various rerankers, with and without LCR enhancement, across datasets (as defined in
Table 2). Here, “+LCR w/o QT” denotes LCR without the query threshold, and “+LCR” includes it. Bold and underlined
values follow the conventions in Table 2, with AVG denoting the average across datasets.

                                                      BEIR                                            TREC
  Method                   NQ       TOUC      SCID    NFCO      DBPE      FEVE      AVG      DL19      DL20     AVG
  Retriever-Only
  -                       .4554     .2612     .1180   .3580     .4285     .7425     .3939    .6858     .4575    .5717
      +LCR w/o QT         .4607     .2675     .1185   .3580     .4294     .7431     .3962    .6974     .4710    .5842
      +LCR                .4607     .2804     .1195   .3589     .4364     .7431     .3998    .6979     .4710    .5845
  Pre-trained LLM Based Rerankers
  YesNo                   .2312     .2476     .0778   .2645     .2928     .2546     .2281    .6990     .5285    .6137
    +LCR w/o QT           .3515     .2577     .0937   .2882     .3512     .4689     .3019    .7088     .5317    .6202
    +LCR                  .3515     .2590     .0937   .2882     .3512     .4689     .3021    .7089     .5373    .6231
"""
RETRIEVAL_AVERAGES = """We evaluated retrieval quality by comparing W-RAC against the baseline agentic chunking approach across the RAG-
Multi-Corpus benchmark.

 Organization           Method     Avg Recall@6    Avg Recall@3    Avg Precision@6    Avg Precision@3    Avg MRR      Avg NDCG@6      Avg NDCG@3     Count
 ZX Bank                Baseline            0.93            0.88               0.39               0.54        0.87            0.88            0.87     200
 ZX Bank                W-RAC               0.88            0.80               0.61               0.81        0.82            0.84            0.81     200

                                              Table 8: Retrieval Performance by Organization
"""
ORGANIZATION_EFFICIENCY = """W-RAC processes the same 236 files with a total content length of 1,062,085 characters.

                                             Total     Input   Output     Time     Total   Avg Input   Avg Output    Avg Time        P90        P95
 Organization           Method
                                           Length    Tokens    Tokens        (s)   Files     Tokens        Tokens           (s)   Time (s)   Time (s)
 Velvera Technologies   Agentic Chunking   191,162    96,826    60,202   379.01      38     2,548.05      1,584.26        9.97      15.13      18.94
 Velvera Technologies   W-RAC              191,162   133,230     7,022    118.6      38     3,506.05        184.79        3.12       4.86       6.52

                                     Table 4: Token and Runtime Comparison by Organization.
"""


class RetrievalTablesTest(unittest.TestCase):
    def extract(self, layout):
        return extract_publication(PDF({}, [Page(10, SETUP, "text_match"), Page(13, layout, "text_match")]))

    def test_reranker_comparison_keeps_matching_retriever_and_threshold(self):
        document = self.extract(RANKING)
        self.assertEqual(len(document["results"]), 60)
        result = next(row for row in document["results"] if row["raw_value"] == ".3019")
        config = next(row for row in document["configurations"] if row["config_id"] == result["config_id"])
        dataset = next(row for row in document["datasets"] if row["dataset_id"] == result["dataset_id"])
        evidence = next(row for row in document["evidence"] if row["entity_id"] == result["result_id"]
                        and row["field"] == "raw_value")
        self.assertEqual((result["metric_name"], result["metric_k"], result["reported_value"],
                          result["raw_unit"], result["aggregation"]),
                         ("NDCG@5", 5, 0.3019, None, "mean across datasets"))
        self.assertEqual((config["rag_variant"], config["retriever_model"], config["generator_model"], config["top_k"]),
                         ("YesNo + LCR w/o query threshold", "Contriever", "Qwen2.5-7B-Instruct", 10))
        self.assertEqual((dataset["dataset_name"], dataset["data_split"]),
                         ("Mean over six BEIR test datasets", None))
        self.assertEqual((evidence["page"], evidence["table"], evidence["table_row"], evidence["table_column"],
                          evidence["fragment"], evidence["verification"]),
                         (13, "Table 3", "YesNo / +LCR w/o QT", "BEIR / AVG", ".3019", "text_match"))

    def test_generic_comparison_without_metric_caption_is_ignored(self):
        self.assertEqual(self.extract(RANKING.replace("NDCG@5", "Scores"))["results"], [])

    def test_misaligned_orphan_enhancement_does_not_become_a_method(self):
        malformed = RANKING.replace(".4554", "?????")
        rows = self.extract(malformed)["results"]
        self.assertEqual(len(rows), 30)
        self.assertNotIn(".4607", {row["raw_value"] for row in rows})

    def test_average_retrieval_scores_keep_query_count_and_cutoff_separate(self):
        document = self.extract(RETRIEVAL_AVERAGES)
        self.assertEqual(len(document["results"]), 14)
        result = next(row for row in document["results"] if row["raw_value"] == "0.81"
                      and row["metric_name"] == "Precision@3")
        config = next(row for row in document["configurations"] if row["config_id"] == result["config_id"])
        dataset = next(row for row in document["datasets"] if row["dataset_id"] == result["dataset_id"])
        evidence = next(row for row in document["evidence"] if row["entity_id"] == result["result_id"]
                        and row["field"] == "raw_value")
        self.assertEqual((result["metric_k"], result["sample_size"], result["aggregation"], result["raw_unit"]),
                         (3, 200, "mean over queries", None))
        self.assertEqual((config["rag_variant"], config["top_k"], dataset["dataset_name"], dataset["data_split"]),
                         ("W-RAC", None, "RAG-Multi-Corpus / ZX Bank", None))
        self.assertEqual((evidence["page"], evidence["table"], evidence["table_row"], evidence["table_column"]),
                         (13, "Table 8", "ZX Bank / W-RAC", "Avg Precision@3"))

    def test_nonretrieval_caption_cannot_claim_average_scores(self):
        rows = self.extract(RETRIEVAL_AVERAGES.replace("Retrieval Performance", "Unrelated Scores"))["results"]
        self.assertEqual(rows, [])

    def test_organization_efficiency_keeps_thousands_separator_and_file_count(self):
        document = self.extract(ORGANIZATION_EFFICIENCY)
        result = next(row for row in document["results"] if row["raw_value"] == "2,548.05")
        config = next(row for row in document["configurations"] if row["config_id"] == result["config_id"])
        dataset = next(row for row in document["datasets"] if row["dataset_id"] == result["dataset_id"])
        evidence = next(row for row in document["evidence"] if row["entity_id"] == result["result_id"]
                        and row["field"] == "raw_value")
        self.assertEqual(len(document["results"]), 20)
        self.assertEqual((result["metric_name"], result["reported_value"], result["sample_size"],
                          result["aggregation"], result["raw_unit"]),
                         ("Mean input tokens per file", 2548.05, 38, "per-file mean", "tokens/file"))
        self.assertEqual((config["rag_variant"], dataset["dataset_name"]),
                         ("Agentic Chunking", "Velvera Technologies"))
        self.assertEqual((evidence["page"], evidence["table"], evidence["table_row"], evidence["table_column"]),
                         (13, "Table 4", "Velvera Technologies / Agentic Chunking", "Avg Input Tokens"))

    def test_organization_efficiency_rejects_short_rows(self):
        rows = self.extract(ORGANIZATION_EFFICIENCY.replace(" 96,826", "       "))["results"]
        self.assertEqual(rows, [])
