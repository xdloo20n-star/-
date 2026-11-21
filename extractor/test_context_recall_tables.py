import unittest

from pdf_pages import PDF, Page
from publication import extract_publication


# Misknow-RAG, PDF pages 5–7. Dataset labels span three rows, and Tables 4 and
# 5 are printed side by side. Preserve the visible cell order in the fixtures.
INDEX_GRID = """records, getting 9M text chunks for the merged                                  Dataset                Knowledge Base # Text Chunks
knowledge base.                                                                                   Wikipedia                     4,760,729
   During evaluation, we use normalized exact                                   Natural Questions History QA Pairs              2,500,931
match (EM) and word-level F1-score to compare                                                     Merged KB                     7,261,660
the final prediction with the ground truth. For the                                                    Wikipedia                 231,528
medical open-domain QA task, we use ROUGE                                       Huatuo-26M             History QA Pairs         8,802,233
                                                                                                       Merged KB                9,033,761
and BLEU scores to evaluate. As the ground-truth
QA pairs are not available across most datasets,
                                                                           Table 2: Text chunk statistics about the knowledge bases
we apply the LLM-as-a-Judge paradigm (Gu et al.,                           across different datasets.
"""
RECALL_GRID = """ously with a reasoner LLM during retrieval and            Dataset          Method               QA Pairs Recall Wiki Recall

generation, considering the interconnection among                           2-way retrieval         48.31          57.83
                                                          Natural Questions Mis Wiki+QA pairs        56.4          57.12
sponding query.                                                            Mis QA Pairs+Wiki       48.51          59.76
                                                                           2-way retrieval         32.87          27.31
                                                          Huatuo-26M       Mis Wiki+QA pairs        35.28          27.23
                                                                           Mis QA Pairs+Wiki        33.05          29.01
3.3 Main Results
As shown in Table 1, we observe that our approach,     Table 3: Results of context recall for QA pairs and
which leverages explicit reasoning based on miss-      Wikipedia chunks across different methods on NQ and
ing knowledge retrieval, achieves gains of 11.41%      Huatuo. Notably, the number of retrieved text chunks is
in EM and 1.65% in ROUGE-1 on both general             identical for evaluation.
"""
QUERY_GRID = """              Natural Questions Huatuo-26M                                  Natural Questions     Huatuo-26M
                                                             Methods
 Methods                                                                     EM       F1        ROUGE_1 BLEU_1
              OrigQ       MisQ       OrigQ MisQ
                                                             Wiki           32.03    21.31       18.32   18.28
 Wiki     54.79           57.48       34.91 35.36             w/o reasoning 30.47    21.29       18.05   17.87
                                                             QA Pairs       33.20    25.22       18.78   18.78
 QA Pairs 36.29           37.11       39.07 41.24             w/o reasoning 32.81    24.65       18.12   17.89
                                                             Wiki+QA Pairs 33.20     24.45       18.06   18.22
Table 4: Evaluation on the retrieved contexts utilizing       w/o reasoning 32.42    23.85       17.94   17.91
original query and the missing knowledge queries with        QA Pairs+Wiki 35.94     27.27       17.92   18.18
recall scores across different datasets.                      w/o reasoning 35.16    27.20       17.66   18.03
"""
SAMPLE = "Limited by our budget, we randomly sample 256 data\nfrom NQ validation set and 128 consultation data from Huatuo26M for evaluation."


class ContextRecallTest(unittest.TestCase):
    def extract(self, index=INDEX_GRID, recall=RECALL_GRID, query=QUERY_GRID):
        pages = [Page(5, index, "text_match", f"{index}\n{SAMPLE}"), Page(6, recall, "text_match"),
                 Page(7, query, "text_match")]
        return extract_publication(PDF({"Title": "Misknow-RAG"}, pages))

    def at(self, payload, table, row, column):
        source = next((e for e in payload["evidence"] if e["field"] == "raw_value" and e.get("table") == table
                       and e.get("table_row") == row and e.get("table_column") == column), None)
        self.assertIsNotNone(source, (table, row, column))
        value = next(r for r in payload["results"] if r["result_id"] == source["entity_id"])
        config = next(c for c in payload["configurations"] if c["config_id"] == value["config_id"])
        dataset = next(d for d in payload["datasets"] if d["dataset_id"] == value["dataset_id"])
        return value, config, dataset, source

    def test_chunk_counts_keep_grouped_digits_and_centered_dataset(self):
        payload = self.extract()
        value, config, dataset, source = self.at(payload, "Table 2", "Natural Questions / Wikipedia", "# Text Chunks")
        self.assertEqual((value["metric_name"], value["raw_value"], value["reported_value"], value["raw_unit"],
                          config["rag_variant"], dataset["dataset_name"], source["page"], source["verification"]),
                         ("Indexed text chunks", "4,760,729", 4760729, None, "Wikipedia", "Natural Questions", 5, "text_match"))

    def test_source_and_query_are_metric_conditions(self):
        payload = self.extract()
        value, config, dataset, source = self.at(payload, "Table 3", "Natural Questions / Mis Wiki+QA pairs", "QA Pairs Recall")
        self.assertEqual((value["metric_name"], value["metric_variant"], value["raw_value"], value["raw_unit"],
                          value["sample_size"], config["rag_variant"], dataset["dataset_name"], source["page"]),
                         ("Context recall", "source-specific QA Pairs", "56.4", None, 256, "Mis Wiki+QA pairs", "Natural Questions", 6))
        value, _, dataset, source = self.at(payload, "Table 4", "QA Pairs", "Huatuo-26M / MisQ")
        self.assertEqual((value["metric_variant"], value["raw_value"], value["sample_size"], dataset["dataset_name"], source["page"]),
                         ("missing query", "41.24", 128, "Huatuo-26M", 7))

    def test_missing_group_and_query_headers_do_not_inherit_conditions(self):
        payload = self.extract(recall=RECALL_GRID.replace("Natural Questions", "                 "),
                               query=QUERY_GRID.replace("OrigQ", "Other"))
        conditions = {r["result_id"] for r in payload["results"] if r["metric_name"] == "Context recall"}
        self.assertFalse(any(e.get("entity_id") in conditions for e in payload["evidence"]))
