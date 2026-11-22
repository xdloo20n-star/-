import unittest

from pdf_pages import PDF, Page
from publication import extract_publication


BREAKDOWN = """RAG QA factuality evaluation.
                                   Correct   Hallucination        Refusal   Unverified   Factuality Score
                   CRAG            59.1%         24.9%            16.0%       0.0%             34.2%
     Baseline
                   HAGRID          93.0%          4.0%             1.0%       2.0%             89.0%
                   MS Macro        90.0%          8.0%             1.0%       1.0%             82.0%
                   CRAG            62.1%         22.9%            15.1%       0.0%             39.2%
     This Work     HAGRID          91.0%          1.0%             7.0%       1.0%             90.0%
                   MS Macro        89.0%          7.0%             1.0%       3.0%             82.0%
                             Table 5: Breakdown of factuality benchmark results.
"""
CLOSED_BOOK = """RAG QA factuality evaluation.
    Model               SimpleQA (closed-book)
                   Acc (Rec)    Miss.   Hall.    Fac.
    Baseline          20.0      44.1    35.9     -15.8
    This Work         14.0      64.0    22.0      -8.0
    Model                 IMDB (closed-book)
                   Acc (Rec)    Miss.   Hall.    Fac.
    Baseline          44.8      34.2    21.0     23.8
    This Work         40.7      47.7    11.7     29.0
 Table 6: Closed-book factuality benchmark results.
"""


class FactualityTableTest(unittest.TestCase):
    def extract(self, text):
        return extract_publication(PDF({}, [Page(12, text, "text_match")]))

    def find(self, payload, raw, metric):
        result = next(item for item in payload["results"] if item["raw_value"] == raw and item["metric_name"] == metric)
        config = next(item for item in payload["configurations"] if item["config_id"] == result["config_id"])
        dataset = next(item for item in payload["datasets"] if item["dataset_id"] == result["dataset_id"])
        source = next(item for item in payload["evidence"] if item["entity_id"] == result["result_id"]
                      and item["field"] == "raw_value")
        return result, config, dataset, source

    def test_centered_method_labels_apply_to_whole_benchmark_block(self):
        payload = self.extract(BREAKDOWN)
        self.assertEqual(len(payload["results"]), 30)
        result, config, dataset, source = self.find(payload, "62.1%", "Accuracy")
        self.assertEqual((config["rag_variant"], dataset["dataset_name"], result["raw_unit"], result["reported_value"]),
                         ("This Work", "CRAG", "%", 62.1))
        self.assertEqual((source["page"], source["table"], source["table_row"], source["table_column"]),
                         (12, "Table 5", "This Work / CRAG", "Correct"))

    def test_ambiguous_groups_are_rejected(self):
        self.assertEqual(self.extract(BREAKDOWN.replace("                   CRAG            62.1%",
                                                       "                   Else            62.1%"))["results"], [])

    def test_closed_book_rows_preserve_negative_factuality_and_unknown_cutoff(self):
        payload = self.extract(CLOSED_BOOK)
        result, config, dataset, source = self.find(payload, "-15.8", "Factuality")
        self.assertEqual((config["rag_variant"], config["top_k"], dataset["dataset_name"]),
                         ("Baseline closed-book", None, "SimpleQA"))
        self.assertEqual((result["reported_value"], result["raw_unit"], result["reported_scale"]), (-15.8, None, "percent"))
        self.assertEqual((source["page"], source["table"], source["table_row"], source["table_column"]),
                         (12, "Table 6", "SimpleQA / Baseline", "Fac."))

    def test_average_row_uses_documented_macro_gain(self):
        grid = """RAG QA factuality evaluation.
    Benchmark               Baseline       NaiveSFT
    CRAG (Yang et al.)          34.2%          27.8%
    CovidQA                     80.0%          83.0%
    Avg.                        57.1%          55.4%

Table 2: Comparison of RAG QA factuality scores for different approaches.
In this comparison we observe an overall macro-
average gain of 5.4% over the baseline.
"""
        statement = "Table 5 presents the breakdown results for the baseline and proposed method."
        for text, expected in ((grid, "macro mean across 2 benchmarks"),
                               (grid.replace("macro-\naverage", "macroaverage"), "macro mean across 2 benchmarks"),
                               (grid.replace("macro-\naverage", "reported"), "mean")):
            payload = extract_publication(PDF({}, [Page(6, text, "text_match"), Page(12, statement, "text_match")]))
            result, config, dataset, source = self.find(payload, "57.1%", "Factuality")
            self.assertEqual((config["rag_variant"], result["aggregation"], result["raw_unit"]),
                             ("Baseline", expected, "%"))
            self.assertEqual((source["page"], source["table"], source["table_row"], source["table_column"]),
                             (6, "Table 2", "Avg.", "Baseline / factuality"))
            self.assertEqual(dataset["dataset_name"], "Macro-average over 2 RAG benchmarks" if expected != "mean" else None)
