import unittest

from pdf_pages import PDF, Page
from publication import extract_publication
from rag_tables import read_tables


# PRO-Step, PDF page 18, Tables 10 and 11.
# https://aclanthology.org/2026.emnlp-main.1667.pdf#page=18
PAIR_GRIDS = """Table 10: Ablation of the VTS branching configuration. The main configuration improves average performance by
1.9 EM and 2.0 F1 over the reduced-branching setting.

Configuration                                      PopQA               HotpotQA           2WikiMulti                 Bamboogle                  MuSiQue                      Average
Reduced branching (K = 2)                        39.5 / 46.5           38.0 / 50.3         39.6 / 48.4               32.8 / 42.5                13.2 / 23.0                32.6 / 42.1
Main configuration (K = 3)                       40.5 / 47.4           38.7 / 51.6         44.1 / 51.4               36.8 / 47.6                12.5 / 22.4                34.5 / 44.1

Table 11: Performance comparison across different Process Reward Model (PRM) weight (α) values. The table
reports EM / F1 scores across 5 benchmarks. ⋆ denotes the selected configuration.

α                           HotpotQA                    PopQA              2WikiMultiHopQA                  Bamboogle                      MuSiQue                        Average
0.0 (no PRM)              36.75 / 49.31           38.48 / 45.01                  41.44 / 48.40              34.40 / 44.20                 12.33 / 22.35                 32.68 / 41.85
0.3 ⋆                     38.73 / 51.63           40.47 / 47.37                  44.07 / 51.43              36.80 / 47.63                 12.49 / 22.41                 34.51 / 44.09
0.5                       34.95 / 47.53           36.62 / 44.33                  41.28 / 48.21              33.60 / 41.91                 10.47 / 20.41                 31.38 / 40.48
1.0                       37.97 / 50.60           39.94 / 46.10                  42.68 / 50.38              32.80 / 41.39                 13.49 / 23.46                 33.38 / 42.39
"""


class PairedScoreTest(unittest.TestCase):
    def test_caption_order_and_parameter_rows_reach_result_evidence(self):
        page = Page(18, PAIR_GRIDS, "text_match")
        pdf = PDF({"Title": "Process rewards for Retrieval-Augmented Generation"}, [page])
        payload = extract_publication(pdf)
        self.assertEqual(len(payload["results"]), 72)
        values = {e["fragment"]: e for e in payload["evidence"] if e["field"] == "raw_value"}
        raw = values["45.01"]
        self.assertEqual((raw["page"], raw["table"], raw["table_row"], raw["table_column"]),
                         (18, "Table 11", "0.0 (no PRM)", "PopQA / F1"))
        result = next(result for result in payload["results"] if result["result_id"] == raw["entity_id"])
        self.assertEqual((result["metric_name"], result["raw_value"], result["reported_value"],
                          result["reported_scale"], result["raw_unit"]), ("F1", "45.01", 45.01, None, None))
        self.assertTrue(any(e["entity_id"] == result["result_id"] and e["field"] == "metric_name"
                            and e["page"] == 18 and e["fragment"] == "EM / F1"
                            and e["verification"] == "text_match" for e in payload["evidence"]))
        dataset = next(dataset for dataset in payload["datasets"] if dataset["dataset_id"] == result["dataset_id"])
        config = next(config for config in payload["configurations"] if config["config_id"] == result["config_id"])
        self.assertEqual((dataset["dataset_name"], config["rag_variant"]), ("PopQA", "0.0 (no PRM)"))
        self.assertEqual((values["39.5"]["table"], values["39.5"]["table_column"]), ("Table 10", "PopQA / EM"))
        self.assertEqual((values["46.5"]["table"], values["46.5"]["table_column"]), ("Table 10", "PopQA / F1"))

    def test_ambiguous_or_unlabeled_pairs_do_not_produce_results(self):
        caption_grid, method_grid = PAIR_GRIDS.split("\n\nTable 11:")
        cases = (
            (caption_grid.replace("1.9 EM and 2.0 F1", "1.8 EM and 2.0 F1"), "text_match"),
            ("Table 11:" + method_grid.replace("EM / F1 scores", "benchmark scores"), "text_match"),
            ("Table 11:" + method_grid.replace("38.48 / 45.01", "      12,483 "), "text_match"),
            ("Table 11:" + method_grid.replace("45.01", "45.O1"), "ocr_match"),
            ("Table 11:" + method_grid.replace("45.01", "45.O1"), "text_match"),
        )
        for layout, verification in cases:
            with self.subTest(verification=verification, caption=layout.splitlines()[0]):
                self.assertEqual(read_tables(Page(18, layout, verification)), [])
