import unittest

from pdf_pages import PDF, Page
from publication import extract_publication
from rag_tables import read_tables
from test_grouped_scores import BACKBONE_GRID


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

# PRO-Step, PDF pages 17 and 21. These captioned grids use the same full
# reference row as the explicitly labeled EM/F1 grid in Table 4.
MARGIN_GRID = """Table 7: Effect of the outcome-margin filter on data scale and downstream performance. Removing the filter
increases the pair count by 2.5× but yields comparable average performance.

Variant                             Pairs    HotpotQA          PopQA         2WikiMulti      Bamboogle        MuSiQue          Average
PRO-S TEP (with margin filter)      15,877   38.73 / 51.63   40.47 / 47.37   44.07 / 51.43   36.80 / 47.63   12.49 / 22.41   34.51 / 44.09
w/o margin filter                   39,578   37.62 / 50.58   39.20 / 46.40   44.47 / 52.62   39.20 / 47.70   12.78 / 23.16   34.65 / 44.09
"""

REGEN_GRID = """Table 23: Downstream evaluation: the regen-augmented variant degrades on every benchmark despite stronger
training metrics.

Variant                     PopQA           HotpotQA         2WikiMulti        Bamboogle        MuSiQue           Average
outcome filter main       40.47 / 47.37    38.73 / 51.63     44.07 / 51.43    36.80 / 47.63    12.49 / 22.41    34.51 / 44.09
outcome filter + regen    28.39 / 37.55    26.02 / 39.08     33.25 / 42.63    24.80 / 35.30    6.70 / 16.80     23.83 / 34.27
∆                        −12.08 / −9.82   −12.71 / −12.55   −10.82 / −8.80   −12.00 / −12.33   −5.79 / −5.61   −10.68 / −9.82
"""

# PDF page 19. Table 19 shares the page beside the transposed score grid.
TRANSPOSED_GRID = """Table 17: Results at the 3B scale (EM / F1, %). Bold                 Table 19: Token-level F1 on recovered trajectories.
denotes the best result; “*” denotes p < 0.05 against
Search-R1.                                                      Dataset            Search-R1          PRO-S TEP            ∆
                                                                HotpotQA               66.6                 69.0          +2.5
Dataset       Base 3B       Search-R1         PRO-Step          PopQA                  54.2                 57.6          +3.4
HotpotQA     5.31 / 17.36   31.82 / 41.30    32.82 / 44.33*     2WikiMulti             58.2                 66.1          +7.8
PopQA        7.24 / 22.23   36.87 / 42.14   39.88* / 46.10*     Bamboogle              87.6                 84.0          −3.6
2WikiMulti   2.27 / 15.28   35.96 / 41.86   44.10* / 50.77*     MuSiQue                48.9                 51.1          +2.1
Bamboogle    4.00 / 10.64   18.40 / 27.97   28.00* / 38.96*
MuSiQue      1.24 / 6.27   8.07 / 14.27    10.38* / 19.85*     Table 20: Final EM on 2WikiMulti by recovery depth
Average      4.01 / 14.35 26.22 / 33.51      31.04 / 40.00      k.
"""


class PairedScoreTest(unittest.TestCase):
    def extract_pages(self, *pages):
        return extract_publication(PDF({"Title": "Process rewards for Retrieval-Augmented Generation"}, list(pages)))

    def result_at(self, payload, table, row, column):
        source = next((source for source in payload["evidence"] if source["field"] == "raw_value"
                       and source.get("table") == table and source.get("table_row") == row
                       and source.get("table_column") == column), None)
        self.assertIsNotNone(source, (table, row, column))
        result = next(result for result in payload["results"] if result["result_id"] == source["entity_id"])
        config = next(config for config in payload["configurations"] if config["config_id"] == result["config_id"])
        dataset = next(dataset for dataset in payload["datasets"] if dataset["dataset_id"] == result["dataset_id"])
        return source, result, config, dataset

    def test_an_explicitly_labeled_reference_row_resolves_unlabeled_score_pairs(self):
        payload = self.extract_pages(Page(9, BACKBONE_GRID, "text_match"), Page(17, MARGIN_GRID, "text_match"))
        source, result, config, dataset = self.result_at(payload, "Table 7", "w/o margin filter", "HotpotQA / F1")
        self.assertEqual((result["metric_name"], result["raw_value"], result["raw_unit"], config["rag_variant"],
                          dataset["dataset_name"], source["page"], source["verification"]),
                         ("F1", "50.58", None, "w/o margin filter", "HotpotQA", 17, "text_match"))
        metric_evidence = next(e for e in payload["evidence"] if e["field"] == "metric_name"
                               and e["entity_id"] == result["result_id"])
        self.assertEqual((metric_evidence["page"], metric_evidence["verification"]), (9, "text_match"))

        unmatched = self.extract_pages(Page(9, BACKBONE_GRID, "text_match"),
                                       Page(17, MARGIN_GRID.replace("47.37", "47.38"), "text_match"))
        self.assertFalse(any(e.get("table") == "Table 7" and e["field"] == "raw_value" for e in unmatched["evidence"]))

    def test_repeated_main_row_requires_explicitly_shared_conditions(self):
        main = Page(9, BACKBONE_GRID, "text_match", """Our main experiments use Qwen2.5-7B-Instruct as
the policy backbone.
Without instruction tuning, the model achieves 32.85 / 42.01 average EM / F1, compared with 34.51 / 44.09 for the main
instruction-tuned backbone.""")
        margin = Page(17, MARGIN_GRID, "text_match")
        regen = Page(21, REGEN_GRID, "text_match",
                     "We applied the same outcome-margin filter as in our main pipeline.")
        payload = self.extract_pages(main, margin, regen)
        _, reference, _, _ = self.result_at(payload, "Table 4", "Qwen2.5-7B Base/Instruct / PRO-S TEP-Instruct",
                                             "PopQA / EM")
        _, main_margin, _, _ = self.result_at(payload, "Table 7", "PRO-S TEP (with margin filter)", "PopQA / EM")
        _, main_regen, _, _ = self.result_at(payload, "Table 23", "outcome filter main", "PopQA / EM")
        self.assertEqual((main_margin["duplicate_of"], main_regen["duplicate_of"]), (reference["result_id"],) * 2)
        _, different, _, _ = self.result_at(payload, "Table 7", "w/o margin filter", "PopQA / EM")
        self.assertIsNone(different.get("duplicate_of"))

        shared_setting_unknown = self.extract_pages(main, margin, Page(21, REGEN_GRID, "text_match"))
        _, main_regen, _, _ = self.result_at(shared_setting_unknown, "Table 23", "outcome filter main", "PopQA / EM")
        self.assertIsNone(main_regen.get("duplicate_of"))

    def test_paired_delta_uses_the_displayed_subtraction(self):
        payload = self.extract_pages(Page(9, BACKBONE_GRID, "text_match"), Page(21, REGEN_GRID, "text_match"))
        source, result, config, dataset = self.result_at(payload, "Table 23", "∆", "PopQA / Delta EM")
        self.assertEqual((result["metric_name"], result["metric_variant"], result["raw_value"], result["reported_value"],
                          config["rag_variant"], dataset["dataset_name"], source["page"], source["verification"]),
                         ("Delta EM", "regen minus outcome-filter main", "−12.08", -12.08,
                          "outcome filter + regen", "PopQA", 21, "text_match"))

        inconsistent = self.extract_pages(Page(9, BACKBONE_GRID, "text_match"),
                                          Page(21, REGEN_GRID.replace("−12.08", "−10.08"), "text_match"))
        self.assertFalse(any(r["metric_name"].startswith("Delta ") for r in inconsistent["results"]))

    def test_transposed_pairs_use_source_dataset_rows_and_explicit_percent_unit(self):
        payload = self.extract_pages(Page(19, TRANSPOSED_GRID, "text_match"))
        source, result, config, dataset = self.result_at(payload, "Table 17", "PopQA", "PRO-Step / EM")
        self.assertEqual((result["metric_name"], result["raw_value"], result["reported_value"],
                          result["reported_scale"], result["raw_unit"], config["rag_variant"],
                          dataset["dataset_name"], source["page"], source["verification"]),
                         ("EM", "39.88", 39.88, "percent", "%", "PRO-Step", "PopQA", 19, "text_match"))
        self.assertEqual(sum(e.get("table") == "Table 17" and e["field"] == "raw_value" for e in payload["evidence"]), 36)

        no_order = self.extract_pages(Page(19, TRANSPOSED_GRID.replace("EM / F1, %", "QA scores"), "text_match"))
        self.assertFalse(any(e.get("table") == "Table 17" for e in no_order["evidence"]))

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
