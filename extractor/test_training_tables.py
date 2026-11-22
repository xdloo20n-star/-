import unittest

from pdf_pages import PDF, Page
from publication import extract_publication


FEVER_LAYOUT = """      Method        Step 25           Step 50          Step 75             tamination.
   PPO          0.394/0.000       0.514/0.000       0.450/0,125              The ablations align with this
   OAS          0.381/0.000      0.458/0.000         0.564/ 0.000
                                                                                                              mechanism, Re-
                                                                           moving old-value anchoring
                                                                                                             increases nVDD,
                                                                          while removing contribution dec
 Table 5: FEVER fact-verification results with Qwen2.5-                                                       ay leaves OAGC
                                                                          close to the PPO baseline, indi
 1.5B-Instruct and fixed Top-3 retrieval. Each entry re-                                                   cating that observa-
                                                                          tion residuals more easily enter
 ports search-on three-way label accuracy /nVDD.                                                           generated-token ad-
"""


class TrainingTablesTest(unittest.TestCase):
    def extract(self, text):
        return extract_publication(PDF({}, [Page(1, "RAG evaluation compares OAS and Search-R1 PPO.", "ocr_match"),
                                           Page(8, text, "ocr_match")]))

    def test_scanned_training_checkpoint_pairs_keep_metric_and_step_separate(self):
        payload = self.extract(FEVER_LAYOUT)
        result = next(row for row in payload["results"] if row["raw_value"] == "0.564")
        config = next(row for row in payload["configurations"] if row["config_id"] == result["config_id"])
        dataset = next(row for row in payload["datasets"] if row["dataset_id"] == result["dataset_id"])
        evidence = next(row for row in payload["evidence"] if row["entity_id"] == result["result_id"]
                        and row["field"] == "raw_value")
        self.assertEqual((result["metric_name"], result["metric_variant"], result["reported_value"], result["metric_k"]),
                         ("Label accuracy", "step 75", 0.564, None))
        self.assertEqual((config["rag_variant"], config["generator_model"], config["top_k"], dataset["dataset_name"]),
                         ("OAS", "Qwen2.5-1.5B-Instruct", 3, "FEVER"))
        self.assertEqual((evidence["page"], evidence["table"], evidence["table_row"], evidence["table_column"],
                          evidence["verification"]), (8, "Table 5", "OAS", "Step 75 / label accuracy", "ocr_match"))
        self.assertEqual(len(payload["results"]), 11)
        self.assertNotIn("0,125", {row["raw_value"] for row in payload["results"]})

    def test_without_printed_metric_order_no_checkpoint_pair_is_created(self):
        self.assertEqual(self.extract(FEVER_LAYOUT.replace("label accuracy /nVDD", "diagnostics"))["results"], [])

    def test_corroboration_recovers_decimal_misread_by_ocr(self):
        prose = "\n accuracy at step 50 but falls to 0.450 at step 75,\n yielding a final nVDD of 0.125. OAS, in contrast,\n"
        payload = self.extract(FEVER_LAYOUT + prose)
        result = next(row for row in payload["results"] if row["raw_value"] == "0.125")
        self.assertEqual((result["metric_name"], result["metric_variant"], len(payload["results"])), ("nVDD", "step 75", 12))
        wrong_step = self.extract(FEVER_LAYOUT + prose.replace("step 75", "step 100"))
        self.assertEqual(len(wrong_step["results"]), 11)
