import unittest

from pdf_pages import PDF, Page
from publication import extract_publication


# PRO-Step, PDF pages 9, 19 and 20. Neighboring prose is kept beside the tables.
RECOVERED_EM = """Table 5: Final EM on recovered trajectories. n is given          subsets are consistently smaller because it fails less
as Search-R1 / PRO-S TEP.                                        often at the initial retrieval. On 2WikiMulti, the
                                                                 gains hold at every recovery depth k with sufficient
Dataset            n          Search-R1         PRO-S TEP
                                                                 support (+4.1, +8.3, and +6.8 EM at k = 1, 2, 3),
HotpotQA      1,726 / 1,081      51.3             54.4           and PRO-S TEP concentrates its recoveries earlier,
PopQA           915 / 623        44.6             49.6
                                                                 resolving 87% of its subset at k = 1 versus 80%
2WikiMulti    4,002 / 2,945      47.4             52.8
Bamboogle        34 / 25         82.4             84.0           for Search-R1. The advantage is not driven by
MuSiQue         550 / 418        38.4             38.5           subset selection, as PRO-S TEP also outperforms
"""

RECOVERED_F1_DEPTH = """                                                                   Table 19: Token-level F1 on recovered trajectories.
                                                                Dataset            Search-R1          PRO-S TEP            ∆
                                                                HotpotQA               66.6                 69.0          +2.5
                                                                PopQA                  54.2                 57.6          +3.4
                                                                2WikiMulti             58.2                 66.1          +7.8
                                                                Bamboogle              87.6                 84.0          −3.6
                                                                MuSiQue                48.9                 51.1          +2.1
MuSiQue      1.24 / 6.27   8.07 / 14.27    10.38* / 19.85*     Table 20: Final EM on 2WikiMulti by recovery depth
                                                                k.

                                                                 Method            k=1           k=2          k=3       k≥4
                                                                 n              3,193 / 2,565   578 / 332    150 / 33   81 / 15
                                                                 Search-R1          49.0          42.6         38.7      34.6
                                                                 PRO-S TEP          53.1          50.9         45.5      60.0
                                                                 ∆                  +4.1          +8.3         +6.8     +25.4

Throughout, n is reported as Search-R1 / PRO-S TEP, since the recovered subset is defined by each model’s own search behavior.
Table 19 reports token-level F1 on the same recovered subsets as Table 5.
"""

INITIAL_EM = """Table 21: Final EM when the initial retrieval already     we applied the same outcome-margin filter (chosen
succeeds. Subset sizes (n, ours / Search-R1) are 3,136    F1 ≥ 0.2 and ∆F1 ≥ 0.2) as in our main pipeline.
/ 3,434, 8,597 / 8,776, 4,302 / 4,732, 23 / 25, and
                                                          This yielded 20,567 preference pairs over 3,128
348 / 405, in row order.

Dataset         Search-R1       PRO-S TEP         ∆       questions in the original outcome filter set). We
HotpotQA           53.1            54.1          +1.0
PopQA              62.3            61.0          −1.3
2WikiMulti         42.5            54.6          +12.1
Bamboogle          34.8            64.0          +29.2
MuSiQue            25.3            27.2          +1.9     Training metrics improve while evaluation col-
"""


class RecoveryScoreTest(unittest.TestCase):
    def extract(self, *pages):
        return extract_publication(PDF({"Title": "RAG retrieval recovery"}, list(pages)))

    def at(self, payload, table, row, column):
        source = next((source for source in payload["evidence"] if source["field"] == "raw_value"
                       and source.get("table") == table and source.get("table_row") == row
                       and source.get("table_column") == column), None)
        self.assertIsNotNone(source, (table, row, column))
        result = next(result for result in payload["results"] if result["result_id"] == source["entity_id"])
        config = next(config for config in payload["configurations"] if config["config_id"] == result["config_id"])
        dataset = next(dataset for dataset in payload["datasets"] if dataset["dataset_id"] == result["dataset_id"])
        return source, result, config, dataset

    def test_recovered_results_keep_the_model_specific_subset_size(self):
        payload = self.extract(Page(9, RECOVERED_EM, "text_match"), Page(19, RECOVERED_F1_DEPTH, "text_match"))
        source, result, config, dataset = self.at(payload, "Table 5", "HotpotQA", "Search-R1 / EM")
        self.assertEqual((source["page"], source["verification"], result["metric_name"], result["raw_value"],
                          result["metric_variant"], result["sample_size"], config["rag_variant"], dataset["dataset_name"]),
                         (9, "text_match", "EM", "51.3", "recovered after initial failure", 1726, "Search-R1", "HotpotQA"))
        source, result, _, _ = self.at(payload, "Table 19", "PopQA", "PRO-S TEP / F1")
        self.assertEqual((source["page"], result["metric_name"], result["raw_value"], result["sample_size"]),
                         (19, "F1", "57.6", 623))
        denominator = next(e for e in payload["evidence"] if e["field"] == "sample_size"
                           and e["entity_id"] == result["result_id"])
        self.assertEqual((denominator["page"], denominator["fragment"], denominator["verification"]),
                         (9, "915 / 623", "text_match"))

    def test_recovery_depth_stays_a_condition_of_em(self):
        reading = "Throughout, n is reported as Search-R1 / PROSTEP, since the recovered subset is defined by each model’s own search behavior."
        payload = self.extract(Page(19, RECOVERED_F1_DEPTH, "text_match", reading))
        source, result, config, dataset = self.at(payload, "Table 20", "PRO-S TEP", "k≥4")
        self.assertEqual((source["page"], result["metric_name"], result["raw_value"], result["sample_size"],
                          result["metric_variant"], result["metric_k"], config["top_k"], dataset["dataset_name"]),
                         (19, "EM", "60.0", 15, "recovery depth >=4", None, None, "2WikiMulti"))
        unclear = self.extract(Page(19, RECOVERED_F1_DEPTH.replace("recovery depth", "some parameter"), "text_match"))
        self.assertFalse(any(e.get("table") == "Table 20" for e in unclear["evidence"]))

    def test_initial_success_does_not_attach_an_inconsistent_denominator(self):
        payload = self.extract(Page(20, INITIAL_EM, "text_match"))
        source, result, _, _ = self.at(payload, "Table 21", "HotpotQA", "PRO-S TEP / EM")
        self.assertEqual((source["page"], result["raw_value"], result["metric_variant"], result["sample_size"]),
                         (20, "54.1", "initial retrieval succeeds", 3136))
        _, conflict, _, _ = self.at(payload, "Table 21", "Bamboogle", "PRO-S TEP / EM")
        self.assertEqual(conflict["raw_value"], "64.0")
        self.assertIsNone(conflict.get("sample_size"))
