import unittest

from pdf_pages import PDF, Page
from publication import extract_publication


# TrustRAG, PDF page 14, Table B1. The first two dataset groups preserve the
# merged cells, wrapped model name, and omitted all-zero comparison rows.
STATISTICS = """                                                    Naive RAG              TrustRAG            ∆      McNemar
Dataset             Failure Model
                                                    % [95% CI]             % [95% CI]          pp        p
                              Qwen-Max    37.50 [32.75, 42.25] 0.00 [0.00, 0.00] 37.50                 1.40e-45
                    Leak      DeepSeek-V3 37.50 [32.75, 42.25] 0.00 [0.00, 0.00] 37.50                 1.40e-45
                              GLM-4-32B-
T EMPLATE                                 35.75 [31.24, 40.50] 0.00 [0.00, 0.00] 35.75                 1.79e-43
                              0414
                           Qwen-Max               5.75 [3.50, 8.00]      0.00 [0.00, 0.00] 5.75        2.38e-07
                    Unsup. DeepSeek-V3            6.25 [4.00, 8.75]      0.00 [0.00, 0.00] 6.25        5.96e-08
                           GLM-4-32B-
                                                  5.25 [3.25, 7.50]      0.00 [0.00, 0.00] 5.25        9.54e-07
                           0414
                              Qwen-Max    38.50 [33.75, 43.25] 0.00 [0.00, 0.00] 38.50                 8.76e-47
                    Stale     DeepSeek-V3 65.75 [61.00, 70.25] 0.00 [0.00, 0.00] 65.75                 1.35e-79
F RESH -S TRESS               GLM-4-32B-
                                          52.50 [47.50, 57.25] 0.00 [0.00, 0.00] 52.50                 1.22e-63
                              0414
                              Qwen-Max            2.50 [1.00, 4.25]      0.00 [0.00, 0.00] 2.50        1.95e-03
                    Unsup.
                              DeepSeek-V3         2.75 [1.25, 4.50]      0.00 [0.00, 0.00] 2.75        9.77e-04

Table B1: Statistical results for target and non-zero failure rows. Failure rates are reported with 95% bootstrap
confidence intervals over 10,000 resamples. ∆ pp denotes the absolute reduction from Naive RAG to TrustRAG
in percentage points. Paired McNemar tests compare Naive RAG and TrustRAG on the same examples. Leak,
Stale, and Unsup. denote unauthorized leakage, stale citation, and unsupported-answer rates, respectively. All-zero
non-target rows are omitted for compactness.
"""


def extract(text, verification="text_match"):
    pages = [Page(1, "Metadata-Governed RAG for Question Answering", "text_match")]
    pages.extend(Page(number, "", "unverified") for number in range(2, 14))
    pages.append(Page(14, text, verification))
    return extract_publication(PDF({"Title": pages[0].text}, pages))


class FailureStatisticsTest(unittest.TestCase):
    def test_preserves_interval_endpoints_and_wrapped_model_coordinates(self):
        payload = extract(STATISTICS)
        self.assertEqual(len(payload["results"]), 88)
        configs = {row["config_id"]: row for row in payload["configurations"]}
        datasets = {row["dataset_id"]: row["dataset_name"] for row in payload["datasets"]}
        locators = {row["entity_id"]: row for row in payload["evidence"]
                    if row["entity_type"] == "result" and row["field"] == "raw_value"}
        lower, = [row for row in payload["results"] if row["raw_value"] == "31.24"]
        locator = locators[lower["result_id"]]
        config = configs[lower["config_id"]]
        self.assertEqual((lower["metric_name"], lower["reported_value"], lower["raw_unit"], lower["reported_scale"]),
                         ("Leakage Rate 95% CI lower", 31.24, "%", "percent"))
        self.assertEqual((config["rag_variant"], config["generator_model"], datasets[lower["dataset_id"]]),
                         ("Naive RAG", "GLM-4-32B-0414", "TEMPLATE"))
        self.assertEqual((locator["page"], locator["table"], locator["table_row"], locator["table_column"],
                          locator["fragment"], locator["verification"]),
                         (14, "Table B1", "TEMPLATE / Leak / GLM-4-32B-0414",
                          "Naive RAG / % [95% CI] / lower bound", "31.24", "text_match"))
        p_value, = [row for row in payload["results"] if row["raw_value"] == "1.35e-79"]
        self.assertEqual((p_value["metric_name"], p_value["reported_value"], p_value["raw_unit"],
                          datasets[p_value["dataset_id"]]),
                         ("Stale Citation Rate paired McNemar p", 1.35e-79, None, "FRESH-STRESS"))
        self.assertEqual(locators[p_value["result_id"]]["table_column"], "McNemar p")
        delta, = [row for row in payload["results"] if row["raw_value"] == "35.75" and row["raw_unit"] == "pp"]
        self.assertEqual((delta["metric_name"], delta["reported_value"], delta["reported_scale"],
                          configs[delta["config_id"]]["rag_variant"], locators[delta["result_id"]]["table_column"]),
                         ("Leakage Rate reduction vs Naive RAG", 35.75, None, "TrustRAG vs Naive RAG", "∆ pp"))
        self.assertFalse(any(
            row["metric_name"].startswith("Unsupported Answer Rate") and datasets[row["dataset_id"]] == "FRESH-STRESS"
            and configs[row["config_id"]]["generator_model"] == "GLM-4-32B-0414"
            for row in payload["results"]
        ))

    def test_rejects_damaged_intervals_or_missing_dataset_labels(self):
        damaged = STATISTICS.replace("[31.24, 40.50]", "[40.50, 31.24]")
        self.assertEqual(extract(damaged)["results"], [])
        unlabeled = STATISTICS.replace("F RESH -S TRESS", "               ")
        self.assertEqual(extract(unlabeled)["results"], [])

    def test_leaves_ocr_statistics_untouched(self):
        self.assertEqual(extract(STATISTICS, "ocr_match")["results"], [])
