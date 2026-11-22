import unittest

from pdf_pages import PDF, Page
from publication import extract_publication


# KMW benchmark, PDF page 7. Each printed row contains three independent metric blocks.
EFFECTIVENESS = """                Table 1: Result of Effectiveness Evaluation across Datasets, LLMs, and Baselines.

          Validity         LLM            Quality                 Scheme             Task             Scheme
Dataset
          Metrics                         Metrics                                    Metrics
                     GPT Gemini Qwen                WARD RAGWM KMW(1) KMW                      ORI WARD RAGWM KMW
         WRR 1.00          1.00     1.00 PPLavg ↓ 16.60        221.41   11.23   10.67 CDPA↑ 1.00 0.87      0.96       0.97
NFCorpus WTRR 1.00         1.00     1.00 KCR↑     1.00           0.22    0.83    0.79 CIRA↑ 1.00 0.83      0.88       0.98
         FHR  0.52         0.48     0.55 RSavg ↑ 9.50           -2.31    8.62    8.51 WDPV↑ 0.64 0.63      0.64       0.99
          WRR 1.00         1.00     1.00 PPLavg ↓ 17.52        144.90   12.01   11.71 CDPA↑ 1.00 0.91      0.98       0.99
TREC-
          WTRR 1.00        1.00     0.99 KCR↑     0.97           0.16    0.86    0.88 CIRA↑ 1.00 0.92      0.96       0.97
COVID
          FHR  0.53        0.44     0.49 RSavg ↑ 9.44           -1.87    8.42    8.99 WDPV↑ 0.63 0.61      0.62       0.97
"""
METHODS = """We utilize three mainstream large language models
in experiments, including two advanced black-box
LLMs (GPT-4.1 (OpenAI, 2025) and Gemini-2.5-
Flash (DeepMind, 2025)) and a white-box LLM
(Qwen2.5-32B-Instruct (Team, 2024)).
Baselines. We compare KMW with two formally accepted approaches,
WARD (Jovanovic et al., 2025) and RAG-WM (Lv et al., 2025),
and additionally include KMW in 1-bit mode (KMW(1))
and a non-watermarked RAG (ORI) as baselines.
Experimental results in Table 1 demonstrate the robust watermark verification of KMW across diverse LLMs and datasets.
The estimated overhead of sampling original texts for watermark prefix construction and
linearly generating watermark texts is shown in Table 3.
For RAG systems, unless otherwise specified, top-1 text was retrieved from the knowledge base
to evaluate the retrievability of watermark texts.
"""
SUMMARIES = """
      Table 2: Average PPL of original datasets.

Metrics NFcorpus TREC-COVID
PPLavg     12.08      14.52

     Table 3: Time Overhead of Watermarking.

Metrics              NFcorpus COVID
Prefix/Text (hour)    2.6/4.3     3.2/4.1
"""
INJECTION = """                     Experimental results in Table 10 show that the
Left prose           WRRs of KMW across four datasets are basically 1.00.
                             Table 10: Knowledge Injection Attack
Left prose
                     Injection
Other Table                        NFcorpus         COVID
                     Rate
Left label           20%               1.00           1.00
Other text           40%               1.00           1.00
Left number 999      80%               1.00           0.99
"""


def extract(layout=EFFECTIVENESS, appendix=None):
    pages = [Page(n, "", "unverified") for n in range(1, 22 if appendix else 8)]
    pages[4] = Page(5, METHODS, "text_match")
    pages[6] = Page(7, layout, "text_match")
    if appendix:
        reading = "\n".join(line[21:] for line in appendix.splitlines()[:2])
        pages[20] = Page(21, appendix, "text_match", reading)
    return extract_publication(PDF({"Title": "A RAG benchmark for knowledge watermarking"}, pages))


class WatermarkTableTest(unittest.TestCase):
    def test_multiple_metric_blocks_keep_merged_dataset_and_scheme(self):
        payload = extract()
        self.assertEqual(len(payload["results"]), 66)
        evidence = next(e for e in payload["evidence"] if e["field"] == "raw_value" and e["fragment"] == "-1.87")
        result = next(r for r in payload["results"] if r["result_id"] == evidence["entity_id"])
        config = next(c for c in payload["configurations"] if c["config_id"] == result["config_id"])
        dataset = next(d for d in payload["datasets"] if d["dataset_id"] == result["dataset_id"])
        self.assertEqual((result["metric_name"], result["raw_value"], result["reported_value"],
                          result["raw_unit"], result["metric_direction"]),
                         ("RSavg", "-1.87", -1.87, None, "higher_is_better"))
        self.assertEqual((dataset["dataset_name"], config["rag_variant"], config["generator_model"]),
                         ("TREC-COVID", "RAG-WM", None))
        self.assertEqual((evidence["page"], evidence["table"], evidence["table_row"], evidence["table_column"]),
                         (7, "Table 1", "TREC-COVID / RSavg", "Quality Metrics / Scheme / RAGWM"))

    def test_validity_uses_the_stated_llm_and_retrieval_protocol(self):
        payload = extract()
        evidence = next(e for e in payload["evidence"] if e["field"] == "raw_value" and e["fragment"] == "0.48")
        result = next(r for r in payload["results"] if r["result_id"] == evidence["entity_id"])
        config = next(c for c in payload["configurations"] if c["config_id"] == result["config_id"])
        self.assertEqual((result["metric_name"], config["rag_variant"], config["generator_model"], config["top_k"]),
                         ("FHR", "KMW", "Gemini-2.5-Flash", 1))
        self.assertEqual((evidence["page"], evidence["table_row"], evidence["table_column"]),
                         (7, "NFCorpus / FHR", "Validity Metrics / LLM / Gemini"))
        self.assertTrue(any(e["field"] == "top_k" and e["entity_id"] == config["config_id"] and e["page"] == 5
                            for e in payload["evidence"]))

    def test_damaged_metric_panel_cannot_shift_values_into_other_schemes(self):
        for damaged in (EFFECTIVENESS.replace("0.88 CIRA", "     CIRA"),
                        EFFECTIVENESS.replace("TREC-\n", "TREC\n"),
                        EFFECTIVENESS.replace("Validity", "Unrelated")):
            with self.subTest(damaged=damaged):
                self.assertEqual(extract(damaged)["results"], [])

    def test_neighboring_summary_grid_keeps_pair_component_and_time_unit(self):
        payload = extract(EFFECTIVENESS + SUMMARIES)
        evidence = next(e for e in payload["evidence"] if e.get("table") == "Table 3" and
                        e["field"] == "raw_value" and e["fragment"] == "4.1")
        result = next(r for r in payload["results"] if r["result_id"] == evidence["entity_id"])
        dataset = next(d for d in payload["datasets"] if d["dataset_id"] == result["dataset_id"])
        self.assertEqual((evidence["page"], evidence["table_row"], evidence["table_column"]),
                         (7, "Prefix/Text (hour)", "COVID / Text"))
        self.assertEqual((dataset["dataset_name"], result["metric_name"], result["raw_unit"], result["reported_value"]),
                         ("TREC-COVID", "Watermark text generation time", "hour", 4.1))
        self.assertTrue(any(e.get("table") == "Table 2" and e["field"] == "raw_value" and e["fragment"] == "14.52"
                            for e in payload["evidence"]))

    def test_damaged_or_unresolved_summary_pairs_are_rejected(self):
        for damaged in (SUMMARIES.replace("3.2/4.1", "3.2"), SUMMARIES.replace("NFcorpus COVID", "NFcorpus Unknown")):
            with self.subTest(damaged=damaged):
                self.assertFalse(any(e.get("table") == "Table 3" for e in extract(EFFECTIVENESS + damaged)["evidence"]))

    def test_injection_rate_is_the_row_condition_not_the_score_unit(self):
        payload = extract(appendix=INJECTION)
        evidence = next(e for e in payload["evidence"] if e.get("table") == "Table 10" and
                        e["field"] == "raw_value" and e["fragment"] == "0.99")
        result = next(r for r in payload["results"] if r["result_id"] == evidence["entity_id"])
        dataset = next(d for d in payload["datasets"] if d["dataset_id"] == result["dataset_id"])
        config = next(c for c in payload["configurations"] if c["config_id"] == result["config_id"])
        self.assertEqual((evidence["page"], evidence["table_row"], evidence["table_column"]),
                         (21, "Injection Rate / 80%", "COVID"))
        self.assertEqual((dataset["dataset_name"], result["metric_name"], result["raw_unit"], result["reported_value"],
                          config["rag_variant"], config["top_k"]),
                         ("TREC-COVID", "WRR", None, 0.99, "KMW", 1))

    def test_injection_grid_requires_a_named_metric_and_complete_rows(self):
        for damaged in (INJECTION.replace("WRRs", "Scores"), INJECTION.replace("1.00           0.99", "1.00")):
            with self.subTest(damaged=damaged):
                self.assertFalse(any(e.get("table") == "Table 10" for e in extract(appendix=damaged)["evidence"]))
