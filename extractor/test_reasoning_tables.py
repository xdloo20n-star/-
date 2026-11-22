import unittest

from pdf_pages import PDF, Page
from publication import extract_publication


LAYOUT = """Retrieval-augmented generation.
              Natural Questions Huatuo-26M                                  Natural Questions     Huatuo-26M
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
                                                             Merged KB      41.41    26.80       18.84   19.61
                                                              w/o reasoning 37.11    25.74       18.63   19.13

3.5 Impact of the Reasoning Enhanced                      Table 5: Ablation results on reasoning content of miss-
    Method                                                ing knowledge RAG framework.
"""
MAIN = """Retrieval-augmented generation.
                                          Natural Questions                                           Huatuo-26M
 Methods
                                  EM       F1     Precision   Recall   ROUGE_1     ROUGE_2   ROUGE_L      BLEU_1    BLEU_2    BLEU_3   BLEU_4
 DirectGen                        22.27   17.11    18.86      20.59     13.91        1.46      9.73         9.38      2.80     0.99       0.30
 Misknow-RAG with Wiki            32.03   21.31    23.76      23.56     18.32        2.05     11.62        18.28      5.47     1.86       0.54
 Misknow-RAG with QA Pairs+Wiki   35.94   27.27    29.55      30.02     18.44        2.45     11.70        18.51      6.12     2.48       0.90
Table 1: A comparison of results from different baselines on the NQ and Huatuo test set.

During evaluation, we use normalized exact
match (EM) and word-level F1-score to compare
the final prediction with the ground truth.
"""


class ReasoningTableTest(unittest.TestCase):
    def extract(self, text):
        return extract_publication(PDF({}, [Page(7, text, "text_match")]))

    def test_neighboring_recall_caption_does_not_claim_reasoning_results(self):
        payload = self.extract(LAYOUT)
        evidence = next(e for e in payload["evidence"] if e["field"] == "raw_value" and e["fragment"] == "17.94")
        result = next(r for r in payload["results"] if r["result_id"] == evidence["entity_id"])
        config = next(c for c in payload["configurations"] if c["config_id"] == result["config_id"])
        dataset = next(d for d in payload["datasets"] if d["dataset_id"] == result["dataset_id"])
        self.assertEqual((result["metric_name"], result["reported_value"], dataset["dataset_name"],
                          config["rag_variant"], config["top_k"]),
                         ("ROUGE_1", 17.94, "Huatuo-26M", "Misknow-RAG with Wiki+QA Pairs / w/o reasoning", None))
        self.assertEqual((evidence["page"], evidence["table"], evidence["table_row"], evidence["table_column"]),
                         (7, "Table 5", "Wiki+QA Pairs / w/o reasoning", "Huatuo-26M / ROUGE_1"))
        self.assertEqual(sum(e.get("table") == "Table 5" and e["field"] == "raw_value" for e in payload["evidence"]), 40)

    def test_missing_ablation_caption_prevents_reassigning_the_neighbor(self):
        payload = self.extract(LAYOUT.replace("Ablation results on reasoning content", "A description of our methods"))
        self.assertFalse(any(e.get("table") == "Table 5" and e["field"] == "raw_value" for e in payload["evidence"]))

    def test_repeated_base_scores_and_conflicting_printed_values_are_distinct(self):
        payload = extract_publication(PDF({}, [Page(5, MAIN, "text_match"), Page(7, LAYOUT, "text_match")]))
        precision = next(r for r in payload["results"] if r["raw_value"] == "18.86")
        self.assertEqual(precision["metric_name"], "Answer precision")
        copy = next(r for r in payload["results"] if r["raw_value"] == "32.03" and r.get("duplicate_of"))
        original = next(r for r in payload["results"] if r["result_id"] == copy["duplicate_of"])
        self.assertEqual((original["raw_value"], original["metric_variant"], original["config_id"]),
                         ("32.03", "normalized exact match", copy["config_id"]))
        alternate = next(r for r in payload["results"] if r["raw_value"] == "17.92")
        self.assertIsNone(alternate.get("duplicate_of"))
