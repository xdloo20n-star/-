import unittest

from pdf_pages import PDF, Page, Word
from publication import extract_publication


# Adaptive-RAG, PDF page 8, Tables 3–4. The QA and classifier scores
# share the right-hand table and must not be assigned the left-hand units.
CLASSIFIER_LAYOUT = """Table 3: The exact elapsed time per query and the percentage    Table 4: Results on QA and complexity classification with
of the predicted labels from the classifier over all samples.   varying the data annotation strategies for training the classifier.

   Labels        Time/Query (Sec.)       Percentage (%)                                         QA       Classifier (Accuracy)
                                                                   Training Strategies     F1    Step   All    No    One Multi
   No (A)                0.35                  8.60
                                                                   Adaptive-RAG (Ours) 46.94 1084 54.52 30.52 66.28 65.45
   One (B)               3.08                 53.33
                                                                   w/o Binary             43.43 640 60.30 62.19 65.70 39.55
   Multi (C)            27.18                 38.07                w/o Silver             48.79 1464 40.00 0.00 53.98 75.91

Classifier Performance To understand how the                   bias in datasets (see Section 3.2). As Table 4 shows,
"""

SIZE_LAYOUT = """    Table 6: Results with varying model sizes for classifiers.                    on a collection of open-domain QA datasets, cover-
                           QA              Classifier (Accuracy)                  ing the multiple query complexities including both
    Sizes             F1      Step       All      No      One Multi               the single- and multi-hop questions. The results
    Small (60M) 45.83 964 53.48 26.65 70.62 53.18
                                                                                  demonstrate that our Adaptive-RAG enhances the
    Base (223M) 45.97 983 53.41 26.42 69.46 56.82                                 overall accuracy and efficiency of QA systems, al-
    Large (770M) 46.94 1084 54.52 30.52 66.28 65.45                               locating more resources to handle complex queries
"""

ROUTER_ABLATION = """Table 2. Ablation study of router’s input features. Using all three features results in
the highest performance on both benchmarks.

                                                       Recall
                   query centroid source-id
                                              MIRAGE FeB4RAG
                     ✓       ✓        ✗        81.53        89.07
                     ✓       ✗        ✓        82.33        88.33
                     ✓       ✓        ✓        83.35        89.70
"""


def router_words(bad_x=False):
    rows = (
        (119.8, "Table 2. Ablation study of router’s input features.", (147, 168, 193, 225, 243, 265, 295, 325)),
        (154.2, "Recall", (357,)),
        (159.7, "query centroid source-id", (223, 254, 291)),
        (172.1, "MIRAGE FeB4RAG", (333, 380)),
        (187.1, "✓ ✓ ✗ 81.53 89.07", (223, 254, 291, 359 if bad_x else 333, 380)),
        (198.0, "✓ ✗ ✓ 82.33 88.33", (223, 254, 291, 333, 380)),
        (209.0, "✓ ✓ ✓ 83.35 89.70", (223, 254, 291, 333, 380)),
    )
    return tuple(Word(text, x - 3, y - 3, x + 3, y + 3)
                 for y, content, xs in rows for text, x in zip(content.split(), xs, strict=True))


def size_table_words():
    rows = (
        (246.7, "Table 6:", (88.4, 104.0)),
        (278.4, "Sizes F1 Step All No One Multi the", (89.6, 147.2, 171.3, 195.3, 220.4, 245.5, 271.3, 312.9)),
        (296.1, "Small (60M) 45.83 964 53.48 26.65 70.62 53.18", (91.1, 115.3, 147.2, 171.3, 195.3, 220.4, 245.5, 271.3)),
        (290.7, "demonstrate", (333.0,)),
        (305.8, "Base (223M) 45.97 983 53.41 26.42 69.46 56.82 overall", (89.2, 113.6, 147.2, 171.3, 195.3, 220.4, 245.5, 271.3, 321.2)),
        (315.5, "Large (770M) 46.94 1084 54.52 30.52 66.28 65.45", (91.5, 118.4, 147.2, 171.3, 195.3, 220.4, 245.5, 271.3)),
    )
    return tuple(Word(text, x - 4, y - 4, x + 4, y + 4)
                 for y, content, xs in rows for text, x in zip(content.split(), xs, strict=True))


class ClassifierTableTest(unittest.TestCase):
    def extract(self, layout, page_number=8, words=()):
        page = Page(page_number, layout, "text_match", words=words, size=(595, 842) if words else None)
        return extract_publication(PDF({"Title": "Adaptive question answering via retrieval-augmented generation"},
                                       [page]))

    def result_at(self, payload, table, row, column):
        source = next((item for item in payload["evidence"] if item["entity_type"] == "result"
                       and item["field"] == "raw_value" and item.get("table") == table
                       and item.get("table_row") == row and item.get("table_column") == column), None)
        self.assertIsNotNone(source)
        result = next(row for row in payload["results"] if row["result_id"] == source["entity_id"])
        config = next(row for row in payload["configurations"] if row["config_id"] == result["config_id"])
        return result, config, source

    def test_neighboring_classifier_tables_keep_their_conditions_and_values(self):
        payload = self.extract(CLASSIFIER_LAYOUT)
        latency, route, left = self.result_at(payload, "Table 3", "No (A)", "Time/Query (Sec.)")
        accuracy, strategy, right = self.result_at(payload, "Table 4", "w/o Silver", "Classifier (Accuracy) / No")
        self.assertEqual((latency["metric_name"], latency["raw_value"], latency["raw_unit"], route["rag_variant"],
                          left["page"], left["verification"]),
                         ("Latency", "0.35", "Sec.", "No (A)", 8, "text_match"))
        self.assertEqual((accuracy["metric_name"], accuracy["raw_value"], accuracy["reported_value"],
                          accuracy["raw_unit"], accuracy["metric_variant"], strategy["rag_variant"],
                          right["page"], right["verification"]),
                         ("Classifier accuracy", "0.00", 0, None, "A (No)", "w/o Silver", 8, "text_match"))
        self.assertEqual(len(payload["results"]), 24)

    def test_incomplete_classifier_row_does_not_shift_later_scores(self):
        payload = self.extract(CLASSIFIER_LAYOUT.replace("1084 54.52", "     54.52"))
        self.assertFalse(any(item.get("table") == "Table 4" for item in payload["evidence"]))
        self.assertEqual(len(payload["results"]), 6)

    def test_size_rows_use_physical_columns_when_layout_text_compresses_them(self):
        words = size_table_words()
        payload = self.extract(SIZE_LAYOUT, 9, words)
        accuracy, condition, source = self.result_at(payload, "Table 6", "Base (223M)", "Classifier (Accuracy) / No")
        self.assertEqual((accuracy["metric_name"], accuracy["raw_value"], accuracy["metric_variant"],
                          condition["rag_variant"], source["page"], source["verification"]),
                         ("Classifier accuracy", "26.42", "A (No)", "Base (223M)", 9, "text_match"))
        self.assertAlmostEqual(source["bbox"][0], 216.4 / 595)
        self.assertAlmostEqual(source["bbox"][1], 301.8 / 842)
        missing_cell = self.extract(SIZE_LAYOUT, 9, tuple(word for word in words if word.text != "26.42"))
        self.assertEqual(missing_cell["results"], [])

    def test_router_ablation_uses_aligned_features_and_datasets(self):
        payload = self.extract(ROUTER_ABLATION, 14, router_words())
        recall, config, source = self.result_at(payload, "Table 2", "✓ ✓ ✗", "FeB4RAG / Recall")
        dataset = next(d for d in payload["datasets"] if d["dataset_id"] == recall["dataset_id"])
        self.assertEqual((recall["metric_name"], recall["raw_value"], recall["raw_unit"], dataset["dataset_name"]),
                         ("Recall", "89.07", None, "FeB4RAG"))
        self.assertIn("query, centroid", config["note"])
        self.assertIsNone(config["generator_model"])
        self.assertEqual(source["page"], 14)
        self.assertAlmostEqual(source["bbox"][0], 377 / 595)
        self.assertEqual(self.extract(ROUTER_ABLATION, 14, router_words(bad_x=True))["results"], [])
