import unittest

from evaluation_scan import ACCURACY, GROUNDING, evaluation_tables
from pdf_pages import Page, Word
from publication import Payload, add_table
from scan_ocr import ScanImage


class EvaluationScanTest(unittest.TestCase):
    image = ScanImage(1000, 1000, b"", 300, 1000, 1000)
    page = Page(11, "Results evaluated using the ROC-AUC metric. Table 1. Performance of Deepchecks, RAGAS, "
                   "and Langsmith faithfulness measurement methods in evaluating factual grounding.", "ocr_match")
    headings = (
        Word("Dataset", 345, 602, 383, 614),
        Word("Deepchecks", 435, 606, 475, 617),
        Word("Grounded", 438, 620, 476, 629),
        Word("Context", 445, 630, 483, 640),
        Word("RAGAS", 530, 604, 560, 616),
        Word("Faithfulness", 516, 620, 578, 629),
        Word("(GPT-4o)", 526, 632, 568, 641),
        Word("Langsmith", 609, 602, 661, 616),
        Word("(GPT-4o)", 619, 643, 659, 652),
    )
    dataset = Word("TRUE", 344, 655, 377, 665)

    def test_roc_auc_uses_dataset_and_evaluator_at_cell_location(self):
        score = Word("0.65", 539, 660, 560, 668)
        words = (*self.headings, self.dataset, score)
        table, = evaluation_tables(self.page, GROUNDING, ((self.image, words), (self.image, words)))
        payload = Payload(document={}, publication={})
        add_table(payload, table, True)
        result, = payload.results
        self.assertEqual((result["metric_name"], result["raw_value"], result["reported_value"], result["raw_unit"]),
                         ("ROC-AUC", "0.65", .65, None))
        self.assertEqual((payload.configurations[0]["rag_variant"], payload.configurations[0]["generator_model"],
                          payload.datasets[0]["dataset_name"]), ("RAGAS", "GPT-4o", "TRUE"))
        source, = [e for e in payload.evidence if e["entity_type"] == "result" and e["field"] == "raw_value"]
        self.assertEqual((source["page"], source["table"], source["table_row"], source["table_column"], source["bbox"]),
                         (11, "Table 1", "TRUE", "RAGAS Faithfulness (GPT-4o)", (.539, .659, .560, .669)))

    def test_conflicting_cell_and_missing_dataset_are_rejected(self):
        first = (*self.headings, self.dataset, Word("0.65", 539, 660, 560, 668),
                 Word("0.83", 539, 672, 560, 680))
        second = (*self.headings, self.dataset, Word("0.85", 539, 660, 560, 668),
                  Word("0.83", 539, 672, 560, 680))
        self.assertEqual(evaluation_tables(self.page, GROUNDING, ((self.image, first), (self.image, second))), [])

    def test_accuracy_grid_uses_the_sloping_client_dataset_label(self):
        page = Page(13, "Table 2. Performance comparison measured by accuracy.", "ocr_match")
        words = (
            Word("Dataset", 247, 679, 299, 690),
            Word("Deepchecks", 414, 668, 499, 681),
            Word("RAGAS", 501, 669, 557, 679),
            Word("Langsmith", 638, 670, 707, 682),
            Word("Client", 255, 743, 297, 753),
            Word("2", 303, 740, 312, 751),
            Word("(Job", 320, 741, 348, 750),
            Word("Profiles)", 356, 736, 412, 746),
            Word("0.81", 449, 735, 476, 743),
        )
        table, = evaluation_tables(page, ACCURACY, ((self.image, words), (self.image, words)))
        payload = Payload(document={}, publication={})
        add_table(payload, table, True)
        result, = payload.results
        source, = [e for e in payload.evidence if e["entity_type"] == "result" and e["field"] == "raw_value"]
        self.assertEqual((result["metric_name"], result["raw_value"], result["raw_unit"]),
                         ("Evaluator accuracy", "0.81", None))
        self.assertEqual(payload.configurations[0]["rag_variant"], "Deepchecks")
        self.assertEqual(payload.datasets[0]["dataset_name"], "Client 2 (Job Profiles)")
        self.assertEqual((source["page"], source["table"], source["table_row"], source["table_column"], source["bbox"]),
                         (13, "Table 2", "Client 2 (Job Profiles)", "Deepchecks", (.449, .734, .476, .744)))
        truncated = tuple(w for w in words if w.text != "Profiles)")
        self.assertEqual(evaluation_tables(page, ACCURACY, ((self.image, truncated), (self.image, truncated))), [])


if __name__ == "__main__":
    unittest.main()
