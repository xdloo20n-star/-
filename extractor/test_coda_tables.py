import unittest

from pdf_pages import PDF, Page, Word
from publication import extract_publication


CAPTION = "Table 1: Contextual faithfulness on CoFaithfulQA."
DEFINITION = ("Context Recall (ConR, Higher is Better)\n"
              "ConR measures the proportion of generated facts that are explicitly supported by the\nretrieved context.")


def page(number, caption, rows):
    words = tuple(Word(value, x - 2, y - 2, x + 2, y + 2)
                  for y, cells in rows for x, value in cells)
    lines = ["   ".join(value for _, value in cells) for _, cells in rows]
    return Page(number, "\n".join([*lines, caption]), "text_match", words=words, size=(595, 842))


class CoDATableTest(unittest.TestCase):
    def extract(self, *pages):
        return extract_publication(PDF({"Title": "CoDA for retrieval-augmented generation"},
                                       [*pages, Page(13, DEFINITION, "text_match")]))

    def at(self, payload, table, row, col):
        source = next((e for e in payload["evidence"] if e["field"] == "raw_value"
                       and e.get("table") == table and e.get("table_row") == row and e.get("table_column") == col), None)
        self.assertIsNotNone(source, (table, row, col))
        value = next(r for r in payload["results"] if r["result_id"] == source["entity_id"])
        config = next(c for c in payload["configurations"] if c["config_id"] == value["config_id"])
        dataset = next(d for d in payload["datasets"] if d["dataset_id"] == value["dataset_id"])
        return value, config, dataset, source

    def recall(self, *, shift=0):
        positions = (366, 394, 420, 451, 480, 507)
        rows = [(328, [(318, "Models"), *zip(positions, ("HotPotQA", "NQ", "NewsQA", "SearchQA", "SQuAD", "TriviaQA"))]),
                (336, [(x - 3, "ConR") for x in positions]),
                (339, [(x + 9, "↑") for x in positions]),
                (348, [(326, "Vanilla-RAG"), *zip(positions, ("60.34", "53.09", "60.27", "66.76", "77.93", "61.80"))]),
                (356, [(325, "Attr_prompt"), *zip((366, 394 + shift, 420, 451, 480, 507),
                                                 ("58.93", "55.36", "58.80", "62.53", "77.35", "59.97"))]),
                (373, [(317, "Table"), (334, "1:"), (365, "Contextual"), (410, "faithfulness"),
                       (441, "on"), (479, "CoFaithfulQA.")])]
        return page(6, CAPTION, rows)

    def test_context_recall_preserves_dataset_column_and_numeric_source(self):
        result, config, dataset, source = self.at(self.extract(self.recall()), "Table 1", "Vanilla-RAG", "NQ / ConR")
        self.assertEqual((result["metric_name"], result["raw_value"], result["raw_unit"], result["reported_scale"],
                          result["metric_variant"], config["rag_variant"], config["generator_model"],
                          dataset["dataset_name"], source["page"]),
                         ("ConR", "53.09", None, "percent", "context-supported facts / all generated facts",
                          "Vanilla-RAG", None, "NQ", 6))
        self.assertEqual(source["bbox"], (392 / 595, 346 / 842, 396 / 595, 350 / 842))

    def test_shifted_cell_does_not_inherit_the_dataset(self):
        payload = self.extract(self.recall(shift=13))
        self.assertFalse(any(r["metric_name"] == "ConR" for r in payload["results"]))

    def test_latency_overhead_preserves_printed_percent_and_model(self):
        caption = "Table 2: Inference latency with and without CoDA."
        rows = [(78, [(323, "Dataset"), (360, "Model"), (399, "Baseline"), (418, "(s)"), (439, "CoDA"),
                      (454, "(s)"), (473, "∆(s)"), (502, "Overhead"), (523, "(%)")]),
                (91, [(329, "RAGTruth")]),
                (103, [(360, "Llama3-8B"), (404, "7.82"), (444, "9.13"), (473, "1.31"), (508, "16.74%")]),
                (134, [(320, "Dolly"), (338, "(AC)")]),
                (147, [(360, "Llama3-8B"), (404, "7.10"), (444, "8.28"), (473, "1.18"), (508, "16.62%")]),
                (182, [(317, "Table"), (334, "2:"), (360, "Inference"), (395, "latency"), (421, "with"),
                       (439, "and"), (464, "without"), (496, "CoDA.")])]
        payload = self.extract(page(8, caption, rows))
        result, config, dataset, source = self.at(payload, "Table 2", "Dolly (AC) / Llama3-8B", "Overhead (%)")
        self.assertEqual((result["metric_name"], result["raw_value"], result["raw_unit"], config["rag_variant"],
                          config["generator_model"], dataset["dataset_name"], source["page"]),
                         ("Relative latency overhead", "16.62%", "%", "CoDA", "Llama3-8B", "Dolly (AC)", 8))
        self.assertEqual(source["bbox"], (506 / 595, 145 / 842, 510 / 595, 149 / 842))

    def test_prompt_latency_keeps_seconds_and_unknown_conditions(self):
        caption = "Table 4: Inference latency under different prompt lengths."
        rows = [(79, [(97, "Prompt"), (128, "Length"), (159, "(tokens)"), (203, "Baseline"), (226, "(s)"),
                      (254, "CoDA"), (273, "(s)")]),
                (95, [(128, "100"), (208, "26.7219"), (260, "33.4681")]),
                (105, [(128, "259"), (208, "28.3332"), (260, "32.9018")]),
                (158, [(82, "Table"), (102, "4:"), (133, "Inference"), (172, "latency"), (203, "under"),
                       (237, "different"), (274, "prompt")])]
        result, config, dataset, source = self.at(self.extract(page(14, caption, rows)), "Table 4", "100", "CoDA (s)")
        self.assertEqual((result["metric_name"], result["metric_variant"], result["raw_value"], result["raw_unit"],
                          config["rag_variant"], config["generator_model"], dataset["dataset_name"], source["page"]),
                         ("Latency", "input prompt 100 tokens", "33.4681", "s", "CoDA", None, None, 14))
        self.assertEqual(source["bbox"], (258 / 595, 93 / 842, 262 / 595, 97 / 842))
