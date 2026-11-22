from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from pdf_pages import Page, ScanSource, Word
from publication import Payload, add_table
from scan_ocr import ScanImage
from scanned_score_tables import CVE_GRID, bootstrap_cell, latency_table, metric_columns, read_scanned_score_tables


CAPTION = "TABLE II: Comparative Assessment of RAG Models on 156 General and 150 Multi-Hop Cybersecurity Questions"
PROSE = """
Using the RAGAS framework, we focused on three metrics: answer relevance, answer similarity,
and answer correctness. The original question uses cosine similarity. N equal to 3.
It combines semantic similarities and factual correctness. F1 score, respectively 0.75 and 0.25.
The metrics for each model are expressed as mean and standard deviation.
"""


class ScannedScoreTableTest(unittest.TestCase):
    def setUp(self):
        self.canvas = ScanImage(3061, 803, b"\xff" * (3061 * 803), 480, 4080, 5280, 530, 380)
        self.scan = ScanSource(Path("renamed-upload.pdf"), 8, "tesseract")
        self.page = Page(8, f"{CAPTION}\n{PROSE}", "ocr_match", scan=self.scan)

    def recognized(self, image, engine, psm=6, digits_only=False):
        if psm != 6 or image.top != self.canvas.top:
            return ()
        if image.height < 430:
            return ()
        if image.width == self.canvas.width and image.height == self.canvas.height:
            return tuple(Word(text, x - 10, y - 7, x + 10, y + 7) for text, x, y in (
                ("General", 1040, 222), ("Cybersecurity", 1300, 222), ("Questions", 1540, 222),
                ("Multi-Hop", 2020, 222), ("Cybersecurity", 2310, 222), ("Questions", 2600, 222),
                ("Model", 405, 306), ("Relevance", 985, 306), ("Similarity", 1310, 306),
                ("Correctness", 1640, 306), ("Relevance", 1970, 306), ("Similarity", 2300, 306),
                ("Correctness", 2630, 306),
            ))
        return ()

    def collect(self, *, ambiguous=False, page=None, prior=()):
        values = ("0.90", "0.05", "0.95", "0.02", "0.71", "0.08",
                  "0.93", "0.05", "0.93", "0.03", "0.70", "0.09")
        words = []
        for y, label in zip((449, 507, 563, 619, 677),
                            ("MoRSE", "GPT-4 0125-Preview", "MIXTRAL 7X8", "HACKERGPT", "GEMINI 1.0 Pro")):
            words.append(Word(label, 305, y - 12, 690, y + 12))
            for x, value in zip((902, 1064, 1228, 1392, 1553, 1720, 1886, 2050, 2214, 2378, 2542, 2708), values):
                words.append(Word(value, x - 20, y - 10, x + 20, y + 10))
        if ambiguous:
            words.append(Word("0.42", 1391, 438, 1418, 459))

        def recognize(image, engine, psm=6, digits_only=False):
            if image.height == self.canvas.height:
                return self.recognized(image, engine)
            if image.height < 100:
                return ()
            dx, dy = image.left - self.canvas.left, image.top - self.canvas.top
            return tuple(Word(w.text, w.x0 - dx, w.y0 - dy, w.x1 - dx, w.y1 - dy) for w in words)

        with patch("scanned_score_tables.scan_region", return_value=self.canvas), \
             patch("scanned_score_tables.deskew", side_effect=lambda image: image), \
             patch("scanned_score_tables.recognize", side_effect=recognize):
            return read_scanned_score_tables([page or self.page], list(prior))

    def test_paired_columns_retain_mean_std_and_bbox(self):
        table, = self.collect()
        payload = Payload(document={}, publication={})
        add_table(payload, table, True)
        mean, std = payload.results[2:4]
        self.assertEqual((mean["metric_name"], mean["aggregation"], mean["raw_value"]),
                         ("answer similarity", "mean", "0.95"))
        self.assertEqual((std["metric_name"], std["aggregation"], std["raw_value"], std["raw_unit"]),
                         ("answer similarity", "std", "0.02", None))
        source, = [e for e in payload.evidence if e["entity_id"] == std["result_id"] and e["field"] == "raw_value"]
        self.assertEqual((source["page"], source["table"], source["table_row"], source["table_column"]),
                         (8, "TABLE II", "MoRSE", "General Cybersecurity Questions / Similarity / std"))
        self.assertEqual(source["bbox"], self.canvas.source_bbox(1372, 439, 1412, 459))
        self.assertEqual(source["verification"], "locator_only")

    def test_an_ambiguous_cell_is_not_borrowed_from_neighbor(self):
        table, = self.collect(ambiguous=True)
        self.assertEqual(table.rows[0][1][2:5], ["0.95", None, "0.71"])

    def test_unrelated_caption_does_not_enable_fixed_grid(self):
        page = Page(8, "TABLE II: Comparative Assessment of Another Model", "ocr_match", scan=self.scan)
        self.assertEqual(self.collect(page=page), [])

    def test_conflicting_crop_does_not_replace_existing_table(self):
        table, = self.collect()
        row, cells = table.rows[0]
        earlier = replace(table, rows=[(row, [*cells[:3], "0.22", *cells[4:]])])
        self.assertEqual(self.collect(prior=(earlier,)), [earlier])

    def test_cve_accuracy_keeps_percent_and_has_no_aggregation(self):
        page = Page(9, "We checked whether the models correctly identified the vulnerability.", "ocr_match")
        columns = metric_columns(page, self.page, CVE_GRID)
        accuracy = columns[-1]
        self.assertEqual((accuracy.dataset, accuracy.header.name, accuracy.header.unit,
                          accuracy.aggregation, accuracy.aggregation_evidence),
                         ("CVE Questions", "Accuracy", "%", None, None))

    def test_bootstrap_cell_keeps_printed_endpoint_order(self):
        words = (Word("1225.54", 811, 343, 955, 365), Word("(1275-1175)", 1000, 343, 1220, 365))
        cropped = (Word("1225.54(1275-1175)", 20, 8, 427, 48),)
        with patch("scanned_score_tables.recognize", return_value=cropped):
            values, bbox = bootstrap_cell(self.canvas, "tesseract", words, 788, 1255, 355)
        self.assertEqual(values, ("1225.54", "1275", "1175"))
        self.assertEqual(bbox, self.canvas.source_bbox(788, 328, 1255, 382))
        with patch("scanned_score_tables.recognize", side_effect=[cropped, ()]):
            self.assertIsNone(bootstrap_cell(self.canvas, "tesseract", (), 788, 1255, 355))

    def test_latency_grid_distinguishes_gpu_cpu_means_and_deviations(self):
        page = Page(13, "TABLE VII: Performance summary of Structured and Unstructured RAG components. "
                        "Mean Time (s) Time Std.", "ocr_match", scan=self.scan)
        words = [Word(label, x - 20, 352, x + 20, 378) for label, x in
                 (("GPU", 858), ("CPU", 1047), ("GPU", 1235), ("CPU", 1410))]
        for label, y in (("Malware", 546), ("Metasploit", 602), ("ExploitDB", 658), ("CWE", 716)):
            words.extend((Word(label, 330, y + 8, 414, y + 29),
                          Word("Retriever", 480, y + 8, 619, y + 29)))
            words.extend(Word(raw, x - 20, y - 9, x + 20, y + 9) for raw, x in
                         zip(("0.061", "0.110", "0.009", "0.013"), (855, 1045, 1235, 1425)))
        with patch("scanned_score_tables.scan_region", return_value=self.canvas), \
             patch("scanned_score_tables.deskew", side_effect=lambda image: image), \
             patch("scanned_score_tables.recognize", return_value=words):
            table = latency_table(page, "TABLE VII", "TABLE VII: Performance summary")
            self.assertIsNone(latency_table(Page(13, "TABLE VII: timings", "ocr_match", scan=self.scan),
                                            "TABLE VII", "TABLE VII: timings"))
        self.assertEqual((table.page.number, table.label, table.rows[0]),
                         (13, "TABLE VII", ("Malware Retriever", ["0.061", "0.110", "0.009", "0.013"])))
        self.assertEqual([(c.header.name, c.aggregation, c.header.unit) for c in table.columns],
                         [("retrieval latency", "mean", "s")] * 2 + [("retrieval latency", "std", "s")] * 2)
        self.assertEqual(table.value_bboxes[0][1], self.canvas.source_bbox(1025, 537, 1065, 555))


if __name__ == "__main__":
    unittest.main()
