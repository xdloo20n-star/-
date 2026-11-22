from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from pdf_pages import Page, ScanSource, Word
from publication import Payload, add_table
from scan_ocr import ScanImage
from scanned_score_tables import (
    CVE_GRID, bootstrap_cell, component_inventory_tables, latency_table, metric_columns, rank_table,
    read_scanned_score_tables,
)


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

    def test_paired_grid_recovers_an_independently_read_decimal(self):
        words = tuple(Word(label, 300, y - 9, 690, y + 9)
                      for label, y in zip(("MoRSE", "GPT-4 0125-Preview", "MIXTRAL 7X8",
                                           "HACKERGPT", "GEMINI 1.0 Pro"), (449, 507, 563, 619, 677)))
        body = words + (Word("043", 1019, 545, 1104, 581),)

        def recognize(image, engine, psm=6, digits_only=False):
            if image.height == self.canvas.height:
                return self.recognized(image, engine)
            if image.height > 100:
                dx, dy = image.left - self.canvas.left, image.top - self.canvas.top
                return tuple(Word(w.text, w.x0 - dx, w.y0 - dy, w.x1 - dx, w.y1 - dy) for w in body)
            x0, y0 = image.left - self.canvas.left, image.top - self.canvas.top
            if image.height > 75 and abs(x0 - 1000) < 25 and abs(y0 - 540) < 10:
                return (Word("0.43", 1019 - x0 + 15, 545 - y0 + 15, 1104 - x0 + 15, 581 - y0 + 15),)
            return ()

        with patch("scanned_score_tables.scan_region", return_value=self.canvas), \
             patch("scanned_score_tables.deskew", side_effect=lambda image: image), \
             patch("scanned_score_tables.recognize", side_effect=recognize):
            table, = read_scanned_score_tables([self.page], [])
        payload = Payload(document={}, publication={})
        add_table(payload, table, True)
        result, = payload.results
        source, = [e for e in payload.evidence if e["entity_id"] == result["result_id"] and e["field"] == "raw_value"]
        self.assertEqual((result["metric_name"], result["raw_value"], result["aggregation"]),
                         ("answer relevance", "0.43", "std"))
        self.assertEqual((source["page"], source["table"], source["table_row"], source["table_column"], source["bbox"]),
                         (8, "TABLE II", "MIXTRAL 7X8", "General Cybersecurity Questions / Relevance / std",
                          self.canvas.source_bbox(1019, 545, 1104, 581)))

    def test_win_grid_uses_the_opponent_and_never_reads_triangular_dashes(self):
        caption = "TABLE IV: Fraction of Wins (No Ties) of Each Combination of Models"
        page = Page(12, caption, "ocr_match", scan=self.scan)
        protocol = Page(10, "Given a query and the corresponding reference response, we ask GPT-4 0125-Preview "
                            "to choose the best response between Model A and Model B.", "ocr_match", scan=self.scan)
        canvas = ScanImage(3346, 703, b"\xff" * (3346 * 703), 480, 4080, 5280, 408, 396)
        opponents = ("GPT-4 0125-Preview", "GEMINI 1.0 Pro", "MIXTRAL 7X8", "HACKERGPT")
        rows = tuple(Word(label, 276, y - 11, 650, y + 11) for label, y in
                     zip(("MoRSE", *opponents[:3]), (407, 466, 522, 580)))

        def collect(bad_heading=False):
            def recognize(image, engine, psm=6, digits_only=False):
                x0, y0 = image.left - canvas.left, image.top - canvas.top
                if image.height == 126:
                    index = (720, 1320, 1930, 2540).index(x0)
                    dataset = "CVE Qs" if index == 0 and bad_heading else "Mul.Hop Qs"
                    return (Word(f"{opponents[index]} General Qs {dataset}", 0, 2, 300, 95),)
                if image.height > 150:
                    return tuple(Word(w.text, w.x0 - x0, w.y0 - y0, w.x1 - x0, w.y1 - y0) for w in rows)
                if abs(x0 - 800) < 30 and (y0 < 410 or 430 < y0 < 463):
                    pad = 15 if image.height > 75 else 0
                    return (Word("70%", 827 - x0 + pad, 393 - y0 + pad, 915 - x0 + pad, 428 - y0 + pad),)
                return ()

            with patch("scanned_score_tables.scan_region", return_value=canvas), \
                 patch("scanned_score_tables.deskew", side_effect=lambda image: image), \
                 patch("scanned_score_tables.recognize", side_effect=recognize):
                return read_scanned_score_tables([self.page, protocol, page], [])

        # Supply the score-table family anchor without invoking an unrelated crop for Table II.
        paired = replace(self.page, text=f"Background {CAPTION}")
        with patch.object(self, "page", paired), patch("scanned_score_tables.scan_grid", return_value=None):
            table, = collect()
            self.assertEqual(collect(bad_heading=True), [])
        payload = Payload(document={}, publication={})
        add_table(payload, table, True)
        result, = payload.results
        source, = [e for e in payload.evidence if e["entity_id"] == result["result_id"] and e["field"] == "raw_value"]
        self.assertEqual((result["metric_name"], result["metric_variant"], result["raw_value"], result["raw_unit"]),
                         ("pairwise win fraction", "reference-guided GPT-4 judging; opponent GPT-4 0125-Preview; ties excluded",
                          "70%", "%"))
        self.assertEqual((source["page"], source["table"], source["table_row"], source["table_column"], source["bbox"]),
                         (12, "TABLE IV", "MoRSE", "GPT-4 0125-Preview / General Qs",
                          canvas.source_bbox(827, 393, 915, 428)))

    def test_unrelated_caption_does_not_enable_fixed_grid(self):
        page = Page(8, "TABLE II: Comparative Assessment of Another Model", "ocr_match", scan=self.scan)
        self.assertEqual(self.collect(page=page), [])

    def test_conflicting_crop_does_not_replace_existing_table(self):
        table, = self.collect()
        row, cells = table.rows[0]
        earlier = replace(table, rows=[(row, [*cells[:3], "0.22", *cells[4:]])])
        self.assertEqual(self.collect(prior=(earlier,)), [earlier])

    def test_cve_accuracy_keeps_percent_and_has_no_aggregation(self):
        page = Page(9, "For CVEs, accuracy is straightforward: we check if the model "
                       "identified the correct vulnerability.", "ocr_match")
        columns = metric_columns(page, self.page, CVE_GRID)
        accuracy = columns[-1]
        self.assertEqual((accuracy.dataset, accuracy.header.name, accuracy.header.unit,
                          accuracy.aggregation, accuracy.aggregation_evidence),
                         ("CVE Questions", "Accuracy", "%", None, None))
        self.assertEqual(accuracy.metric_variant, "correct identification of vulnerability")
        self.assertEqual(accuracy.metric_variant_evidence, (page, "identified the correct vulnerability"))
        other = Page(9, "We discuss whether an analyst identified the correct vulnerability.", "ocr_match")
        self.assertIsNone(metric_columns(other, self.page, CVE_GRID)[-1].metric_variant)

    def test_bootstrap_cell_keeps_printed_endpoint_order(self):
        words = (Word("1225.54", 811, 343, 955, 365), Word("(1275-1175)", 1000, 343, 1220, 365))
        cropped = (Word("1225.54(1275-1175)", 20, 8, 427, 48),)
        with patch("scanned_score_tables.recognize", return_value=cropped):
            values, bbox = bootstrap_cell(self.canvas, "tesseract", words, 788, 1255, 355)
        self.assertEqual(values, ("1225.54", "1275", "1175"))
        self.assertEqual(bbox, self.canvas.source_bbox(788, 328, 1255, 382))
        with patch("scanned_score_tables.recognize", side_effect=[cropped, ()]):
            self.assertIsNone(bootstrap_cell(self.canvas, "tesseract", (), 788, 1255, 355))

    def test_rank_grid_retains_abbreviated_model_and_reads_shifted_combined_interval(self):
        caption = "TABLE V: Elo Ratings, MLE Ratings, and Bootstrap Ratings"
        page = Page(12, caption, "ocr_match", scan=self.scan)
        policy = (self.page, "median, 2.5% and 97.5% percentiles")
        canvas = ScanImage(3077, 703, b"\xff" * (3077 * 703), 480, 4080, 5280, 514, 1077)
        headings = tuple(Word(label, x, 254, x + 20, 279) for x, label in enumerate(
            ("Model", "Elo", "Elo", "Bootstrap", "Elo", "Elo", "Bootstrap", "Elo", "Elo", "Bootstrap")))
        models = (Word("MoRSE", 55, 332, 229, 369),)

        def recognize(image, engine, psm=6, digits_only=False):
            if image == canvas:
                return headings
            if image.left == canvas.left + 25 and image.top == canvas.top + 377:
                return (Word("GPT-4", 30, 8, 157, 40), Word("0125", 180, 10, 279, 44))
            if image.left in (canvas.left + 2580, canvas.left + 2584) and 406 <= image.top - canvas.top <= 415:
                return (Word("1041.05(1050-932)", 43, 10, 430, 47),)
            return ()

        with patch("scanned_score_tables.scan_region", return_value=canvas), \
             patch("scanned_score_tables.deskew", side_effect=lambda image: image), \
             patch("scanned_score_tables.cropped_words", side_effect=(models, ())), \
             patch("scanned_score_tables.recognize", side_effect=recognize):
            table = rank_table(page, "TABLE V", caption, policy)
        self.assertEqual(table.rows[1], ("GPT-4 0125", [None] * 12 + ["1041.05", "1050", "932"]))
        payload = Payload(document={}, publication={})
        add_table(payload, table, True)
        result = payload.results[-1]
        source, = [e for e in payload.evidence if e["entity_id"] == result["result_id"] and e["field"] == "raw_value"]
        self.assertEqual((result["metric_name"], result["raw_value"], result["aggregation"]),
                         ("bootstrap Elo", "932", "q0.025"))
        self.assertEqual((source["page"], source["table"], source["table_row"], source["table_column"], source["bbox"]),
                         (12, "TABLE V", "GPT-4 0125", "Complete Cybersecurity Questions / Bootstrap / q0.025",
                          canvas.source_bbox(2580, 408, 3050, 462)))

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

    def test_component_inventory_retains_attached_units_and_approximation(self):
        headings = [Word("Size", 2265, 269, 2327, 295), Word("No.", 2854, 258, 2900, 281),
                    Word("Doc.", 2952, 256, 3012, 279)]
        labels = [Word("Malware", 330, 553, 414, 579), Word("Retriever", 480, 553, 619, 579)]
        clipped = [Word("3.9MB", 2245, 536, 2351, 563), Word("~", 2861, 533, 2891, 548),
                   Word("1000", 2940, 521, 3014, 549)]
        alternate = [Word("3.9MB", 2243, 535, 2353, 563), Word("~1000", 2859, 522, 3016, 550)]
        page = Page(13, "TABLE VII: Performance summary of Structured and Unstructured RAG components.",
                    "ocr_match", scan=self.scan)
        views = ((self.canvas, headings + labels), (self.canvas, clipped), (self.canvas, alternate))
        table, = component_inventory_tables(page, "TABLE VII", page.text, views)
        payload = Payload(document={}, publication={})
        add_table(payload, table, True)
        self.assertEqual([(r["metric_name"], r["raw_value"], r["raw_unit"], r["reported_value"])
                          for r in payload.results],
                         [("dense retriever size", "3.9MB", "MB", 3.9),
                          ("knowledge-base document count", "∼1000", None, 1000)])
        self.assertEqual(payload.configurations[0]["rag_variant"], "Malware Retriever")
        self.assertIsNone(payload.configurations[0]["generator_model"])
        size, count = [e for e in payload.evidence if e["entity_type"] == "result" and e["field"] == "raw_value"]
        self.assertEqual((count["page"], count["table"], count["table_row"], count["table_column"]),
                         (13, "TABLE VII", "Malware Retriever", "No. Doc."))
        self.assertEqual(size["bbox"], self.canvas.source_bbox(2245, 536, 2351, 563))
        self.assertEqual(count["bbox"], self.canvas.source_bbox(2861, 521, 3014, 549))

    def test_component_inventory_leaves_disagreeing_counts_unread(self):
        headings = [Word("Size", 2265, 269, 2327, 295), Word("No.", 2854, 258, 2900, 281),
                    Word("Doc.", 2952, 256, 3012, 279)]
        labels = [Word("ExploitDB", 330, 666, 414, 691), Word("Retriever", 480, 660, 619, 684)]
        dash = Word("-", 2290, 658, 2308, 667)
        clipped = [dash, Word("~13000", 2859, 638, 3014, 665)]
        alternate = [dash, Word("~18000", 2859, 638, 3014, 665)]
        page = Page(13, "TABLE VII: Performance summary of Structured and Unstructured RAG components.",
                    "ocr_match", scan=self.scan)
        views = ((self.canvas, headings + labels), (self.canvas, clipped), (self.canvas, alternate))
        table, = component_inventory_tables(page, "TABLE VII", page.text, views)
        self.assertEqual(table.rows, [("ExploitDB Retriever", ["-"])])
        payload = Payload(document={}, publication={})
        add_table(payload, table, True)
        row, = payload.results
        self.assertEqual((row["metric_name"], row["raw_value"], row["extraction_status"], row["reported_value"]),
                         ("dense retriever size", "-", "absent", None))
        self.assertEqual(component_inventory_tables(page, "TABLE VII", page.text,
                                                   ((self.canvas, labels), *views[1:])), [])


if __name__ == "__main__":
    unittest.main()
