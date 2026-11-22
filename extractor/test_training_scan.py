import unittest
from unittest.mock import patch

from pdf_pages import Page, Word
from scan_ocr import ScanImage
from training_scan import bounded_checkpoint_cells, overhead_table, time_aligned_tables


def words(*rows):
    return tuple(Word(text, x-8, y-5, x+8, y+5) for y, row in rows for x, text in row)


class ScannedTrainingTablesTest(unittest.TestCase):
    def test_computational_overhead_keeps_cells_conditions_and_printed_units(self):
        page = Page(19, "Table 8: Computational overhead on NQ with Qwen2.5-", "ocr_match")
        layout = words(
            (75, [(54, "Method"), (286, "Sreptime(s)"), (420, "Total"), (481, "ime"), (522, "(h)"),
                  (626, "Sampled"), (706, "peak"), (762, "GPU"), (822, "(GiB)"),
                  (944, "Non-ckpt."), (1046, "storage"), (1123, "(MiB)")]),
            (125, [(65, "Search-R1"), (152, "PPO"), (322, "91.89"), (515, "642"),
                   (821, "93.97"), (1124, "0862")]),
            (156, [(36, "OAS"), (317, "116.07"), (516, "8.10"), (822, "94.16"), (1124, "0857")]),
            (263, [(63, "Table"), (152, "8:"), (340, "Computational"), (599, "overhead"),
                   (730, "on"), (806, "NQ"), (902, "with"), (1062, "Qwen2.5-")]),
            (323, [(150, "1.5B-Instruct"), (342, "and"), (459, "Top-3"), (626, "retrieval")]),
        )
        def read_cell(word):
            corrections = {"642": "6.42", "93.97": "94.97", "0862": "0.862", "0857": "0.857"}
            return corrections.get(word.text, word.text), (0.61, 0.32, 0.66, 0.34)

        tables = overhead_table(page, layout, read_cell)
        first = tables[0]
        self.assertEqual((first.page.number, first.label, first.top_k, first.rows[0]),
                         (19, "Table 8", 3, ("Search-R1 PPO", ["91.89", "6.42", "94.97", "0.862"])))
        peak = first.columns[2]
        self.assertEqual((peak.header.name, peak.header.unit, peak.generator, peak.dataset, peak.source_label),
                         ("Sampled peak GPU memory", "GiB", "Qwen2.5-1.5B-Instruct", "NQ", "Sampled peak GPU (GiB)"))
        self.assertEqual(first.value_bboxes[0][2], (0.61, 0.32, 0.66, 0.34))
        bad_units = tuple(Word("(s)", w.x0, w.y0, w.x1, w.y1) if w.text == "(GiB)" else w for w in layout)
        self.assertEqual(overhead_table(page, bad_units, read_cell), [])

    def test_aligned_retrieval_behavior_keeps_checkpoint_and_verifies_second_view(self):
        caption = ("Table 10: Time-aligned task, training-diagnostic, and retrieval-behavior statistics on the fixed "
                   "Qwen2.5-1.5B NQ Top-3 evaluation set. The miss column reports an answer-string retrieval-miss proxy.")
        context = Page(9, "The figure presents one controlled trajectory per method. On a fixed Qwen2.5-1.5B NQ Top-3 "
                          "evaluation set, exact outcomes appear in Table 10.", "ocr_match")
        page = Page(20, caption, "ocr_match")
        layout = words(
            (25, [(10, "Run"), (40, "EM/nVDD"), (70, "ATE/OAGC")]),
            (45, [(10, "PPO,"), (25, "step"), (40, "150"), (65, "0.285/0.040"), (105, "2.540/0.073")]),
            (100, [(10, "Run"), (40, "Searches/saturation"), (70, "Miss"), (110, "No"), (130, "search")]),
            (120, [(10, "OAS,"), (25, "step"), (40, "250"), (65, "1.154/0.127"), (105, "0.509"), (135, "0.017")]),
            (170, [(120, caption)]),
        )
        altered = tuple(Word("0.027", w.x0, w.y0, w.x1, w.y1) if w.text == "0.017" else w for w in layout)
        source_box = (0.63, 0.22, 0.66, 0.23)
        tables = time_aligned_tables(page, [context], layout, altered, lambda *bounds: source_box)
        self.assertEqual(tables[-1].rows, [("OAS, step 250", ["1.154", "0.127", "0.509"])])
        self.assertEqual(tables[-1].value_bboxes, ((source_box,) * 3,))
        column = tables[-1].columns[-1]
        self.assertEqual((column.header.name, column.metric_variant, column.dataset, column.aggregation),
                         ("Answer-string retrieval-miss proxy", "step 250", "NQ", "single controlled trajectory"))
        self.assertEqual(time_aligned_tables(page, [], layout, altered, lambda *bounds: source_box), [])

    def test_aligned_supplement_recovers_a_cell_while_preserving_confirmed_neighbors(self):
        caption = ("Table 10: Time-aligned task, training-diagnostic, and retrieval-behavior statistics on the fixed "
                   "Qwen2.5-1.5B NQ Top-3 evaluation set. The miss column reports an answer-string retrieval-miss proxy.")
        protocol = Page(9, "The figure presents one controlled trajectory per method. On a fixed Qwen2.5-1.5B NQ "
                           "Top-3 evaluation set, exact outcomes appear in Table 10.", "ocr_match")
        page = Page(20, caption, "ocr_match")
        layout = words(
            (82, [(288, "Run"), (624, "EM/nVDD"), (882, "ATE/OAGC")]),
            (154, [(288, "PPO,"), (366, "step"), (436, "150"), (620, "0.285/0.040"), (825, "2.540/0.073")]),
            (370, [(184, "Run"), (490, "Searches"), (568, "/"), (653, "saturation"), (827, "Miss"),
                   (952, "No"), (1029, "search")]),
            (484, [(185, "PPO,"), (262, "step"), (328, "250"), (517, "1.673/0.357")]),
            (684, [(120, caption)]),
        )
        other = tuple(Word("3.540/0.073", w.x0, w.y0, w.x1, w.y1) if w.text == "2.540/0.073" else w for w in layout)
        box = (0.748, 0.142, 0.775, 0.151)
        search_box = (0.653, 0.206, 0.680, 0.215)
        supplement = {
            (0, "PPO", "150"): {0: ("0.295", box), 2: ("2.540", box)},
            (1, "PPO", "250"): {0: ("1.673", search_box)},
        }
        upper, lower = time_aligned_tables(page, [protocol], layout, other, lambda *bounds: box,
                                           read_missing=lambda confirmed: supplement)
        self.assertEqual(upper.rows, [("PPO, step 150", ["0.285", "0.040", "2.540", "0.073"])])
        self.assertEqual(lower.rows, [("PPO, step 250", ["1.673"])])
        column, = lower.columns
        self.assertEqual((column.header.name, column.metric_variant, column.dataset, column.generator,
                          column.aggregation, lower.top_k, lower.value_bboxes[0][0]),
                         ("Executed searches", "step 250", "NQ", "Qwen2.5-1.5B",
                          "single controlled trajectory", 3, search_box))

    def test_bounded_checkpoint_ocr_requires_the_two_headers_and_a_matching_row(self):
        image = ScanImage(1195, 910, b"\xff" * (1195 * 910), 405, 3351, 4737, 1725, 525)
        layout = words(
            (82, [(288, "Run"), (624, "EM/nVDD"), (882, "ATE/OAGC")]),
            (154, [(288, "PPO,"), (366, "step"), (436, "150")]),
            (370, [(184, "Run"), (490, "Searches"), (568, "/"), (653, "saturation"), (827, "Miss"),
                   (952, "No"), (1029, "search")]),
        )
        checkpoint = (0, "PPO", "150")
        position = (825 - 53, 158 - 18)
        box = image.source_bbox(position[0] + 8, position[1] + 4, position[0] + 86, position[1] + 29)
        existing = {checkpoint: {i: ("0.040", box) for i in (0, 1, 3)}}
        recognized = (Word("2.540", 8, 4, 86, 29),)
        with patch("training_scan.recognize", return_value=recognized):
            result = bounded_checkpoint_cells(image, image, layout, layout, existing, "tesseract")
        self.assertEqual(result, {checkpoint: {2: ("2.540", box)}})

        shifted_row = tuple(Word("250", *[w.x0, w.y0, w.x1, w.y1]) if w.text == "150" else w for w in layout)
        missing_header = tuple(w for w in layout if w.text != "Miss")
        for damaged in (shifted_row, missing_header):
            with patch("training_scan.recognize", return_value=recognized):
                self.assertEqual(bounded_checkpoint_cells(image, image, damaged, damaged, existing, "tesseract"), {})

        inconsistent = (Word("2.549", 8, 4, 86, 29),)
        with patch("training_scan.recognize", side_effect=[recognized, recognized, inconsistent]):
            self.assertEqual(bounded_checkpoint_cells(image, image, layout, layout, existing, "tesseract"), {})
