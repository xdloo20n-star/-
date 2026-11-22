import unittest

from pdf_pages import Page
from training_scan_pairs import paired_tables


CAPTION = ("Table 3: Additional OAS comparison with Qwen2.5-1.5B-Instruct. Exact-match percentages "
           "are reported as mean (sample SD) over two runs (n = 2).")
HEAD = "Top-k = 2 Top-k = 3 Top-k = 5 Method NQ TriviaQA HotpotQA NQ TriviaQA HotpotQA NQ TriviaQA HotpotQA"
PRINTED_ROWS = ["Search-R1 PPO", "Search-R1 GRPO", "DAPO", "KL-Cov", "DAR", "I-PPO",
                "Ours (OVR only)", "Ours (OAS)"]
BOX = (0.32726, 0.13704, 0.38997, 0.14590)


class TrainingScanPairsTest(unittest.TestCase):
    def test_printed_pair_preserves_score_and_sd_with_conditions(self):
        page = Page(8, CAPTION, "ocr_match")
        tables = paired_tables(page, CAPTION, HEAD, PRINTED_ROWS, {(0, 1): ("45.0(.95)", "45.0(.95)", BOX)})
        table, = tables
        self.assertEqual((table.page.number, table.label, table.top_k, table.rows, table.value_bboxes),
                         (8, "Table 3", 2, [("Search-R1 PPO", ["45.0", ".95"])], ((BOX, BOX),)))
        self.assertEqual([(c.header.name, c.dataset, c.generator, c.aggregation, c.reported_scale, c.header.unit)
                          for c in table.columns],
                         [("EM", "TriviaQA", "Qwen2.5-1.5B-Instruct", "mean over 2 runs", "percent", None),
                          ("EM sample SD", "TriviaQA", "Qwen2.5-1.5B-Instruct",
                           "sample standard deviation over 2 runs", "percent", None)])

    def test_disagreeing_ocr_or_shifted_row_does_not_emit_a_pair(self):
        page = Page(8, CAPTION, "ocr_match")
        contradict = {(0, 1): ("45.0(.95)", "45.0(.05)", BOX)}
        self.assertEqual(paired_tables(page, CAPTION, HEAD, PRINTED_ROWS, contradict), [])
        wrong_rows = [*PRINTED_ROWS]
        wrong_rows[0], wrong_rows[1] = wrong_rows[1], wrong_rows[0]
        self.assertEqual(paired_tables(page, CAPTION, HEAD, wrong_rows, {(0, 1): ("45.0(.95)", "45.0(.95)", BOX)}), [])
