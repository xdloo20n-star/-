import unittest
from dataclasses import replace

from pdf_pages import Page, Word
from prostep_appendix import read_prostep_appendix


def row(y, *groups, glyph_width=4):
    words = []
    for start, text in groups:
        x = start
        for token in text.split():
            width = glyph_width * len(token)
            words.append(Word(token, x, y - 3, x + width, y + 3))
            x += width + 2
    return words


def page(number, *rows, reading=""):
    layout = "\n".join(" ".join(word.text for word in words) for words in rows)
    return Page(number, layout, "text_match", reading, tuple(word for words in rows for word in words), (595, 842))


# Table 13's Agreement cell wraps, while the right column contains prose.
AUDIT = page(
    18,
    row(501, (72, "Table 13: Agreement between QwQ-32B (our")),
    row(513, (72, "annotator) and Claude Opus 4.7 on the 500-sample")),
    row(525, (72, "audit.")),
    row(550, (81, "Metric"), (177, "Value")),
    row(566, (81, "Audited samples"), (179, "500")),
    row(576, (81, "Agreement"), (179, "80.56% (402 / 499 with"), (303, "A separate 402")),
    row(586, (179, "valid labels)"), (303, "model produces 499")),
    row(596, (81, "Cohen’s κ"), (179, "0.6104 (substantial)")),
    row(606, (81, "QwQ-labeled GOOD"), (179, "274")),
    row(615, (81, "Frontier-labeled GOOD"), (179, "255")),
    row(644, (72, "Table 14: Per-step-type agreement on the 500-sample")),
    row(656, (72, "audit.")),
    row(681, (97, "Step type"), (153, "N"), (168, "Agreement"), (227, "Cohen’s κ")),
    row(697, (97, "answer"), (146, "200"), (184, "91.00%"), (246, "0.82"), (303, "unrelated 0.88")),
    row(707, (97, "search"), (146, "249"), (184, "77.11%"), (246, "0.52")),
    row(717, (97, "reason"), (146, "50"), (184, "56.00%"), (246, "0.12")),
)

AUDIT_SCOPE = Page(17, "", "text_match", "Frontier-model audit. We further audit 500 stratified samples drawn "
                   "from QwQ-32B-labeled trajectories (held-out from PRM training) by having an "
                   "independent frontier-class LLM (Claude Opus 4.7) re-evaluate them.")

# Table 17's star is attached to F1 in the first row and to both metrics in the second.
MARKERS = page(
    19,
    row(76, (72, "Table 17: Results at the 3B scale (EM / F1, %). Bold"),
        (310, "Table 19: Token-level F1 on recovered trajectories.")),
    row(88, (72, "denotes the best result; “*” denotes p < 0.05 against")),
    row(100, (72, "Search-R1.")),
    row(126, (73, "Dataset"), (125, "Base 3B"), (180, "Search-R1"), (244, "PRO-Step")),
    row(143, (73, "HotpotQA"), (120, "5.31 / 17.36"), (174, "31.82 / 41.30"), (232, "32.82 / 44.33*"), (312, "HotpotQA 66.6")),
    row(154, (73, "PopQA"), (120, "7.24 / 22.23"), (174, "36.87 / 42.14"), (232, "39.88* / 46.10*")),
    row(205, (73, "Average"), (120, "4.01 / 14.35"), (174, "26.22 / 33.51"), (232, "31.04 / 40.00")),
)

IFBC = page(
    19,
    row(359, (72, "Table 18: Intermediate-Flawed-But-Correct (IFBC) rate")),
    row(371, (72, "among correct trajectories. Statistical tests compare")),
    row(383, (72, "each method with PRO-S TEP . Lower is better.")),
    row(408, (73, "Method"), (165, "IFBC"), (200, "p vs. Ours"), (251, "# Correct")),
    row(412, (190, "↓")),
    row(423, (73, "PRO-S TEP"), (167, "26.8%"), (222, "–"), (265, "474")),
    row(433, (73, "Outcome-only ablation"), (167, "40.3%"), (207, "< 10 −4"), (265, "447")),
    row(443, (73, "Search-R1"), (167, "33.3%"), (211, "0.033"), (265, "439"), (303, "0.66 in neighboring text")),
    row(453, (73, "StepSearch"), (167, "33.0%"), (211, "0.037"), (265, "457")),
    row(463, (73, "ReasonRAG"), (167, "21.6%"), (204, "0.071 ( NS )"), (265, "430")),
)

# Table 16's right-column probability follows its signed difference in each comparison cell.
PAIRED = page(
    18,
    row(402, (310, "Table 16: Per-dataset paired t-test on EM. n = test split")),
    row(415, (310, "size.")),
    row(435, (303, "Dataset"), (363, "n"), (380, "vs. Search-R1"), (429, "vs. ReasonRAG"), (481, "vs. StepSearch"),
        glyph_width=2.7),
    row(443, (72, "unrelated p=0.66"), (303, "HotpotQA"), (349, "7,405"), (373, "+0.85, p=0.10 NS"),
        (428, "+2.36, p < 10 −6 ⋆⋆⋆"), (484, "+0.01, p=0.98 NS"), glyph_width=2.7),
    row(451, (303, "PopQA"), (347, "14,267"), (373, "-0.18, p=0.56 NS"),
        (428, "+2.69, p < 10 −15 ⋆⋆⋆"), (481, "+1.23, p < 10 −3 ⋆⋆⋆"), glyph_width=2.7),
    row(463, (303, "Bamboogle"), (354, "125"), (373, "+3.20, p=0.40 NS"),
        (428, "-1.60, p=0.66 NS"), (484, "+3.20, p=0.43 NS"), glyph_width=2.7),
)
PAIR_SCOPE = Page(17, "", "text_match", "We supplement Table 1 with bootstrap confidence intervals and per-dataset "
                  "paired t-tests, computed on per-question results against PRO-STEP.")

# Table 6 interleaves counts and rates above a right-column caption.
PRM_COUNTS = page(
    15,
    row(402, (303, "Step Type"), (360, "Count (%)"), (430, "GOOD"), (490, "BAD")),
    row(414, (303, "Search"), (350, "78,055"), (384, "(71.2%)"), (418, "55,710"), (441, "(71.4%)"),
        (472, "22,345"), (505, "(28.6%)"), glyph_width=2.7),
    row(422, (303, "Answer"), (350, "31,062"), (384, "(28.3%)"), (418, "14,436"), (441, "(46.5%)"),
        (472, "16,626"), (505, "(53.5%)"), glyph_width=2.7),
    row(430, (303, "Reason"), (360, "547"), (384, "(0.5%)"), (424, "144"), (441, "(26.3%)"),
        (474, "403"), (505, "(73.7%)"), glyph_width=2.7),
    row(443, (303, "Total"), (347, "109,664"), (384, "(100%)"), (416, "70,290"), (441, "(64.1%)"),
        (472, "39,374"), (505, "(35.9%)"), glyph_width=2.7),
    row(465, (303, "Table 6: Statistics of the PRM training corpus:")),
)


def parsed(pages):
    return [(table.label, table.config_name, name, column, raw, bbox)
            for table in read_prostep_appendix(pages, [])
            for (name, values), boxes in zip(table.rows, table.value_bboxes, strict=True)
            for column, raw, bbox in zip(table.columns, values, boxes, strict=True)]


class ProstepAppendixTests(unittest.TestCase):
    def test_audit_cells_keep_wrap_stratum_and_denominator(self):
        results = parsed([AUDIT_SCOPE, AUDIT])
        valid = next(result for result in results if result[3].header.name == "Valid audited labels")
        self.assertEqual((valid[0], valid[4]), ("Table 13", "499"))
        self.assertEqual(valid[3].dataset, "Frontier-label audit")
        self.assertEqual(valid[3].data_split, "held out from PRM training")
        self.assertEqual(valid[3].sample_size, 500)
        self.assertEqual(valid[5], (229 / 595, 573 / 842, 241 / 595, 579 / 842))
        _, _, name, column, raw, bbox = next(
            result for result in results if result[0] == "Table 14" and result[3].metric_variant == "reason"
            and result[3].header.name == "Cohen's kappa")
        self.assertEqual((name, raw, column.sample_size), ("reason", "0.12", 50))
        self.assertGreater(bbox[1], 0.84)

    def test_audit_rejects_wrapped_denominator_with_inconsistent_rate(self):
        malformed = replace(AUDIT, words=tuple(replace(word, text="400") if word.text == "499" else word
                                               for word in AUDIT.words))
        self.assertFalse(any(result[0] == "Table 13" for result in parsed([AUDIT_SCOPE, malformed])))

    def test_significance_bound_only_on_marked_model_and_metric(self):
        results = parsed([MARKERS])
        self.assertEqual([(r[3].dataset, r[3].metric_variant, r[4]) for r in results], [
            ("HotpotQA", "F1 vs Search-R1 3B", "<0.05"),
            ("PopQA", "EM vs Search-R1 3B", "<0.05"),
            ("PopQA", "F1 vs Search-R1 3B", "<0.05"),
        ])
        self.assertEqual(results[0][3].header.name, "Significance p upper bound")
        self.assertEqual(results[0][1], "PRO-STEP")
        self.assertEqual(results[0][5], (260 / 595, 140 / 842, 284 / 595, 146 / 842))

    def test_markers_without_probability_legend_are_not_p_values(self):
        ungrounded = replace(MARKERS, words=tuple(replace(word, text="0.95") if word.text == "<" else word
                                                for word in MARKERS.words))
        self.assertEqual(parsed([ungrounded]), [])

    def test_ifbc_count_and_bound_in_distinct_columns(self):
        results = parsed([IFBC])
        effect = [result for result in results if result[1] == "Outcome-only ablation"]
        self.assertEqual([(r[3].header.name, r[4]) for r in effect], [
            ("Correct trajectories", "447"), ("IFBC difference p", "<10−4"),
        ])
        self.assertEqual(effect[1][3].metric_variant, "vs PRO-STEP; upper exclusive bound")
        self.assertEqual(effect[1][3].dataset, "IFBC paired evaluation subset")
        self.assertLess(effect[1][5][2], effect[0][5][0])
        self.assertNotIn("0.66", [result[4] for result in results])

    def test_ifbc_rejects_a_test_against_unknown_baseline(self):
        ambiguous = replace(IFBC, words=tuple(replace(word, text="Prior") if word.text == "Ours" else word
                                             for word in IFBC.words))
        self.assertEqual(parsed([ambiguous]), [])

    def test_paired_t_tests_keep_comparator_and_test_size(self):
        results = parsed([PAIR_SCOPE, PAIRED])
        exact, bound, _ = [r for r in results if r[2] == "HotpotQA" and r[3].header.name == "Paired t-test p"]
        self.assertEqual((exact[0], exact[1], exact[3].header.name, exact[4]),
                         ("Table 16", "PRO-STEP", "Paired t-test p", "0.10"))
        self.assertEqual((exact[3].metric_variant, exact[3].data_split, exact[3].sample_size),
                         ("vs Search-R1", "test", 7405))
        self.assertEqual((bound[3].metric_variant, bound[4]), ("vs ReasonRAG upper exclusive bound", "<10−6"))
        self.assertGreater(bound[5][0], exact[5][2])

    def test_paired_em_differences_keep_printed_signs_and_separate_p_columns(self):
        results = parsed([PAIR_SCOPE, PAIRED])
        first = [(r[3].metric_variant, r[4], r[3].header.unit, r[5]) for r in results
                 if r[2] == "PopQA" and r[3].header.name == "Delta EM"]
        self.assertEqual([r[:3] for r in first], [("vs Search-R1", "-0.18", None),
                                                  ("vs ReasonRAG", "+2.69", None), ("vs StepSearch", "+1.23", None)])
        self.assertLess(first[0][3][2], first[1][3][0])

    def test_paired_t_test_rejects_a_cell_crossing_comparator_columns(self):
        displaced = replace(PAIRED, words=tuple(replace(word, x0=word.x0 + 30, x1=word.x1 + 30)
                                               if word.text == "+2.36," else word for word in PAIRED.words))
        self.assertEqual(parsed([PAIR_SCOPE, displaced]), [])

    def test_prm_counts_keep_printed_grouping_and_good_bad_conditions(self):
        results = parsed([PRM_COUNTS])
        self.assertEqual(len(results), 12)
        search = [r for r in results if r[3].metric_variant == "Search"]
        self.assertEqual([(r[3].header.name, r[4]) for r in search],
                         [("Step count", "78,055"), ("GOOD count", "55,710"), ("BAD count", "22,345")])
        self.assertTrue(all(r[3].grouped_digits and r[3].header.unit is None for r in search))
        self.assertGreater(search[0][5][0], 0.5)

    def test_prm_count_totals_reject_misattributed_cell(self):
        altered = replace(PRM_COUNTS, words=tuple(replace(w, text="404") if w.text == "403" else w
                                                for w in PRM_COUNTS.words))
        self.assertEqual(parsed([altered]), [])


if __name__ == "__main__":
    unittest.main()
