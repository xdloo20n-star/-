import unittest
from dataclasses import replace

from pdf_pages import Page, Word
from watermark_appendix import read_watermark_appendix


def geometry(number, groups, reading=""):
    rows = []
    for y, cells in groups:
        words = []
        for x, text in cells:
            for token in text.split():
                width = 3.5 * len(token)
                words.append(Word(token, x, y - 3, x + width, y + 3))
                x += width + 2
        rows.append(words)
    return Page(number, "\n".join(" ".join(w.text for w in row) for row in rows), "text_match", reading,
                tuple(w for row in rows for w in row), (595, 842))


DATASETS = {"NFCorpus", "TREC-COVID", "DROP", "MS-MARCO"}
SETUP = Page(19, "", "text_match", "For RAG systems, unless otherwise specified, top-1 text was retrieved from the "
             "knowledge base to evaluate the retrievability of watermark texts. Unless otherwise specified, the evaluation is "
             "carried out on watermarked NFCorpus. In the RAG setting, we applied context compression by retrieving the "
             "top-3 texts and using GPT-4.1 to summarize them.")
QUOTAS = geometry(20, [
    (578, [(73, "Table 6: Query Quotas for Reliable Detection of differ-")]),
    (590, [(73, "ent watermarking schemes.")]),
    (615, [(142, "WARD"), (170, "RAG-WM"), (217, "KMW(1)"), (256, "KMW")]),
    (630, [(73, "Query Quota"), (146, "80"), (183, "30"), (225, "20"), (259, "500"), (310, "another 200")]),
])
EXAMPLES = geometry(21, [
    (532, [(89, "Table 7: Factual Incorrect/Unreasonable Texts")]),
    (557, [(77, "PPL"), (111, "Texts")]),
    (573, [(77, "1908.05"), (111, "Trout contributes to current smoking."), (309, "neighbor")]),
    (583, [(77, "1428.29"), (111, "Current smoking is influenced by trout.")]),
    (623, [(111, "High phytate consumption finds application with")]),
    (628, [(77, "2159.43")]),
    (633, [(111, "Hungarian fathers.")]),
], "Examples of RAG-WM Watermark Texts. We select several of the high perplexity texts from the generated texts by RAG-WM.")
ROBUSTNESS = geometry(21, [
    (220, [(325, "Table 8: Unrelated Content Removal Attack")]),
    (246, [(384, "NFcorpus"), (427, "COVID"), (462, "DROP"), (494, "MARCO")]),
    (261, [(311, "WRR"), (391, "1.00"), (430, "1.00"), (462, "1.00"), (497, "1.00")]),
    (271, [(310, "Filtering Rate (%)"), (391, "2.19"), (430, "1.07"), (462, "0.00"), (497, "0.00")]),
    (366, [(327, "Table 9: Synonym Substitution attack.")]),
    (392, [(355, "NFCorpus"), (408, "COVID"), (448, "DROP"), (487, "MARCO")]),
    (407, [(310, "WRR"), (363, "1.00"), (409, "1.00"), (448, "0.99"), (490, "1.00")]),
], "Table 8 illustrates that KMW can resist this attack. Table 9 exhibits the WRR of KMW under synonym substitution attack. "
   "We reduce the watermark capacity to 16 bits and segment it so that each segment is 1 bit long. We then evaluate "
   "the robustness of KMW against contextual summarization and style-control system prompts. The results, reported in Table 11, show the WRR.")
LOW_CAPACITY = geometry(22, [
    (76, [(73, "Table 11: RAG Setting Restrictions on low-capacity"), (310, "neighbor")]),
    (88, [(73, "KMW Setting.")]),
    (113, [(170, "ConSum"), (214, "Sty")]),
    (129, [(137, "WRR"), (175, "0.93"), (213, "0.87")]),
])


def parsed(pages):
    return [(table, row, column, raw, bbox)
            for table in read_watermark_appendix(pages, DATASETS, set())
            for (row, values), bboxes in zip(table.rows, table.value_bboxes, strict=True)
            for column, raw, bbox in zip(table.columns, values, bboxes, strict=True)]


class WatermarkAppendixTests(unittest.TestCase):
    def test_query_quota_keeps_scheme_and_unknown_dataset(self):
        results = parsed([QUOTAS])
        self.assertEqual([(table.config_name, raw) for table, _, _, raw, _ in results],
                         [("WARD", "80"), ("RAG-WM", "30"), ("KMW(1)", "20"), ("KMW", "500")])
        table, _, column, _, bbox = results[-1]
        self.assertEqual((table.label, column.header.name), ("Table 6", "Queries required for stable detection"))
        self.assertIsNone(column.dataset)
        self.assertEqual(bbox, (259 / 595, 627 / 842, 269.5 / 595, 633 / 842))

    def test_query_quota_rejects_one_misaligned_count(self):
        changed = replace(QUOTAS, words=tuple(replace(w, x0=w.x0 + 15, x1=w.x1 + 15) if w.text == "30" else w
                                             for w in QUOTAS.words))
        self.assertEqual(parsed([changed]), [])

    def test_example_text_preserves_both_sides_of_number_baseline(self):
        results = parsed([EXAMPLES])
        table, _, column, value, bbox = results[-1]
        self.assertEqual((table.config_name, column.header.name, value), ("RAG-WM", "Example text perplexity", "2159.43"))
        self.assertEqual(column.metric_variant, "High phytate consumption finds application with Hungarian fathers.")
        self.assertLess(bbox[2], 0.2)

    def test_orphaned_example_text_is_ambiguous(self):
        altered = replace(EXAMPLES, words=tuple(w for w in EXAMPLES.words if w.text != "2159.43"))
        self.assertEqual(parsed([altered]), [])

    def test_wrr_keeps_datasets_and_distinct_source_rows(self):
        results = parsed([SETUP, ROBUSTNESS])
        self.assertTrue(all(t.config_name == "KMW" and t.top_k == 1 for t, *_ in results))
        self.assertEqual([(t.label, column.dataset, raw) for t, _, column, raw, _ in results if column.dataset == "DROP"],
                         [("Table 8", "DROP", "1.00"), ("Table 9", "DROP", "0.99")])
        self.assertTrue(all(column.header.name == "WRR" and bbox[0] > 0.6 for _, _, column, _, bbox in results))

    def test_wrr_rejects_unknown_dataset_header(self):
        altered = replace(ROBUSTNESS, words=tuple(replace(w, text="Other") if w.text == "DROP" else w
                                                for w in ROBUSTNESS.words))
        self.assertEqual(parsed([SETUP, altered]), [])

    def test_low_capacity_keeps_prompt_conditions_separate(self):
        results = [r for r in parsed([SETUP, ROBUSTNESS, LOW_CAPACITY]) if r[0].label == "Table 11"]
        self.assertEqual([(c.header.name, c.source_label, value) for _, _, c, value, _ in results],
                         [("WRR", "ConSum", "0.93"), ("WRR", "Sty", "0.87")])
        self.assertIn("16 bits", results[0][0].caption)
        self.assertEqual([(table.top_k, column.dataset) for table, _, column, _, _ in results],
                         [(3, "NFCorpus"), (1, "NFCorpus")])
        self.assertLess(results[0][4][2], results[1][4][0])

    def test_low_capacity_requires_setup_from_the_paper(self):
        self.assertEqual(parsed([LOW_CAPACITY]), [])


if __name__ == "__main__":
    unittest.main()
