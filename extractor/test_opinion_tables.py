from dataclasses import replace
import unittest

from opinion_tables import read_opinion_tables
from pdf_pages import PDF, Page, Word
from publication import extract_publication
from rag_tables import Table


CAPTION = "Table 6: Diversity metrics at different retrieval depths (k) using hybrid retrieval."
HEADINGS = ("k", "KB", "Config", "Match", "%", "Sent-Int", "Biz", "Impact", "Revenue", "Band", "Tenure", "Evidence", "Marker")
CENTERS = (196, 239, 285, 343, 393, 434, 476)
MAIN = """Table 2: Diversity metrics at k=20 (hybrid retrieval,
forums dataset). Statistical significance via Wilcoxon
signed-rank test (n=48): bold = p<0.05, * = p<0.001.

    Metric               Raw KB Enr. KB            ∆
    Entity Match (%)       24.8         35.4     +42.7%*
    Sent-Int σ              9.12       11.56     +26.8%*
    Biz Impact σ           10.55       12.08     +14.5%
    Emot. Markers           7.2         8.9      +23.6%*
    Revenue Band (%)       78.3        68.1      −13.0%*
    Tenure (%)             81.2        75.6       −6.9%
    Evidence (%)           44.3        46.4       +4.7%
"""


def depth_page(with_depth=True, *, repeat=False):
    raw = ("24.8", "9.12", "10.55", "78.3", "81.2", "44.3", "7.2") if repeat else (
        "32.7", "10.10", "11.48", "63.7", "65.7", "38.0", "5.5"
    )
    enriched = ("35.4", "11.56", "12.08", "68.1", "75.6", "46.4", "8.9") if repeat else (
        "48.5", "13.51", "12.73", "51.8", "59.2", "40.1", "7.0"
    )
    rows = [
        (77, list(zip((86, 105, 131, 166, 187, 210, 246, 278, 299, 319, 345, 378), CAPTION.split()))),
        (102, list(zip((196, 239, 272, 297, 356, 398, 426, 476), "Entity Scored Diversity (σ) Categorical Coverage (%) Emot.".split()))),
        (117, list(zip((109, 134, 155, 191, 209, 239, 271, 292, 332, 359, 393, 434, 476), HEADINGS))),
        (133, [(135, "Raw"), *zip(CENTERS, raw)]),
        (143, [(143, "Enriched"), *zip(CENTERS, enriched)]),
    ]
    if with_depth:
        rows.append((138, [(111, "20" if repeat else "10")]))
    words = tuple(Word(text, x - 2, y - 2, x + 2, y + 2) for y, cells in rows for x, text in cells)
    return Page(15, CAPTION, "text_match", words=words, size=(595, 842))


def comparison_page(number, caption, y, headings, centers, cells):
    rows = [(y + 12 * i, [(76 + 19 * n, raw) for n, raw in enumerate(line.split())])
            for i, line in enumerate(caption.splitlines())]
    rows.append((y + 51, [(88 + 34 * n, raw) for n, raw in enumerate(headings.split()[:-3])]
                 + list(zip(centers, headings.split()[-3:], strict=True))))
    for i, (label, raw) in enumerate(cells):
        rows.append((y + 70 + 10 * i, [(83 + 18 * n, word) for n, word in enumerate(label.split())]
                     + list(zip(centers, raw.split(), strict=True))))
    words = tuple(Word(text, x - 3, at - 2, x + 3, at + 2) for at, row in rows for x, text in row)
    return Page(number, caption, "text_match", words=words, size=(595, 842))


QUESTION_CAPTION = "Table 10: Sentiment-Intensity σ by question type\n(k=20)."
QUESTION_ROWS = [
    ("Detailed Overview", "9.14 11.80 +29.1%"),
    ("Experience", "10.64 11.30 +6.2%"),
    ("Business Impact", "8.56 11.89 +38.9%"),
    ("Specific Examples", "9.02 11.14 +23.5%"),
    ("Process", "7.82 10.44 +33.5%"),
    ("Emotional", "9.56 12.82 +34.1%"),
]
ENTITY_CAPTION = "Table 12: Diversity metrics on entity-matched docu-\nments only (k=20, forums dataset)."
ENTITY_ROWS = [
    ("Matched Docs (of 960)", "238 340 +42.9%*"),
    ("Sent-Int σ", "14.29 16.57 +16.0%"),
    ("Biz Impact σ", "4.45 5.47 +22.9%"),
    ("Revenue Band Cov. (%)", "42.9 47.0 +9.6%"),
    ("Tenure Cov. (%)", "40.5 53.3 +31.6%*"),
    ("Evidence Cov. (%)", "44.3 46.4 +4.7%"),
    ("Emot. Markers", "7.2 8.9 +23.6%*"),
]


class OpinionComparisonTest(unittest.TestCase):
    def test_questions_keep_signed_relative_change_and_printed_cutoff(self):
        page = comparison_page(16, QUESTION_CAPTION, 227, "Question Type Raw Enriched ∆", (183, 229, 281),
                               QUESTION_ROWS)
        # Bold percentage glyphs can extend past the column midpoint.
        page = replace(page, words=tuple(replace(w, x0=262, x1=300) if w.text == "+38.9%" else w
                                         for w in page.words))
        payload = extract_publication(PDF({"Title": "Opinion-Enriched RAG"}, [page]))
        self.assertEqual(len(payload["results"]), 18)
        source = next(e for e in payload["evidence"] if e["field"] == "raw_value" and e.get("table") == "Table 10"
                      and e.get("table_row") == "Business Impact" and e.get("table_column") == "∆")
        result = next(r for r in payload["results"] if r["result_id"] == source["entity_id"])
        config = next(c for c in payload["configurations"] if c["config_id"] == result["config_id"])
        self.assertEqual((result["metric_name"], result["raw_value"], result["raw_unit"], result["metric_k"],
                          config["top_k"], source["page"]),
                         ("Sentiment-Intensity sigma relative change", "+38.9%", "%", 20, 20, 16))
        self.assertEqual(source["bbox"], (262 / 595, 315 / 842, 300 / 595, 319 / 842))

    def test_entity_matched_grid_replaces_generic_table_only_when_complete(self):
        headings = "Metric Raw Enr. ∆"
        centers = (190, 222, 261)
        page = comparison_page(17, ENTITY_CAPTION, 162, headings, centers, ENTITY_ROWS)
        generic = Table(page, "Table 12", ENTITY_CAPTION, 0, [], [])
        tables = read_opinion_tables([page], [generic])
        self.assertNotIn(generic, tables)
        document_count = next(t for t in tables if t.rows == [("Matched Docs (of 960)", ["340", "+42.9%"])])
        delta = document_count.columns[1]
        self.assertEqual((delta.header.name, delta.header.unit, delta.header.cutoff, delta.dataset, document_count.top_k),
                         ("entity-matched retrieved document count relative change", "%", 20,
                          "E-commerce seller forums", 20))
        self.assertEqual(sum(len(t.columns) for t in tables), 21)
        damaged = comparison_page(17, ENTITY_CAPTION, 162, headings, centers, ENTITY_ROWS[:-1])
        self.assertEqual(read_opinion_tables([damaged], [generic]), [generic])


class OpinionDepthTest(unittest.TestCase):
    def extract(self, with_depth=True):
        return extract_publication(PDF({"Title": "Opinion-Enriched RAG"}, [depth_page(with_depth)]))

    def test_depth_is_bound_to_both_grid_rows_and_the_evaluated_document_cutoff(self):
        payload = self.extract()
        source = next((e for e in payload["evidence"] if e["field"] == "raw_value" and e.get("table") == "Table 6"
                       and e.get("table_row") == "k=10 / Enriched" and e.get("table_column") == "Match %"), None)
        self.assertIsNotNone(source)
        result = next(r for r in payload["results"] if r["result_id"] == source["entity_id"])
        config = next(c for c in payload["configurations"] if c["config_id"] == result["config_id"])
        self.assertEqual((result["metric_name"], result["raw_value"], result["raw_unit"], result["metric_k"],
                          config["top_k"], source["page"]), ("Entity Match Rate", "48.5", "%", 10, 10, 15))
        self.assertEqual(source["bbox"], (194 / 595, 141 / 842, 198 / 595, 145 / 842))

    def test_missing_center_label_does_not_assign_a_depth(self):
        self.assertFalse(any(e.get("table") == "Table 6" for e in self.extract(False)["evidence"]))

    def test_complete_same_depth_rows_link_but_a_mismatched_reference_does_not(self):
        for change in (False, True):
            with self.subTest(reference_changed=change):
                reference = MAIN.replace("11.56", "11.57") if change else MAIN
                payload = extract_publication(PDF({"Title": "Opinion-Enriched RAG"},
                                                  [Page(6, reference, "text_match"), depth_page(repeat=True)]))
                results = {r["result_id"]: r for r in payload["results"]}
                sources = {e["entity_id"]: e for e in payload["evidence"] if e["field"] == "raw_value"}
                repeats = [r for r in results.values() if r.get("duplicate_of")]
                self.assertEqual(len(repeats), 0 if change else 14)
                for repeat in repeats:
                    target = results[repeat["duplicate_of"]]
                    self.assertEqual((repeat["raw_value"], repeat["config_id"], repeat["dataset_id"], repeat["metric_k"]),
                                     (target["raw_value"], target["config_id"], target["dataset_id"], 20))
                    self.assertEqual((sources[repeat["result_id"]]["page"], sources[target["result_id"]]["page"]),
                                     (15, 6))
