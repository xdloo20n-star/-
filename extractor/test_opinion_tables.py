import unittest

from pdf_pages import PDF, Page, Word
from publication import extract_publication


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
