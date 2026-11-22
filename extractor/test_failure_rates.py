from dataclasses import replace
import unittest

from pdf_pages import PDF, Page, Word
from publication import extract_publication


# Coordinates and words from the upper panel of TrustRAG, PDF page 7, Table 1.
# https://aclanthology.org/2026.emnlp-main.1533.pdf#page=7
CENTERS = (227, 252, 280, 306, 331, 359, 385, 408, 434, 459, 483, 508)
HEADERS = tuple(zip(CENTERS, ("Leak", "Stale", "Unsup.") * 4, strict=True))
FIXTURE = (
    (78, ((410, "GLM-4-32B-"),)),
    (83, ((255, "Qwen-Max"), (334, "DeepSeek-V3"), (485, "Avg."))),
    (87, ((88, "Setting"), (149, "Method"), (410, "0414"))),
    (101, HEADERS),
    (116, ((275, "Main"), (294, "failure"), (320, "reduction"))),
    (126, ((145, "Naive"), (165, "RAG"),
           *zip(CENTERS, "37.50 0.00 5.75 37.50 0.00 6.25 35.75 0.00 5.25 36.92 0.00 5.75".split()))),
    (131, ((78, "T"), (93, "EMPLATE"))),
    (136, ((152, "TrustRAG"), *zip(CENTERS, ("0.00",) * 12))),
    (147, ((145, "Naive"), (165, "RAG"),
           *zip(CENTERS, "0.00 38.50 2.50 0.00 65.75 2.75 0.00 52.50 0.00 0.00 52.25 1.75".split()))),
    (152, ((78, "F"), (89, "RESH"), (102, "-S"), (117, "TRESS"))),
    (157, ((152, "TrustRAG"), *zip(CENTERS, ("0.00",) * 12))),
    (183, ((81, "Table"), (99, "1:"), (120, "All"), (140, "values"), (160, "are"),
           (180, "failure"), (200, "rates"), (225, "(%),"), (247, "where"), (270, "lower"),
           (285, "is"), (300, "better."))),
)


def fixture_page(rows=FIXTURE, page=7) -> Page:
    words = tuple(Word(value, x - len(value) * 1.8, y - 4, x + len(value) * 1.8, y + 4)
                  for y, row in rows for x, value in row)
    text = "\n".join(" ".join(value for _, value in row) for _, row in rows)
    return Page(page, text, "text_match", words=words)


def payload_for(page: Page, definition="Leak, Stale, and Unsup. denote unauthorized leakage, stale citation, "
                                       "and unsupported-answer rates.") -> dict:
    front = Page(1, definition, "text_match")
    pages = [front, *(Page(i, "", "unverified") for i in range(2, page.number)), page]
    return extract_publication(PDF({"Title": "Governed Retrieval-Augmented Generation"}, pages))


class FailureRateTest(unittest.TestCase):
    def test_assigns_centered_settings_and_wrapped_model_in_main_grid(self):
        payload = payload_for(fixture_page())
        self.assertEqual(len(payload["results"]), 48)
        result = next(r for r in payload["results"] if r["raw_value"] == "52.50")
        self.assertEqual((result["metric_name"], result["reported_value"], result["raw_unit"],
                          result["reported_scale"], result["metric_direction"]),
                         ("Stale citation rate", 52.50, "%", "percent", "lower_is_better"))
        configuration = next(c for c in payload["configurations"] if c["config_id"] == result["config_id"])
        self.assertEqual((configuration["rag_variant"], configuration["generator_model"]),
                         ("Naive RAG", "GLM-4-32B-0414"))
        dataset = next(d for d in payload["datasets"] if d["dataset_id"] == result["dataset_id"])
        self.assertEqual(dataset["dataset_name"], "FRESH-STRESS")
        evidence = next(e for e in payload["evidence"] if e["field"] == "raw_value"
                        and e["entity_id"] == result["result_id"])
        self.assertEqual((evidence["page"], evidence["table"], evidence["table_row"], evidence["table_column"]),
                         (7, "Table 1", "FRESH-STRESS / Naive RAG", "GLM-4-32B-0414 / Stale"))
        self.assertEqual(evidence["fragment"], "52.50")
        model_evidence = next(e for e in payload["evidence"] if e["field"] == "generator_model"
                              and e["entity_id"] == configuration["config_id"])
        self.assertEqual(model_evidence["table_column"], "GLM-4-32B-0414")

    def test_ambiguous_numeric_geometry_and_ocr_are_rejected(self):
        page = fixture_page()
        ambiguous = replace(page, words=tuple(replace(w, x0=w.x0 + 25, x1=w.x1 + 25)
                                              if w.text == "52.50" else w for w in page.words))
        for candidate in (ambiguous, replace(page, verification="ocr_match")):
            with self.subTest(candidate=candidate.verification):
                self.assertEqual(payload_for(candidate)["results"], [])

    def test_short_headers_without_governance_definition_are_rejected(self):
        definition = "Leak, Stale, and Unsup. denote memory leaks, stale reads, and unsupported statements."
        self.assertEqual(payload_for(fixture_page(), definition)["results"], [])

    def test_right_column_slice_grid_uses_wrapped_footer_unit(self):
        # Table B2, PDF page 17. The other column contains separate prose.
        rows = (
            (440, ((328, "Method"), (384, "Slice"), (427, "Leak"), (460, "Stale"), (496, "Unsup."))),
            (443.5, ((441, "↓"), (473, "↓"), (514, "↓"))),
            (503, ((388, "Retrieval"), (411, "on"), (421, "T"), (436, "EMPLATE"))),
            (524, ((220, "99.9"), (324, "BM25"), (348, "RAG"), (393, "Restricted"),
                   (431, "53.00"), (463, "0.00"), (500, "53.00"))),
            (682, ((316, "Table"), (337, "B2:"), (376, "Supplementary"), (430, "results."))),
            (694, ((320, "One"), (360, "end-to-end"), (414, "retrieval-pipeline"), (465, "run."))),
            (706, ((320, "All"), (345, "values"), (375, "are"), (406, "failure"), (445, "rates"))),
            (718, ((314, "(%)."),)),
        )
        payload = payload_for(fixture_page(rows, page=17))
        self.assertEqual(len(payload["results"]), 3)
        result = payload["results"][0]
        self.assertEqual((result["raw_value"], result["raw_unit"], result["aggregation"]),
                         ("53.00", "%", "single retrieval-pipeline run"))
        dataset = payload["datasets"][0]
        self.assertEqual((dataset["dataset_name"], dataset["data_split"]), ("TEMPLATE", "Restricted"))
        source = next(e for e in payload["evidence"] if e["field"] == "raw_value"
                      and e["entity_id"] == result["result_id"])
        self.assertEqual((source["page"], source["table"], source["table_row"], source["table_column"]),
                         (17, "Table B2", "TEMPLATE / Restricted / BM25 RAG", "Leak"))
