import unittest

from pdf_pages import PDF, Page
from publication import extract_publication, scale_and_value
from rag_tables import Header, assign_groups


class PublicationTest(unittest.TestCase):
    def test_table_extraction_connects_result_and_source(self):
        page = Page(1, """
Retrieval Augmented Generation for Question Answering

Method                    Accuracy (%)    F1 (%)
Hybrid                    79.2            72.1
Baseline                  0               —

Table 1: Results on Natural Questions test set.
""", "text_match")
        payload = extract_publication(PDF({}, [page]))
        self.assertEqual(len(payload["results"]), 4)
        result = payload["results"][0]
        self.assertEqual(result["raw_value"], "79.2")
        self.assertEqual(result["reported_scale"], "percent")
        self.assertEqual(result["dataset_id"], payload["datasets"][0]["dataset_id"])
        self.assertEqual(payload["datasets"][0]["dataset_name"], "Natural Questions")
        self.assertEqual(payload["results"][2]["reported_value"], 0.0)
        self.assertEqual(payload["results"][3]["extraction_status"], "absent")
        source = next(e for e in payload["evidence"] if e["entity_id"] == result["result_id"])
        self.assertEqual((source["page"], source["table"], source["table_row"]), (1, "Table 1", "Hybrid"))
        self.assertEqual(source["fragment"], "79.2")

    def test_groups_follow_merged_header_centers(self):
        centers = [35, 44, 54.5, 65, 74.5, 86.5, 96.5, 109, 119, 129, 138]
        columns = [Header("score", "score", center, None, None) for center in centers]
        self.assertEqual(assign_groups(columns, [("A", 50.5), ("B", 107)]), ["A"] * 4 + ["B"] * 7)

    def test_unknown_scale_and_ambiguous_ocr_token(self):
        self.assertEqual(scale_and_value("22.27", None), (None, None, 22.27, "extracted"))
        self.assertEqual(scale_and_value("033", None), (None, None, None, "present_not_extractable"))

    def test_out_of_scope_requires_no_dependent_rows(self):
        page = Page(1, "Experimental comparison of thin-film solar cells", "text_match")
        payload = extract_publication(PDF({}, [page]))
        self.assertFalse(payload["publication"]["in_scope"])
        self.assertTrue(payload["publication"]["exclusion_reason"])
        self.assertEqual(payload["results"], [])


if __name__ == "__main__":
    unittest.main()
