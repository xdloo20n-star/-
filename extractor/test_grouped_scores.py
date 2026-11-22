import unittest

from pdf_pages import PDF, Page
from publication import extract_publication
from rag_tables import read_tables


# PRO-Step, PDF page 6, Tables 2 and 3:
# https://aclanthology.org/2026.emnlp-main.1667.pdf#page=6
ADJACENT_GRIDS = """Table 2: PRM component ablation.

                    PopQA          HotpotQA              2WikiMulti          Bamboogle       MuSiQue        Average
Variant
                   EM       F1      EM         F1        EM       F1         EM      F1      EM      F1     EM     F1
PRO-S TEP         40.5     47.4    38.7    51.6          44.1     51.4       36.8    47.6    12.5    22.4   34.5   44.1
Without PRM       38.5     45.0    36.8    49.3          41.4     48.4       34.4    44.2    12.3    22.4   32.7   41.9

Table 3: Optimization strategies.

                          PopQA         HotpotQA           2WikiMulti         Bamboogle       MuSiQue        Average
Method
                         EM       F1     EM         F1      EM        F1       EM      F1     EM      F1    EM     F1
PRO-S TEP (SFT)          38.8    45.7   36.7    48.3       32.9       41.3    31.2    40.4    11.3   20.0   30.2   39.1
PRO-S TEP (KTO)          41.8    46.3   32.3    43.5       36.1       42.0    20.8    32.7     8.9   18.5   28.0   36.6
"""

BACKBONE_GRID = """Table 4: Results across different policy backbones.

                         PopQA           HotpotQA        2WikiMulti          Bamboogle        MuSiQue         Average
Method
                       EM      F1       EM        F1     EM           F1     EM       F1     EM       F1     EM       F1
Llama-3.1-8B-Instruct
Base                   5.49   16.14      5.71    14.63   5.50        12.97   7.20    15.60   1.49    6.08     5.08   13.08
PRO-S TEP             22.94   28.12     18.88    27.12   25.40       31.74   8.80    16.99   4.76    8.86    16.16   22.57
Qwen2.5-7B Base/Instruct
PRO-S TEP-Base      40.42     46.04     34.69    47.47   41.09       48.57   36.00   45.38   12.08   22.61   32.85   42.01
PRO-S TEP-Instruct 40.47      47.37     38.73    51.63   44.07       51.43   36.80   47.63   12.49   22.41   34.51   44.09

Table 5: Final EM on recovered trajectories.
"""


def extract(layout, page, methods=None, findings=None):
    pages = [Page(i, "", "unverified") for i in range(1, max(page, 17 if findings else page) + 1)]
    pages[page - 1] = Page(page, layout, "text_match")
    if methods:
        pages[4] = Page(5, methods, "text_match")
    if findings:
        pages[16] = Page(17, findings, "text_match")
    return extract_publication(PDF({"Title": "Process rewards for Retrieval-Augmented Generation"}, pages))


class GroupedScoreTest(unittest.TestCase):
    def test_adjacent_grids_keep_their_caption_and_literal_groups(self):
        payload = extract(ADJACENT_GRIDS, page=6)
        self.assertEqual(len(payload["results"]), 48)
        source = next(e for e in payload["evidence"] if e["field"] == "raw_value" and e["fragment"] == "36.8"
                      and e.get("table_column") == "Bamboogle / EM")
        self.assertEqual((source["page"], source["table"], source["table_row"], source["verification"]),
                         (6, "Table 2", "PRO-S TEP", "text_match"))
        result = next(r for r in payload["results"] if r["result_id"] == source["entity_id"])
        self.assertEqual((result["metric_name"], result["raw_value"], result["reported_value"],
                          result["reported_scale"]), ("EM", "36.8", 36.8, None))
        dataset = next(d for d in payload["datasets"] if d["dataset_id"] == result["dataset_id"])
        self.assertEqual(dataset["dataset_name"], "Bamboogle")
        self.assertIsNone(result["aggregation"])
        self.assertEqual({e["table"] for e in payload["evidence"] if e["field"] == "raw_value"},
                         {"Table 2", "Table 3"})

    def test_backbone_section_is_in_the_row_locator(self):
        payload = extract(BACKBONE_GRID, page=9)
        self.assertEqual(len(payload["results"]), 48)
        source = next(e for e in payload["evidence"] if e["field"] == "raw_value" and e["fragment"] == "22.94")
        self.assertEqual((source["page"], source["table"], source["table_row"], source["table_column"]),
                         (9, "Table 4", "Llama-3.1-8B-Instruct / PRO-S TEP", "PopQA / EM"))
        result = next(r for r in payload["results"] if r["result_id"] == source["entity_id"])
        config = next(c for c in payload["configurations"] if c["config_id"] == result["config_id"])
        self.assertEqual((config["rag_variant"], config["generator_model"]),
                         ("PRO-S TEP", "Llama-3.1-8B-Instruct"))
        average = next(r for r in payload["results"] if r["raw_value"] == "16.16")
        dataset = next(d for d in payload["datasets"] if d["dataset_id"] == average["dataset_id"])
        self.assertEqual((average["aggregation"], dataset["dataset_name"]), ("reported average", None))

    def test_misaligned_native_values_do_not_use_the_legacy_fallback(self):
        # There is still the expected number of numeric tokens, but two are in the same column.
        layout = ADJACENT_GRIDS.split("\n\nTable 3")[0].replace("36.8    47.6", "     36.8 47.6").replace(
            "34.4    44.2", "     34.4 44.2")
        self.assertEqual(read_tables(Page(6, layout, "text_match")), [])

    def test_ambiguous_group_names_do_not_use_the_legacy_fallback(self):
        layout = ADJACENT_GRIDS.split("\n\nTable 3")[0]
        for replacement in ("HotpotQA ", " " * len("Bamboogle")):
            with self.subTest(replacement=replacement):
                ambiguous = layout.replace("Bamboogle", replacement)
                self.assertEqual(read_tables(Page(6, ambiguous, "text_match")), [])

    def test_aggregation_uses_the_evaluation_protocol_and_compatible_macro_scores(self):
        methods = "We report Exact Match (EM) and F1 score using the FlashRAG evaluation framework."
        findings = "The method improves macro-AVG EM and F1."
        payload = extract(ADJACENT_GRIDS, page=6, methods=methods, findings=findings)
        score = next(r for r in payload["results"] if r["raw_value"] == "36.8")
        average = next(r for r in payload["results"] if r["raw_value"] == "34.5")
        self.assertEqual((score["aggregation"], average["aggregation"]), ("mean per question", "macro-average"))
        citations = {e["entity_id"]: e for e in payload["evidence"] if e["field"] == "aggregation"}
        self.assertEqual((citations[score["result_id"]]["page"], citations[average["result_id"]]["page"]), (5, 17))
        self.assertTrue(all(e["verification"] == "text_match" for e in citations.values()))

        incompatible = extract(ADJACENT_GRIDS.replace("34.5", "99.5"), page=6,
                               methods=methods, findings=findings)
        mislabeled = next(r for r in incompatible["results"] if r["raw_value"] == "99.5")
        self.assertEqual(mislabeled["aggregation"], "reported average")

        pooled = extract(ADJACENT_GRIDS.replace("PRM component ablation", "PRM scores using pooled questions"),
                         page=6, methods=methods, findings=findings)
        local = next(r for r in pooled["results"] if r["raw_value"] == "34.5")
        self.assertEqual(local["aggregation"], "reported average")
