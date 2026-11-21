import unittest

from pdf_pages import PDF, Page
from publication import extract_publication


# Adaptive-RAG, PDF page 7, Table 2. Both panels share the caption:
# https://aclanthology.org/2024.naacl-long.389.pdf#page=7
DATASET_PANELS = """ Table 2: Results on each of a collection of datasets with FLAN-T5-XL (3B) as the LLM.
                                                                      SQuAD                              Natural Questions
    Data          Types      Methods                  EM       F1       Acc      Step   Time    EM       F1     Acc         Step    Time
                             No Retrieval             3.60    10.50     5.00     0.00   0.11    14.20   19.00   15.60       0.00    0.13
                  Simple
                             Single-step Approach     27.80   39.30     34.00    1.00   1.00    37.80   47.30   44.60       1.00    1.00

                                                                      MuSiQue                                 HotpotQA
    Data          Types      Methods                  EM       F1       Acc      Step   Time    EM       F1     Acc         Step    Time
                             No Retrieval             2.40    10.70     3.20     0.00   0.11    16.60   22.71   17.20       0.00    0.11
                  Simple
                             Single-step Approach     13.80   22.80     15.20    1.00   1.00    34.40   46.15   36.40       1.00    1.00

Figure 3: Performance on QA and query-complexity assessment.
                             Unrelated figure        1.00    2.00      3.00     4.00   5.00    6.00    7.00    8.00        9.00    10.00
"""


# Adaptive-RAG, PDF page 6, Table 1.
MODEL_PANEL = """Table 1: Averaged results on benchmark datasets with different LLMs.
Self-RAG∗ is trained with a different base LLM, namely LLaMA2.
                                              FLAN-T5-XL (3B)                       FLAN-T5-XXL (11B)
  Types      Methods                  EM       F1     Acc     Step   Time   EM       F1     Acc     Step   Time
             No Retrieval             14.87   21.12   15.97   0.00   0.11   17.83   25.14   19.33   0.00   0.08
  Adaptive   Self-RAG∗                 9.90   20.79   31.57   0.72   0.43   10.87   22.98   34.13   0.74   0.23
             Adaptive-RAG (Ours)      37.17   46.94   42.10   2.17   3.60   38.90   48.62   43.77   1.35   2.00
"""


EFFECTIVENESS = "For effectiveness, we use F1, EM, and Accuracy (Acc), following the standard evaluation protocol."
EFFICIENCY = ("For efficiency, we measure the number of retrievaland-generate steps and the average time "
              "for answering each query relative to the one-step approach.")
CODE_LINK = "Code is available at: https://\ngithub.com/starsuzi/Adaptive-RAG."
DATASET_METHOD = ("The single-hop QA datasets are namely 1) SQuAD v1.1 (Rajpurkar et al., 2016), "
                  "and 2) Natural Questions (Kwiatkowski et al., 2019).\n"
                  "Multi-hop QA We use two benchmark multi-hop QA datasets, which require sequential reasoning "
                  "over multiple documents, namely 1) MuSiQue (Trivedi et al., 2022a), and 2) HotpotQA (Yang et al., 2018).")


def extract(layout, page=7, verification="text_match", context=()):
    pages = {i: Page(i, "", "unverified") for i in range(1, page)}
    pages[page] = Page(page, layout, verification)
    pages.update({item.number: item for item in context})
    return extract_publication(PDF({"Title": "Adaptive Retrieval-Augmented Generation"}, list(pages.values())))


class QAComparisonTest(unittest.TestCase):
    def test_dataset_panels_preserve_methods_values_and_source_coordinates(self):
        payload = extract(DATASET_PANELS)
        self.assertEqual(len(payload["results"]), 40)
        evidence = next(e for e in payload["evidence"] if e["entity_type"] == "result"
                        and e["field"] == "raw_value" and e["fragment"] == "46.15")
        self.assertEqual(
            (evidence["page"], evidence["table"], evidence["table_row"], evidence["table_column"],
             evidence["verification"]),
            (7, "Table 2", "Single-step Approach", "FLAN-T5-XL (3B) / HotpotQA / F1", "text_match"),
        )
        result = next(r for r in payload["results"] if r["result_id"] == evidence["entity_id"])
        self.assertEqual((result["metric_name"], result["raw_value"], result["reported_value"],
                          result["reported_scale"], result["raw_unit"]), ("F1", "46.15", 46.15, None, None))
        config = next(c for c in payload["configurations"] if c["config_id"] == result["config_id"])
        dataset = next(d for d in payload["datasets"] if d["dataset_id"] == result["dataset_id"])
        self.assertEqual((config["rag_variant"], config["generator_model"], dataset["dataset_name"]),
                         ("Single-step Approach", "FLAN-T5-XL (3B)", "HotpotQA"))
        self.assertTrue(any(r["metric_name"] == "Step" and r["reported_value"] == 0
                            and r["raw_value"] == "0.00" for r in payload["results"]))

    def test_model_panel_uses_the_column_models_and_leaves_named_exception_out(self):
        payload = extract(MODEL_PANEL, page=6)
        self.assertEqual(len(payload["results"]), 20)
        evidence = next(e for e in payload["evidence"] if e["field"] == "raw_value" and e["fragment"] == "48.62")
        self.assertEqual((evidence["page"], evidence["table"], evidence["table_row"], evidence["table_column"]),
                         (6, "Table 1", "Adaptive-RAG (Ours)", "FLAN-T5-XXL (11B) / F1 / reported average"))
        self.assertTrue(all(d["dataset_name"] is None for d in payload["datasets"]))
        self.assertTrue(all(c["rag_variant"] != "Self-RAG∗" for c in payload["configurations"]))

    def test_ambiguous_ocr_layout_does_not_enter_the_native_parser(self):
        self.assertEqual(extract(DATASET_PANELS.replace("46.15", "46.I5"), verification="ocr_match")["results"], [])

    def test_native_panel_with_an_ambiguous_cell_is_rejected(self):
        first_panel = DATASET_PANELS.split("\n\n", 1)[0]
        self.assertEqual(extract(first_panel.replace("10.50", "10.5O"))["results"], [])

    def test_uses_documented_names_from_native_methodology(self):
        methodology = Page(6, EFFICIENCY, "text_match")
        payload = extract(DATASET_PANELS, context=(methodology,))
        result = next(r for r in payload["results"] if r["metric_name"] == "Relative time per query"
                      and r["raw_value"] == "0.13")
        self.assertIsNone(result["raw_unit"])
        self.assertIsNone(result["reported_scale"])
        evidence = next(e for e in payload["evidence"] if e["entity_id"] == result["result_id"]
                        and e["field"] == "metric_name")
        self.assertEqual((evidence["page"], evidence["verification"], evidence["fragment"]),
                         (6, "text_match", "average time for answering each query relative to the one-step approach"))
        self.assertNotIn("table", evidence)
        self.assertTrue(any(r["metric_name"] == "Retrieval-and-generation steps per query" for r in payload["results"]))

    def test_query_mean_requires_linked_evaluator_and_metric_description(self):
        link_page = Page(1, CODE_LINK, "text_match")
        methodology = Page(6, EFFECTIVENESS, "text_match")
        for context, expected in (((link_page, methodology), "mean across queries"),
                                  ((link_page,), None), ((methodology,), None)):
            with self.subTest(expected=expected, pages=[p.number for p in context]):
                payload = extract(DATASET_PANELS, context=context)
                scores = [r for r in payload["results"] if r["metric_name"] in {"EM", "F1", "Accuracy"}]
                self.assertEqual({r["aggregation"] for r in scores}, {expected})
                self.assertTrue(all(r["raw_unit"] is None and r["reported_scale"] is None for r in scores))

    def test_all_named_datasets_are_required_for_the_aggregate_count(self):
        context = ("Table 1 shows our main results averaged over all considered datasets. In addition, we report "
                   "the more granular results with FLAN-T5-XL on each of the single-hop and multihop datasets in Table 2.")
        text = DATASET_METHOD + "\n" + context
        partial_grid = DATASET_PANELS.split("\n\n", 1)[0]
        for grid, expected in ((DATASET_PANELS, "mean across four datasets"), (partial_grid, "reported average")):
            with self.subTest(expected=expected):
                dataset_page = Page(7, grid + "\n" + text, "text_match", text)
                payload = extract(MODEL_PANEL, page=6, context=(dataset_page,))
                evidence = next(e for e in payload["evidence"] if e["field"] == "raw_value" and e["fragment"] == "48.62")
                result = next(r for r in payload["results"] if r["result_id"] == evidence["entity_id"])
                self.assertEqual(result["aggregation"], expected)
                self.assertEqual(any(e["entity_id"] == result["result_id"] and e["field"] == "aggregation"
                                     and e["page"] == 7 and e["fragment"] == context for e in payload["evidence"]),
                                 grid == DATASET_PANELS)
