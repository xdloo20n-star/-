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

SELF_RAG_NOTE = ("Self-RAG∗ is trained with a different base LLM, namely LLaMA2 (Touvron et al., 2023); therefore, "
                 "we compare the results of FLAN-T5-XL (3B) with the results from Self-RAG with LLaMA2 (7B) "
                 "and the results of others with the results from Self-RAG with LLaMA2 (13B).")
SELF_RAG_DATASET_ROW = ("    Single-step   Adaptive   Self-RAG∗                 2.20   11.20     18.40    0.63   0.50"
                        "    31.40   39.00   33.60       0.63    0.17")
ADAPTIVE_DATASET_ROW = ("                             Adaptive Retrieval       13.40   23.10     17.60    0.50   0.55"
                        "    28.20   36.00   33.00       0.50    0.56")
XXL_APPENDIX_PANEL = """Table 7: Results on each of a collection of datasets with FLAN-T5-XXL (11B) as the LLM.
                                                                   SQuAD                            Natural Questions                             TriviaQA
 Data          Types      Methods                  EM       F1       Acc     Step   Time   EM       F1     Acc     Step   Time   EM       F1        Acc     Step   Time
                          No Retrieval             7.00    14.40     8.40    0.00   0.08   18.80   25.50   20.40    0.00   0.08   32.80   39.20     35.40    0.00   0.08
                          Single-step Approach     28.80   40.80     35.00   1.00   1.00   41.40   51.20   47.60    1.00   1.00   56.00   64.70     61.80    1.00   1.00
 Single-step   Adaptive   Self-RAG∗                 1.60   11.90     20.80   0.59   0.31   39.20   47.10   42.40    0.75   0.09   14.60   33.70     60.20    0.76   0.22
"""
GPT_APPENDIX_PANEL = """Table 8: Results on each of a collection of datasets with GPT-3.5 (Turbo) as the LLM.
                                                                   SQuAD                            Natural Questions                             TriviaQA
 Data          Types      Methods                  EM       F1       Acc     Step   Time   EM       F1     Acc     Step   Time   EM       F1        Acc     Step   Time
                          No Retrieval             16.00   29.20     23.80   0.00   0.62   39.80   55.70   55.00    0.00   0.56   64.00   75.60     75.80    0.00   0.68
                          Single-step Approach     18.00   33.80     29.20   1.00   1.00   32.40   46.80   54.80    1.00   1.00   55.20   66.50     65.80    1.00   1.00
 Single-step   Adaptive   Self-RAG∗                 1.60   11.90     20.80   0.59   1.91   39.20   47.10   42.40    0.75   0.52   14.60   33.70     60.20    0.76   1.59
"""
BENCHMARK_LINK = ("Table 1 shows our main results averaged over all considered datasets. In addition, we report "
                  "the more granular results with FLAN-T5-XL on each of the single-hop and multihop datasets in "
                  "Table 2 (and more with different LLMs in Table 7 and Table 8 of Appendix).")


EFFECTIVENESS = "For effectiveness, we use F1, EM, and Accuracy (Acc), following the standard evaluation protocol."
EFFICIENCY = ("For efficiency, we measure the number of retrievaland-generate steps and the average time "
              "for answering each query relative to the one-step approach.")
CODE_LINK = "Code is available at: https://\ngithub.com/starsuzi/Adaptive-RAG."
GENERATOR_METHOD = ("Regarding the LLMs that are used to generate answers, we use the FLAN-T5 series models "
                    "(Chung et al., 2022) of XL with 3B parameters and XXL with 11B parameters, and the GPT-3.5 "
                    "model (gpt-3.5-turbo-instruct).")
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

    def test_marked_method_uses_caption_baseline_without_replacing_column_locator(self):
        layout = MODEL_PANEL.replace("Self-RAG∗ is trained with a different base LLM, namely LLaMA2.", SELF_RAG_NOTE)
        payload = extract(layout, page=6)
        self.assertEqual(len(payload["results"]), 30)
        for raw, model, printed in (("20.79", "LLaMA2 (7B)", "FLAN-T5-XL (3B)"),
                                    ("22.98", "LLaMA2 (13B)", "FLAN-T5-XXL (11B)")):
            with self.subTest(model=model):
                evidence = next(e for e in payload["evidence"] if e["field"] == "raw_value" and e["fragment"] == raw)
                self.assertEqual((evidence["page"], evidence["table"], evidence["table_row"], evidence["table_column"]),
                                 (6, "Table 1", "Self-RAG∗", f"{printed} / F1 / reported average"))
                result = next(r for r in payload["results"] if r["result_id"] == evidence["entity_id"])
                config = next(c for c in payload["configurations"] if c["config_id"] == result["config_id"])
                self.assertEqual((config["rag_variant"], config["generator_model"]), ("Self-RAG", model))
                self.assertTrue(any(e["entity_type"] == "configuration" and e["entity_id"] == config["config_id"]
                                    and e["field"] == "generator_model" and e["page"] == 6
                                    and e["verification"] == "text_match" and e["fragment"] == SELF_RAG_NOTE
                                    for e in payload["evidence"]))

    def test_marked_model_transfers_only_to_paper_linked_tables_with_same_marker(self):
        model_page = Page(6, MODEL_PANEL.replace("Self-RAG∗ is trained with a different base LLM, namely LLaMA2.",
                                                SELF_RAG_NOTE), "text_match")
        grid = DATASET_PANELS.replace("\n\n", f"\n{SELF_RAG_DATASET_ROW}\n\n", 1)
        for table, printed, marker, expected in (("Table 2", "FLAN-T5-XL (3B)", "∗", "LLaMA2 (7B)"),
                                                 ("Table 7", "FLAN-T5-XXL (11B)", "∗", "LLaMA2 (13B)"),
                                                 ("Table 7", "FLAN-T5-XL (3B)", "∗", None),
                                                 ("Table 9", "FLAN-T5-XL (3B)", "∗", None),
                                                 ("Table 2", "FLAN-T5-XL (3B)", "†", None)):
            with self.subTest(table=table, marker=marker):
                layout = (grid.replace("Table 2", table).replace("FLAN-T5-XL (3B)", printed)
                          .replace("Self-RAG∗", f"Self-RAG{marker}"))
                context = Page(7, layout, "text_match", BENCHMARK_LINK)
                payload = extract(layout, context=(model_page, context))
                config_by_id = {c["config_id"]: c for c in payload["configurations"]}
                marked_evidence = [e for e in payload["evidence"] if e["field"] == "raw_value" and e["page"] == 7
                                   and e.get("table_row", "").startswith("Self-RAG")]
                if expected is None:
                    self.assertEqual(marked_evidence, [])
                    continue
                evidence = next(e for e in marked_evidence if e["fragment"] == "11.20")
                self.assertEqual(evidence["table_column"], f"{printed} / SQuAD / F1")
                result = next(r for r in payload["results"] if r["result_id"] == evidence["entity_id"])
                self.assertEqual(config_by_id[result["config_id"]]["generator_model"], expected)

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

    def test_fractional_step_counts_establish_a_query_mean(self):
        grid = DATASET_PANELS.replace("\n\n", f"\n{ADAPTIVE_DATASET_ROW}\n\n", 1)
        methodology = Page(6, EFFICIENCY, "text_match")
        for source, value, expected in ((methodology, "0.50", "mean across queries"),
                                        (None, "0.50", None), (methodology, "2.00", None)):
            with self.subTest(source=source, value=value):
                payload = extract(grid.replace("0.50", value), context=(source,) if source else ())
                evidence = next(e for e in payload["evidence"] if e["field"] == "raw_value"
                                and e.get("table_row") == "Adaptive Retrieval" and e["fragment"] == value)
                result = next(r for r in payload["results"] if r["result_id"] == evidence["entity_id"])
                config = next(c for c in payload["configurations"] if c["config_id"] == result["config_id"])
                dataset = next(d for d in payload["datasets"] if d["dataset_id"] == result["dataset_id"])
                self.assertEqual((evidence["page"], evidence["table"], evidence["table_column"]),
                                 (7, "Table 2", "FLAN-T5-XL (3B) / SQuAD / Step" + (f" / {expected}" if expected else "")))
                self.assertEqual((config["generator_model"], dataset["dataset_name"], result["aggregation"]),
                                 ("FLAN-T5-XL (3B)", "SQuAD", expected))
                self.assertEqual(result["metric_name"], "Retrieval-and-generation steps per query" if source else "Step")
                self.assertIsNone(result["raw_unit"])

    def test_relative_time_uses_the_colocated_one_step_reference(self):
        methodology = Page(6, EFFICIENCY, "text_match")
        for text, expected in (
                (DATASET_PANELS, "denominator FLAN-T5-XL (3B) / Natural Questions / Single-step Approach"),
                (DATASET_PANELS.replace("44.60       1.00    1.00", "44.60       1.00    1.02"), None)):
            with self.subTest(expected=expected):
                payload = extract(text, context=(methodology,))
                evidence = next(e for e in payload["evidence"] if e["field"] == "raw_value"
                                and e.get("table_row") == "No Retrieval" and e["fragment"] == "0.13")
                result = next(r for r in payload["results"] if r["result_id"] == evidence["entity_id"])
                self.assertEqual((result["metric_name"], result["metric_variant"], result["raw_unit"]),
                                 ("Relative time per query", expected, None))
                self.assertEqual((evidence["page"], evidence["table"], evidence["table_column"]),
                                 (7, "Table 2", "FLAN-T5-XL (3B) / Natural Questions / Time / mean across queries"))
                proof = [e for e in payload["evidence"] if e["entity_id"] == result["result_id"]
                         and e["field"] == "metric_variant"]
                self.assertEqual([e["page"] for e in proof], [6] if expected else [])

    def test_self_rag_duplicate_requires_the_same_model_dataset_and_measurement(self):
        aggregate = (MODEL_PANEL.replace("Self-RAG∗ is trained with a different base LLM, namely LLaMA2.", SELF_RAG_NOTE)
                     .replace("FLAN-T5-XXL (11B)\n", "FLAN-T5-XXL (11B)                      GPT-3.5 (Turbo)\n")
                     .replace("Acc     Step   Time\n", "Acc     Step   Time   EM       F1     Acc     Step   Time\n")
                     .replace("19.33   0.00   0.08\n", "19.33   0.00   0.08   35.77   48.56   44.27   0.00   0.71\n")
                     .replace("34.13   0.74   0.23\n", "34.13   0.74   0.23   10.87   22.98   34.13   0.74   1.50\n")
                     .replace("43.77   1.35   2.00\n", "43.77   1.35   2.00   37.97   50.91   48.97   1.03   1.46\n"))
        source = Page(6, aggregate + EFFICIENCY, "text_match")
        link = Page(7, BENCHMARK_LINK, "text_match")
        for gpt, matching_f1 in ((GPT_APPENDIX_PANEL, True), (GPT_APPENDIX_PANEL.replace("1.60   11.90", "1.60   11.91"), False)):
            with self.subTest(matching_f1=matching_f1):
                payload = extract(XXL_APPENDIX_PANEL + "\n" + gpt, page=15, context=(source, link))
                results = {r["result_id"]: r for r in payload["results"]}
                locators = [e for e in payload["evidence"] if e["field"] == "raw_value"
                            and e["page"] == 15 and e.get("table_row") == "Self-RAG∗"]
                f1 = next(e for e in locators if e["table"] == "Table 8" and " / SQuAD / F1" in e["table_column"])
                prior = next(e for e in locators if e["table"] == "Table 7" and " / SQuAD / F1" in e["table_column"])
                result = results[f1["entity_id"]]
                config = next(c for c in payload["configurations"] if c["config_id"] == result["config_id"])
                dataset = next(d for d in payload["datasets"] if d["dataset_id"] == result["dataset_id"])
                self.assertEqual((result["metric_name"], result["raw_value"], config["generator_model"],
                                  dataset["dataset_name"], f1["table_column"]),
                                 ("F1", "11.90" if matching_f1 else "11.91", "LLaMA2 (13B)", "SQuAD",
                                  "GPT-3.5 (Turbo) / SQuAD / F1"))
                self.assertEqual(result.get("duplicate_of"), prior["entity_id"] if matching_f1 else None)
                if matching_f1:
                    for field in ("config_id", "dataset_id", "experiment_id", "task_id"):
                        self.assertEqual(result[field], results[prior["entity_id"]][field])
                time = next(e for e in locators if e["table"] == "Table 8" and " / SQuAD / Time" in e["table_column"])
                self.assertIsNone(results[time["entity_id"]].get("duplicate_of"))

    def test_published_generator_identifiers_keep_literal_table_locators(self):
        source = Page(6, GENERATOR_METHOD, "text_match")
        for printed, canonical in (("FLAN-T5-XL (3B)", "FLAN-T5-XL 3B"), ("FLAN-T5-XXL (11B)", "FLAN-T5-XXL 11B"),
                                   ("GPT-3.5 (Turbo)", "gpt-3.5-turbo-instruct"),
                                   ("FLAN-T5-XL (4B)", "FLAN-T5-XL (4B)"), ("GPT-3.5 (Beta)", "GPT-3.5 (Beta)")):
            with self.subTest(printed=printed):
                layout = DATASET_PANELS.replace("FLAN-T5-XL (3B)", printed)
                payload = extract(layout, context=(source,))
                evidence = next(e for e in payload["evidence"] if e["field"] == "raw_value" and e["fragment"] == "46.15")
                self.assertEqual(evidence["table_column"], f"{printed} / HotpotQA / F1")
                result = next(r for r in payload["results"] if r["result_id"] == evidence["entity_id"])
                config = next(c for c in payload["configurations"] if c["config_id"] == result["config_id"])
                self.assertEqual(config["generator_model"], canonical)
                model_source = next(e for e in payload["evidence"] if e["field"] == "generator_model"
                                    and e["entity_id"] == config["config_id"])
                self.assertEqual(model_source["page"], 6 if canonical != printed else 7)
                self.assertEqual(model_source["verification"], "text_match")

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
