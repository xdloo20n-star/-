import unittest

from pdf_pages import PDF, Page, Word
from publication import extract_publication


def fixture_page(number, caption, rows):
    words = []
    for y, items in rows:
        words.extend(Word(value, x - len(value) * 1.6, y - 3, x + len(value) * 1.6, y + 3)
                     for x, value in items)
    layout = "\n".join(" ".join(value for _, value in items) for _, items in rows)
    return Page(number, f"{layout}\n{caption}", "text_match", words=tuple(words), size=(595.276, 841.890))


# XRAG, PDF page 6, Table 2 above the right-column caption.
XRAG_CAPTION = "Table 2: Statistics of XRAG question types in monolin-"
XRAG_COLUMNS = (419.1, 462.8, 479.6, 496.3, 514.4)
XRAG_ROWS = [
    (75.4, ((405.2, "Monolingual"), (438.5, "Retrieval"), (475.4, "Multilingual"), (508.1, "Retrieval"))),
    (85.0, tuple(zip((399.3, 406.3, 412.7, 419.1, 425.8, 432.6, 439.2, *XRAG_COLUMNS[1:]),
                     "De / Es / Zh / Ar De Es Zh Ar".split(), strict=True))),
    *[(y, (*names, *zip(XRAG_COLUMNS, values.split(), strict=True))) for y, names, values in (
        (94.8, ((325.8, "Example"), (352.1, "Number")), "1000 300 300 300 300"),
        (104.5, ((362.8, "Aggregation"),), "313 86 99 106 95"),
        (114.1, ((362.5, "Comparison"),), "260 98 109 81 85"),
        (123.7, ((359.5, "Multi-hop"),), "215 45 40 65 57"),
        (133.2, ((349.6, "Set"),), "212 71 52 48 63"),
        (143.0, ((356.9, "Original"),), "872 291 296 284 263"),
        (152.5, ((359.1, "Corrected"),), "128 9 4 16 37"),
    )],
    (118.6, ((326.1, "Question"),)),
    (147.4, ((324.3, "Answer"),)),
    (173.2, ((81.5, "2024"), *zip((317.0, 333.0, 359.0, 383.0, 400.0, 438.0, 467.0, 487.0, 505.0),
                                 XRAG_CAPTION.split(), strict=True))),
]


# BayesRAG, PDF page 5, Table 4. Right-column prose shares the same visual rows.
RECALL_CAPTION = "Table 4: Retrieval performance comparison on the"
RECALL_COLUMNS = (134.4, 168.1, 201.8, 237.2, 274.3)
RECALL_ROWS = [
    (251.2, ((81.2, "Method"), *zip(RECALL_COLUMNS, ("Recall@1", "Recall@3", "Recall@5", "Recall@10", "Recall@20")),
             (324.3, "achieves"), (472.4, "51.2%"))),
    (262.9, ((79.2, "Vector"), (102.9, "Retrieval"), *zip(RECALL_COLUMNS, "24.4 41.7 46.4 51.6 56.6".split()))),
    (271.1, ((88.5, "BayesRAG"), *zip(RECALL_COLUMNS, "27.5 44.3 54.7 61.7 76.6".split()), (470.0, "2.5%"))),
    (292.9, (*zip((81.5, 100.5, 126.5, 182.5, 236.5, 263.5, 282.5), RECALL_CAPTION.split()),
             (327.0, "high-noise"), (487.0, "(+12.8%)"))),
    (304.9, ((108.0, "MMLongBench-Doc"), (170.0, "dataset."))),
]


# BayesRAG, PDF page 17, Table 7. Each generator label is centered across two rows.
GENERATOR_CAPTION = "Table 7: Performance comparison with different generator models on MMLongBench-Doc."
DOMAINS = "Reports Tutorials Academic Guidebooks Brochures Industry Financial Overall"
GENERATOR_CENTERS = (150.9, 188.9, 230.2, 276.6, 323.7, 365.2, 405.1, 443.0)
GENERATOR_ROWS = [
    (78.9, ((84.3, "Method"), *zip(GENERATOR_CENTERS, DOMAINS.split()), (482.6, "Generator"), (513.5, "Model"))),
    *[(y, (*name, *zip(GENERATOR_CENTERS, values.split()))) for y, name, values in (
        (93.2, ((94.3, "RAGAnything"),), "45.3 46.0 34.6 43.5 39.6 33.3 42.7 41.5"),
        (103.2, ((89, "BayesRAG"), (118.8, "(ours)")), "46.7 53.6 34.8 46.4 39.4 38.6 48.6 44.1"),
        (117.4, ((94.3, "RAGAnything"),), "53.4 47.4 40.2 51.9 39.6 37.0 50.4 47.1"),
        (127.4, ((89, "BayesRAG"), (118.8, "(ours)")), "61.8 57.8 58.2 58.9 54.9 54.4 65.4 59.4"),
    )],
    (98.2, ((494.6, "GPT-4o-mini"),)),
    (122.4, ((491.1, "Qwen3-VL-32B"),)),
    (150.4, tuple(zip((81.7, 99.9, 134.0, 186.7, 222.5, 252.0, 291.6, 328.5, 351.3, 403.6),
                      GENERATOR_CAPTION.split(), strict=True))),
]

# BayesRAG, PDF page 17, Table 8. Overall spans both header rows.
DOCBENCH_CAPTION = "Table 8: Performance comparison on DocBench with different generators."
DOCBENCH_CENTERS = (157.2, 184.5, 212, 240.3, 270.4, 299.9, 328.1, 357.6, 390.6, 429.2)
DOCBENCH_ROWS = [
    (213.5, ((214.9, "Domains"), (347.7, "Types"))),
    (219.4, ((86.5, "Method"), (429.2, "Overall"), (475.6, "Generator"), (511.6, "Model"))),
    (230.0, tuple(zip(DOCBENCH_CENTERS, "Aca. Fin. Gov. Law. News Text Meta MM. Unans.".split()))),
    *[(y, (*name, *zip(DOCBENCH_CENTERS, values.split()))) for y, name, values in (
        (246.6, ((98.3, "RAGAnything"),), "51.4 42.3 43.2 50.2 57.5 77.6 10.0 57.4 11.2 48.7"),
        (258.3, ((92.1, "BayesRAG"), (126.9, "(ours)")), "52.4 37.5 56.0 52.3 66.8 84.2 19.7 43.8 25.8 51.2"),
        (274.9, ((98.3, "RAGAnything"),), "57.4 44.7 49.3 50.7 61.0 82.0 11.2 60.7 19.3 52.4"),
        (286.6, ((92.1, "BayesRAG"), (126.9, "(ours)")), "65.6 53.8 61.4 59.6 70.3 87.3 30.2 62.6 39.5 61.7"),
    )],
    (252.5, ((489.5, "GPT-4o-mini"),)),
    (280.7, ((485.5, "Qwen3-VL-32B"),)),
    (309, tuple(zip((82, 100, 134, 185, 220, 272, 314, 350, 385), DOCBENCH_CAPTION.split(), strict=True))),
]


# BayesRAG, PDF page 6, Table 5. The inference column contains check marks.
ABLATION_CAPTION = "Table 5: Ablation study on Bayesian components."
ABLATION_CENTERS = (274.2, 305.4, 339.3, 377.3, 416.0, 450.0, 482.8, 513.8)
ABLATION_ROWS = [
    (77.5, ((164.1, "Bayesian"),)),
    (81.6, ((81.9, "Method"), (207.2, "Likelihood"), (246.4, "Prior"), *zip(ABLATION_CENTERS, DOMAINS.split()))),
    (85.7, ((164.1, "Inference"),)),
    *[(y, (*metadata, *zip(ABLATION_CENTERS, values.split()))) for y, metadata, values in (
        (98.3, ((94.0, "Embedding-RAG"), (164.1, "%"), (207.2, "-"), (246.2, "-")),
         "39.2 27.9 26.4 31.3 32.6 25.3 19.8 32.7"),
        (107.3, ((85.8, "BayesRAG"), (112.0, "(Linear"), (133.5, "Fusion)"), (164.1, "!"),
                 (193.8, "Linear"), (216.6, "Weighted"), (246.2, "Graph")),
         "45.0 41.3 33.3 39.3 35.0 26.2 35.7 40.6"),
        (116.4, ((85.8, "BayesRAG"), (112.8, "(Layout"), (132.6, "Prior)"), (164.1, "!"),
                 (207.2, "Dempster-Shafer"), (246.2, "Layout")),
         "38.4 33.5 30.5 40.6 36.5 21.2 27.2 36.8"),
        (125.4, ((85.8, "BayesRAG"), (109.8, "(Full)"), (164.1, "!"), (207.2, "Dempster-Shafer"), (246.2, "Graph")),
         "46.7 53.6 34.8 46.4 39.4 38.6 48.6 44.1"),
    )],
    (147.2, tuple(zip((81.0, 96.5, 114.0, 150.0, 168.0, 194.0, 234.0), ABLATION_CAPTION.split(), strict=True))),
]
SETUP_PAGE = Page(4, "We employ GPT-4o-mini as the final response generator across all experiments.", "text_match")

# XRAG, PDF page 7, Table 4. All documents are English, even for non-English queries.
XRAG_MONO_CAPTION = "Table 4: LLM QA accuracy in the XRAG monolingual"
XRAG_MONO_CENTERS = (379.0, 405.1, 431.2, 457.3, 483.4, 509.5)
XRAG_MONO_ROWS = [
    (216.1, ((322.0, "Doc."), (342.5, "Lang."), (444.2, "En"))),
    (228.3, ((325.3, "Query."), (349.0, "Lang."), *zip(XRAG_MONO_CENTERS, "En De Es Zh Ar Avg.".split()))),
    *[(y, (*name, *zip(XRAG_MONO_CENTERS, values.split()))) for y, name, values in (
        (243.9, ((327.2, "GPT-4o"),), "62.40 55.90 56.80 54.70 54.70 55.50"),
        (256.1, ((325.9, "Claude"), (345.3, "3.5")), "42.80 37.40 40.10 37.60 38.50 38.40"),
        (268.3, ((335.9, "Mistral-large"),), "43.30 36.50 39.50 30.60 18.90 31.40"),
        (280.6, ((338.2, "Command-R+"),), "45.70 39.80 41.20 34.30 33.80 37.30"),
        (292.8, ((330.2, "Nova-Pro"),), "54.00 44.80 49.30 37.30 34.30 41.43"),
    )],
    (317, tuple(zip((316.8, 334.0, 351.5, 371.7, 398.9, 422.9, 435.3, 457.8, 499.3),
                    XRAG_MONO_CAPTION.split(), strict=True))),
]
XRAG_SCALE = "Finally, we report each model’s accuracy (%), as determined by the LLM judge panel."
XRAG_MULTI_CAPTION = "Table 5: LLM QA accuracy in the XRAG multilingual retrieval setting,"
XRAG_MULTI_CENTERS = (156.7, 199.5, 241.8, 283.6, 325.8, 368.7, 411.4, 453.9, 497.8)
XRAG_MULTI_ROWS = [
    (78.7, ((88, "Doc."), (108.5, "Lang."),
            *zip((152.9, 199.5, 238.5, 283.6, 322.2, 368.7, 408.0, 453.9, 497.8),
                 "En+En En+De En+En En+Es En+En En+Zh En+En En+Ar Avg.".split(), strict=True))),
    (80.8, tuple(zip((169.3, 254.4, 338.5, 424.1), "De Es Zh Ar".split(), strict=True))),
    (89.3, ((91.2, "Query."), (115.1, "Lang."),
            *zip(XRAG_MULTI_CENTERS, "En De En Es En Zh En Ar (crossling.)".split(), strict=True))),
    (104.3, ((93.1, "GPT-4o"), *zip(XRAG_MULTI_CENTERS, "63.33 61.67 59.33 56.00 63.00 59.33 60.67 53.33 57.58".split()))),
    (114.8, ((91.9, "Claude"), (111.4, "3.5"),
             *zip(XRAG_MULTI_CENTERS, "51.00 45.67 46.33 42.67 46.67 48.00 47.33 39.67 44.00".split()))),
    (170.8, tuple(zip((81.3, 98.4, 115.7, 135.6, 162.5, 186.2, 198.5, 220.6, 260.8, 303.9, 337.3),
                      XRAG_MULTI_CAPTION.split(), strict=True))),
]

# XRAG, PDF page 9, Table 6. The neighboring prose retains a separate geometry.
XRAG_CONTROL_CAPTION = "Table 6: Controlled analysis of GPT-4o on the mul-"
XRAG_CONTROL_CENTERS = (150.1, 181.8, 213.6, 246.0, 275.1)
XRAG_CONTROL_ROWS = [
    (78.8, ((92.9, "GPT-4o"), *zip(XRAG_CONTROL_CENTERS, "En+De En+Es En+Zh En+Ar Avg.".split()),
            (338, "additional"), (435, "languages"))),
    *[(y, (*method, *zip(XRAG_CONTROL_CENTERS, values.split()))) for y, method, values in (
        (93.9, ((105.1, "XRAG-MultiR"),), "61.67 56.00 59.33 53.33 57.58"),
        (105.6, ((87.7, "+EQ"),), "63.00 52.67 60.67 56.67 58.25"),
        (117.4, ((88.7, "+EQ,"), (107.2, "+ES")), "65.00 57.33 62.00 60.33 61.16"),
        (129.1, ((98.5, "MonoRAG"),), "63.33 59.33 63.00 60.67 61.58"),
    )],
    (153.2, tuple(zip((81.7, 100.6, 132, 173.9, 198.3, 222.3, 247.1, 262.2, 281.5),
                      XRAG_CONTROL_CAPTION.split(), strict=True))),
    (165.1, ((88.1, "tilingual"), (126.7, "retrieval"), (161.6, "setting"), (183.5, "of"), (207.3, "XRAG,"))),
]

# Japanese RAG benchmark, PDF page 7, Tables 1 and 2, in the left column.
JAPANESE_STATS_CAPTION = "Table 1: Statistics of the datasets. #Doc, #QA,"
JAPANESE_TYPES_CAPTION = "Table 2: Number of examples and percentages by"
JAPANESE_STATS_ROWS = [
    (72.1, tuple(zip((153.1, 180.3, 206.9, 235.4, 267.5), "#Doc #QA minL medL maxL".split(), strict=True))),
    *[(y, ((100, dataset), *zip((153.1, 181.7, 208.9, 238.7, 269.7), values.split(), strict=True)))
      for y, dataset, values in (
          (80.8, "Wikipedia", "50 165 485 727 1,055"),
          (90.4, "News", "47 173 445 785 1,150"),
          (100, "ProductReview", "40 48 564 1,379 2,838"),
          (109.6, "CompanyRules", "24 175 979 1,607 11,478"),
      )],
    (131.3, tuple(zip((83.9, 104.3, 134.8, 163.2, 178.4, 210, 251, 280),
                      JAPANESE_STATS_CAPTION.split(), strict=True))),
    (193.5, ((223.4, "Product"), (267.9, "Company"))),
    (197.8, ((138.4, "Wikipedia"), (180.9, "News"))),
    (201.9, ((223.4, "Review"), (267.9, "Rules"))),
    (211.6, ((96.5, "#QA"), (108.7, "→"), *zip((138.4, 180.9, 223.4, 267.9), "165 173 48 175".split(), strict=True))),
    *[(y, ((94.1, question), *zip((125.4, 143.1, 168, 185.6, 210.5, 228.2, 254.9, 274.5),
                                  values.split(), strict=True))) for y, question, values in (
        (220.2, "Extracting", "95 (57.6%) 90 (52.0%) 13 (27.1%) 119 (68.0%)"),
        (228.6, "Reasoning", "36 (21.8%) 36 (20.8%) 15 (31.2%) 45 (25.7%)"),
        (236.9, "Computing", "21 (12.7%) 18 (10.4%) 0 (0.0%) 2 (1.1%)"),
        (249.3, "Integrating", "21 (12.7%) 31 (17.9%) 24 (50.0%) 11 (6.3%)"),
    )],
    (245.3, ((95.3, "Information"),)),
    (272.1, tuple(zip((83.6, 102.3, 127.7, 152.2, 178, 212.1, 250.3, 285.4),
                      JAPANESE_TYPES_CAPTION.split(), strict=True))),
]
JAPANESE_CAPTIONS = (f"{JAPANESE_STATS_CAPTION}\n"
                    "minL, medL, and maxL indicate the number of documents, number of QA pairs, and the minimum, "
                    "median, and maximum document lengths (in characters), respectively.\n"
                    f"{JAPANESE_TYPES_CAPTION}")


def extract(*pages):
    return extract_publication(PDF({}, [Page(1, "Retrieval-Augmented Generation benchmark", "text_match"), *pages]))


def find(payload, table, row, column):
    source, = [item for item in payload["evidence"] if item["field"] == "raw_value"
               and (item.get("table"), item.get("table_row"), item.get("table_column")) == (table, row, column)]
    result = next(item for item in payload["results"] if item["result_id"] == source["entity_id"])
    config = next(item for item in payload["configurations"] if item["config_id"] == result["config_id"])
    dataset = next(item for item in payload["datasets"] if item["dataset_id"] == result["dataset_id"])
    return result, config, dataset, source


class NativeBenchmarkTableTest(unittest.TestCase):
    def test_regeneration_training_metrics_use_the_reported_rows_and_units(self):
        # PRO-STEP, PDF page 20, right-column Table 22. Text on the left
        # column shares the y positions of the heading and both data rows.
        caption = "Table 22: Training metrics for the outcome filter main"
        rows = [
            (342.86, tuple(zip((316.97, 337.07, 363.9, 398.58, 421.89, 436.47, 462.59, 492.11, 514.29),
                               caption.split(), strict=True))),
            (354.82, tuple(zip((311.79, 326.48, 341.74, 386.42, 430.53, 451, 483.34, 514.26),
                               "set vs. the regen-augmented set under identical DPO".split(), strict=True))),
            (366.77, ((339.32, "hyperparameters"), (379.46, "(β"), (391.10, "="), (407.01, "0.1)."))),
            (391.79, ((180, "text"), (320.68, "Variant"), (413.55, "Loss"), (465.14, "Margin"), (516.20, "Acc"))),
            (407.34, ((163, "8"), (321.59, "outcome"), (347.74, "filter"), (367.41, "main"),
                      (413.56, "0.42"), (465.14, "2.17"), (516.21, "81%"))),
            (417.31, ((168, "9"), (317.64, "+"), (332.30, "regen"),
                      (413.56, "0.22"), (465.14, "3.48"), (516.21, "91%"))),
        ]
        page = fixture_page(20, caption, rows)
        payload = extract(page)
        self.assertEqual(len(payload["results"]), 6)
        loss, config, dataset, source = find(payload, "Table 22", "+ regen", "Loss")
        self.assertEqual((loss["metric_name"], loss["raw_value"], loss["raw_unit"], config["rag_variant"],
                          dataset["dataset_name"], source["page"], source["verification"]),
                         ("DPO loss", "0.22", None, "outcome filter + regen", None, 20, "text_match"))
        self.assertAlmostEqual((source["bbox"][0] + source["bbox"][2]) / 2, 413.56 / 595.276)
        self.assertAlmostEqual((source["bbox"][1] + source["bbox"][3]) / 2, 417.31 / 841.89)
        accuracy, _, _, _ = find(payload, "Table 22", "outcome filter main", "Acc")
        self.assertEqual((accuracy["metric_name"], accuracy["raw_value"], accuracy["raw_unit"]),
                         ("Preference accuracy", "81%", "%"))
        displaced = [(y, tuple((x + 14 if raw == "3.48" else x, raw) for x, raw in line)) for y, line in rows]
        self.assertEqual(extract(fixture_page(20, caption, displaced))["results"], [])

    def test_xrag_counts_resolve_merged_language_headers(self):
        payload = extract(fixture_page(6, XRAG_CAPTION, XRAG_ROWS))
        self.assertEqual(len(payload["results"]), 35)
        result, config, dataset, source = find(payload, "Table 2", "Question / Aggregation", "Multilingual Retrieval / Es")
        self.assertEqual((result["metric_name"], result["raw_value"], result["raw_unit"], dataset["dataset_name"],
                          config["rag_variant"], source["page"], source["verification"]),
                         ("Aggregation questions", "99", None, "XRAG / Multilingual Retrieval / Es", "XRAG benchmark construction",
                          6, "text_match"))
        self.assertAlmostEqual((source["bbox"][0] + source["bbox"][2]) / 2, 479.6 / 595.276)
        self.assertAlmostEqual((source["bbox"][1] + source["bbox"][3]) / 2, 104.5 / 841.89)

    def test_xrag_rejects_an_inconsistent_language_partition(self):
        rows = [(y, tuple((x, "98" if value == "99" else value) for x, value in items)) for y, items in XRAG_ROWS]
        self.assertEqual(extract(fixture_page(6, XRAG_CAPTION, rows))["results"], [])

    def test_recall_uses_header_cutoff_and_primary_benchmark(self):
        payload = extract(fixture_page(5, RECALL_CAPTION, RECALL_ROWS))
        self.assertEqual(len(payload["results"]), 10)
        result, config, dataset, source = find(payload, "Table 4", "BayesRAG", "Recall@20")
        self.assertEqual((result["metric_name"], result["raw_value"], result["metric_k"], result["raw_unit"],
                          config["rag_variant"], dataset["dataset_name"], source["page"], source["verification"]),
                         ("Page-evidence Recall", "76.6", 20, None, "BayesRAG", "MMLongBench-Doc / Overall",
                          5, "text_match"))
        self.assertAlmostEqual((source["bbox"][0] + source["bbox"][2]) / 2, 274.3 / 595.276)
        self.assertAlmostEqual((source["bbox"][1] + source["bbox"][3]) / 2, 271.1 / 841.89)

    def test_recall_rejects_a_value_moved_out_of_its_column(self):
        rows = [(y, tuple((x + 13 if value == "44.3" else x, value) for x, value in items))
                for y, items in RECALL_ROWS]
        self.assertEqual(extract(fixture_page(5, RECALL_CAPTION, rows))["results"], [])

    def test_generator_assignment_uses_the_model_between_each_row_pair(self):
        payload = extract(fixture_page(17, GENERATOR_CAPTION, GENERATOR_ROWS))
        self.assertEqual(len(payload["results"]), 32)
        result, config, dataset, source = find(payload, "Table 7", "BayesRAG (ours)", "Qwen3-VL-32B / Reports")
        self.assertEqual((result["metric_name"], result["raw_value"], result["raw_unit"], result["reported_scale"],
                          config["generator_model"], dataset["dataset_name"], source["page"], source["verification"]),
                         ("LLM Score", "61.8", None, "percent", "Qwen3-VL-32B", "MMLongBench-Doc / Reports",
                          17, "text_match"))
        self.assertAlmostEqual((source["bbox"][0] + source["bbox"][2]) / 2, 150.9 / 595.276)

    def test_generator_grid_rejects_a_model_outside_its_row_pair(self):
        misplaced = [(107 if y == 98.2 else y, row) for y, row in GENERATOR_ROWS]
        self.assertEqual(extract(fixture_page(17, GENERATOR_CAPTION, misplaced))["results"], [])

    def test_docbench_keeps_question_types_separate_from_overall(self):
        payload = extract(fixture_page(17, DOCBENCH_CAPTION, DOCBENCH_ROWS))
        self.assertEqual(len(payload["results"]), 40)
        result, config, dataset, source = find(payload, "Table 8", "BayesRAG (ours)", "Qwen3-VL-32B / Text")
        self.assertEqual((result["metric_name"], result["raw_value"], config["generator_model"], dataset["dataset_name"],
                          source["page"], source["verification"]),
                         ("LLM accuracy", "87.3", "Qwen3-VL-32B", "DocBench / Text", 17, "text_match"))
        self.assertAlmostEqual((source["bbox"][0] + source["bbox"][2]) / 2, 299.9 / 595.276)
        misleading = [(y, tuple((x, "All" if word == "Overall" else word) for x, word in items))
                      for y, items in DOCBENCH_ROWS]
        self.assertEqual(extract(fixture_page(17, DOCBENCH_CAPTION, misleading))["results"], [])

    def test_ablation_and_generator_repeat_requires_equal_primary_configuration(self):
        appendix = fixture_page(17, GENERATOR_CAPTION, GENERATOR_ROWS)
        ablation = fixture_page(6, ABLATION_CAPTION, ABLATION_ROWS)
        payload = extract(SETUP_PAGE, ablation, appendix)
        first, _, _, _ = find(payload, "Table 5", "BayesRAG (Full)", "Overall")
        repeat, _, _, _ = find(payload, "Table 7", "BayesRAG (ours)", "GPT-4o-mini / Overall")
        other, _, _, _ = find(payload, "Table 7", "BayesRAG (ours)", "Qwen3-VL-32B / Overall")
        self.assertEqual((first["raw_value"], repeat["raw_value"], other["raw_value"]), ("44.1", "44.1", "59.4"))
        self.assertEqual(repeat["duplicate_of"], first["result_id"])
        self.assertNotIn("duplicate_of", other)
        self.assertEqual(sum(bool(result.get("duplicate_of")) for result in payload["results"]), 8)
        uncertain = extract(ablation, appendix)
        self.assertFalse(any(row.get("duplicate_of") for row in uncertain["results"]))

    def test_xrag_accuracy_keeps_query_and_document_languages(self):
        page = fixture_page(7, f"{XRAG_MONO_CAPTION}\n{XRAG_SCALE}", XRAG_MONO_ROWS)
        payload = extract(page)
        self.assertEqual(len(payload["results"]), 30)
        result, config, dataset, source = find(payload, "Table 4", "GPT-4o", "En documents / Es queries")
        self.assertEqual((result["metric_name"], result["raw_value"], result["raw_unit"], result["reported_scale"],
                          result["aggregation"], config["generator_model"], dataset["dataset_name"], source["page"]),
                         ("QA accuracy", "56.80", None, "percent", "mean over questions", "GPT-4o",
                          "XRAG / monolingual retrieval / Es", 7))
        self.assertAlmostEqual((source["bbox"][0] + source["bbox"][2]) / 2, 431.2 / 595.276)
        average, _, _, _ = find(payload, "Table 4", "GPT-4o", "Avg.")
        self.assertEqual(average["aggregation"], "macro average over four non-English languages")

    def test_xrag_rejects_accuracy_grid_with_mismatched_average(self):
        rows = [(y, tuple((x, "59.50" if value == "55.50" else value) for x, value in words))
                for y, words in XRAG_MONO_ROWS]
        self.assertEqual(extract(fixture_page(7, f"{XRAG_MONO_CAPTION}\n{XRAG_SCALE}", rows))["results"], [])

    def test_xrag_control_compares_identical_and_replaced_language_conditions(self):
        multilingual = fixture_page(8, f"{XRAG_MULTI_CAPTION}\n{XRAG_SCALE}", XRAG_MULTI_ROWS)
        controlled = fixture_page(9, XRAG_CONTROL_CAPTION, XRAG_CONTROL_ROWS)
        payload = extract(multilingual, controlled)
        self.assertEqual(len(payload["results"]), 38)
        main, _, _, _ = find(payload, "Table 5", "GPT-4o", "En+Es / Es queries")
        repeat, generator, dataset, source = find(payload, "Table 6", "XRAG-MultiR", "En+Es")
        changed, _, _, _ = find(payload, "Table 6", "+EQ", "En+Es")
        self.assertEqual((repeat["metric_name"], repeat["raw_value"], repeat["duplicate_of"],
                          generator["generator_model"], dataset["dataset_name"], source["page"]),
                         ("QA accuracy", "56.00", main["result_id"], "GPT-4o", "XRAG / multilingual retrieval / Es", 9))
        self.assertEqual(changed["raw_value"], "52.67")
        self.assertNotIn("duplicate_of", changed)
        self.assertEqual(sum(bool(row.get("duplicate_of")) for row in payload["results"]), 9)
        self.assertAlmostEqual((source["bbox"][0] + source["bbox"][2]) / 2, 181.8 / 595.276)
        corrupted = [(y, [(x, "En+Pt" if value == "En+Es" else value) for x, value in row])
                     for y, row in XRAG_CONTROL_ROWS]
        recovered = extract(multilingual, fixture_page(9, XRAG_CONTROL_CAPTION, corrupted))
        self.assertEqual(len(recovered["results"]), 18)

    def test_japanese_benchmark_preserves_percent_and_length_units(self):
        payload = extract(fixture_page(7, JAPANESE_CAPTIONS, JAPANESE_STATS_ROWS))
        self.assertEqual(len(payload["results"]), 56)
        maximum, _, data, source = find(payload, "Table 1", "CompanyRules", "maxL")
        self.assertEqual((maximum["metric_name"], maximum["raw_value"], maximum["reported_value"],
                          maximum["raw_unit"], data["dataset_name"], source["page"]),
                         ("Maximum document length", "11,478", 11478, "characters", "CompanyRules", 7))
        value, _, data, source = find(payload, "Table 2", "Computing", "Product Review / Percentage")
        self.assertEqual((value["metric_name"], value["metric_variant"], value["raw_value"], value["raw_unit"],
                          value["sample_size"], data["dataset_name"], source["verification"]),
                         ("Question proportion by type", "Computing", "0.0%", "%", 48, "ProductReview", "text_match"))
        self.assertAlmostEqual((source["bbox"][0] + source["bbox"][2]) / 2, 228.2 / 595.276)
        total, _, _, _ = find(payload, "Table 2", "#QA →", "Product Review / #QA")
        canonical, _, _, _ = find(payload, "Table 1", "ProductReview", "#QA")
        self.assertEqual(total["duplicate_of"], canonical["result_id"])

    def test_japanese_benchmark_rejects_a_percentage_with_the_wrong_denominator(self):
        rows = [(y, tuple((x, "(7.5%)" if value == "(6.3%)" else value) for x, value in words))
                for y, words in JAPANESE_STATS_ROWS]
        payload = extract(fixture_page(7, JAPANESE_CAPTIONS, rows))
        self.assertEqual(len(payload["results"]), 20)
        self.assertFalse(any(item.get("table") == "Table 2" for item in payload["evidence"]))
