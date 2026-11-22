import unittest

from pdf_pages import Page, Word
from publication import Payload, add_table
from threadrag_tables import read_threadrag_tables


DEFINITION = Page(6, "Exact Match (EM) and F1 scores to measure absolute correctness and token-level overlap, applying\n"
                     "standard text normalization to both predictions and\nground-truth answers prior to computation.",
                  "text_match")


def fixture(number, caption, caption_y, caption_x, body):
    rows = [(caption_y + 12 * i, [(caption_x + n * 14, token) for n, token in enumerate(line.split())])
            for i, line in enumerate(caption.splitlines())]
    # Long caption lines can reach the edge of the column without overflowing it.
    rows = [(y, [(min(x, caption_x + 203), token) for x, token in items]) for y, items in rows]
    words = tuple(Word(token, x - 2, y - 2, x + 2, y + 2)
                  for y, items in body + rows for x, token in items)
    return Page(number, caption, "text_match", words=words, size=(595, 842))


def cells(text, *centers):
    return list(zip(centers, text.split(), strict=True))


def ablation_page(misplaced=False):
    body = [
        (79, cells("Ablation Setting Datasets", 95, 124, 215)),
        (93, cells("SN STG EDP HotpotQA 2Wiki MuSiQue", 83, 105, 130, 167, 215, 263)),
        (108, cells("✓ ✓ ✓ .422 .372 .324", 83, 105, 130, 167, 215, 263)),
    ]
    for y, settings, scores in [
        (119, (), ".238 (↓ 18.4%) .224 (↓ 14.8%) .172 (↓ 15.2%)"),
        (129, (130,), ".326 (↓ 9.6%) .308 (↓ 6.4%) .232 (↓ 9.2%)"),
        (140, (83, 105), ".398 (↓ 2.4%) .334 (↓ 3.8%) .286 (↓ 3.8%)"),
        (150, (105, 130), ".388 (↓ 3.4%) .342 (↓ 3%) .292 (↓ 3.2%)"),
    ]:
        body.append((y, [(x, "✓") for x in settings] + cells(scores, 155, 167, 179, 203, 215, 227,
                                                               251, 263, 288 if misplaced else 275)))
    return fixture(7, "Table 2: Ablation study of the STM (SN, STG) and\nEDP modules across three datasets.", 173, 82,
                   body)


def cost_page(misplaced=False):
    return fixture(7, "Table 3: Comparison of construction (Const.), inference\n(Inf.), and total token costs.", 149, 317, [
        (78, cells("HotpotQA MuSiQue", 395, 480)),
        (82, [(325, "Method")]),
        (91, cells("Const. Inf. Total Const. Inf. Total", 367, 394, 422, 452, 478, 507)),
        (104, cells("LightRAG 26.2M 47.4k 26.25M 36.4M 57.6k 36.46M", 328, 367, 394, 422, 452, 478, 507)),
        (113, cells("KnowTrace 0.15M 72.6k 222.6k 0.22M 94.4k 314.4k", 330, 367, 394, 422, 452,
                    493 if misplaced else 478, 507)),
        (126, cells("ThreadRAG 0 58.5k 58.5k 0 73.4k 73.4k", 330, 367, 394, 422, 452, 478, 507)),
    ])


def graphfree_page(bad_header=False):
    return fixture(14, "Table 6: Controlled Graph-Free ablation in terms of\nExact Match (EM). The Graph-Free variant preserves\n"
                       "retrieval and iterative reasoning while removing graph\nconstruction and EDP.", 262, 317, [
        (213, cells("Method HotpotQA Wiki MuSiQue" if bad_header else "Method HotpotQA 2Wiki MuSiQue", 337, 400, 446, 490)),
        (229, cells("Graph-Free 0.274 0.258 0.206", 343, 400, 446, 490)),
        (239, cells("ThreadRAG 0.422 0.372 0.324", 344, 400, 446, 490)),
    ])


def selector_page(misplaced=False):
    body = [(636, cells("Subgraph Strategy EM ∆ vs. EDP", 351, 388, 431, 461, 473, 490)),
            (651, cells("No subgraph 0.286", 337, 362, 431)), (655, [(478, "−0.038")]),
            (661, cells("BFS (depth = 3) 0.294", 340, 362, 378, 387, 431)),
            (665, [(454 if misplaced else 478, "−0.030")]),
            (671, cells("DFS (depth = 3) 0.302", 340, 362, 378, 387, 431)), (675, [(478, "−0.022")]),
            (681, cells("EDP 0.324 –", 341, 431, 478))]
    return fixture(14, "Table 7: Comparison of evidence selection strategies on\nMuSiQue using LLaMA3-8B-Instruct. All methods use\n"
                       "the same candidate graph and evidence budget.", 705, 317, body)


def boundary_page(bad_value=False):
    return fixture(15, "Table 8: Intrinsic evaluation of the semantic boundary\n"
                       "predictor on 50 held-out Wikipedia documents. Scores\n"
                       "measure agreement with teacher-generated reference\nboundaries.", 119, 82, [
        (79, cells("Precision Recall F1", 134, 188, 233)),
        (95, cells("0.948 8.61 0.902" if bad_value else "0.948 0.861 0.902", 134, 188, 233)),
    ])


def case_page(total="270"):
    return fixture(13, "Table 4: Case on MuSiQue.", 545, 253, [
        (254, cells("# Semantic Topology Graph (STG) Construction:", 85, 115, 163, 203, 236, 289)),
        (265, cells("- Graph initialized with 270 context-intact nodes and 878 structural (narrative/semantic/lexical) edges.",
                    84, 99, 130, 158, 175, 209, 246, 265, 281, 307, 375, 437)),
        (313, cells("- 5 query-relevant anchor nodes identified based on the extracted entities.",
                    84, 90, 120, 160, 185, 215, 244, 261, 274, 298, 330)),
        (326, cells("# Graph Pruning & Topological Linearization:", 85, 106, 144, 171, 209, 281)),
        (337, cells(f"- 9 highly activated nodes retained out of {total} after energy threshold pruning.",
                    84, 90, 106, 135, 164, 191, 214, 225, 238, 255, 278, 309, 343)),
    ])


def case_payload(page):
    payload = Payload({}, {})
    duplicates = {}
    for table in read_threadrag_tables([page], []):
        add_table(payload, table, True, duplicate_index=duplicates)
    return payload


def at(page, label, row, column):
    payload = Payload({}, {})
    for table in read_threadrag_tables([DEFINITION, page], []):
        add_table(payload, table, True)
    source = next(e for e in payload.evidence if e["field"] == "raw_value" and e.get("table") == label
                  and e.get("table_row") == row and e.get("table_column") == column)
    result = next(r for r in payload.results if r["result_id"] == source["entity_id"])
    config = next(c for c in payload.configurations if c["config_id"] == result["config_id"])
    dataset = next(d for d in payload.datasets if d["dataset_id"] == result["dataset_id"])
    return result, config, dataset, source


class ThreadRagTableTest(unittest.TestCase):
    def test_ablation_binds_arrow_and_checked_components_to_dataset(self):
        self.assertEqual(sum(len(table.columns) for table in read_threadrag_tables([DEFINITION, ablation_page()], [])), 27)
        result, config, dataset, source = at(ablation_page(), "Table 2", "STG+EDP", "2Wiki / ↓")
        self.assertEqual((result["metric_name"], result["raw_value"], result["raw_unit"], config["rag_variant"],
                          dataset["dataset_name"], source["bbox"]),
                         ("EM decrease", "↓ 3%", "%", "ThreadRAG ablation: STG+EDP; no SN", "2WikiMultihopQA",
                          (213 / 595, 148 / 842, 229 / 595, 152 / 842)))

    def test_ablation_rejects_delta_that_does_not_align_with_dataset(self):
        self.assertEqual(read_threadrag_tables([DEFINITION, ablation_page(misplaced=True)], []), [])

    def test_cost_keeps_printed_magnitude_and_semantic_unit(self):
        result, config, dataset, source = at(cost_page(), "Table 3", "LightRAG", "HotpotQA / Const.")
        self.assertEqual((result["metric_name"], result["metric_variant"], result["raw_value"], result["raw_unit"],
                          result["reported_value"], config["rag_variant"], dataset["dataset_name"], source["bbox"]),
                         ("Token cost", "construction", "26.2M", "tokens", 26200000, "LightRAG", "HotpotQA",
                          (365 / 595, 102 / 842, 369 / 595, 106 / 842)))

    def test_cost_rejects_number_with_shifted_phase_column(self):
        self.assertEqual(read_threadrag_tables([DEFINITION, cost_page(misplaced=True)], []), [])

    def test_graphfree_preserves_unknown_generator_and_normalization(self):
        result, config, dataset, source = at(graphfree_page(), "Table 6", "Graph-Free", "2Wiki / EM")
        self.assertEqual((result["metric_name"], result["metric_variant"], result["raw_value"], config["generator_model"],
                          dataset["dataset_name"], source["bbox"]),
                         ("EM", "exact match after standard text normalization", "0.258", None, "2WikiMultihopQA",
                          (444 / 595, 227 / 842, 448 / 595, 231 / 842)))

    def test_graphfree_rejects_wrong_dataset_heading(self):
        self.assertEqual(read_threadrag_tables([DEFINITION, graphfree_page(bad_header=True)], []), [])

    def test_selection_matches_shifted_delta_to_its_seed_row(self):
        result, config, dataset, source = at(selector_page(), "Table 7", "BFS (depth = 3)", "∆ vs. EDP")
        self.assertEqual((result["metric_name"], result["raw_value"], result["raw_unit"], config["generator_model"],
                          dataset["dataset_name"], source["bbox"]),
                         ("EM difference vs EDP", "−0.030", None, "LLaMA3-8B-Instruct", "MuSiQue",
                          (476 / 595, 663 / 842, 480 / 595, 667 / 842)))

    def test_selection_rejects_misaligned_difference(self):
        self.assertEqual(read_threadrag_tables([DEFINITION, selector_page(misplaced=True)], []), [])

    def test_boundary_predictor_keeps_document_sample_and_score(self):
        result, config, dataset, source = at(boundary_page(), "Table 8", "Semantic boundary predictor", "Recall")
        self.assertEqual((result["metric_name"], result["raw_value"], result["sample_size"], config["generator_model"],
                          dataset["dataset_name"], source["bbox"]),
                         ("Recall", "0.861", 50, None, "held-out Wikipedia semantic-boundary evaluation",
                          (186 / 595, 93 / 842, 190 / 595, 97 / 842)))

    def test_boundary_predictor_rejects_number_outside_fraction_scale(self):
        self.assertEqual(read_threadrag_tables([DEFINITION, boundary_page(bad_value=True)], []), [])

    def test_case_reads_retained_nodes_and_links_the_repeated_graph_size(self):
        result, config, dataset, source = at(case_page(), "Table 4", "Graph pruning", "Retained nodes")
        self.assertEqual((result["metric_name"], result["raw_value"], config["rag_variant"], dataset["dataset_name"],
                          source["bbox"]), ("Retained node count", "9", "ThreadRAG Table 4 example", "MuSiQue",
                                            (88 / 595, 335 / 842, 92 / 595, 339 / 842)))
        payload = case_payload(case_page())
        initial = next(r for r in payload.results if r["metric_name"] == "Semantic node count")
        repeats = [r for r in payload.results if r.get("duplicate_of")]
        self.assertEqual((len(payload.results), len(repeats), repeats[0]["duplicate_of"]), (5, 1, initial["result_id"]))

    def test_case_keeps_a_different_printed_denominator_as_its_own_result(self):
        payload = case_payload(case_page("269"))
        self.assertFalse(any(r.get("duplicate_of") for r in payload.results))
        self.assertIn("269", [r["raw_value"] for r in payload.results if r["metric_name"] == "Semantic node count"])
