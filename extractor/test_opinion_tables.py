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


def positioned_grid(number, caption, rows):
    words = tuple(Word(text, x - 2, y - 2, x + 2, y + 2)
                  for y, cells in rows for x, text in cells)
    return Page(number, caption, "text_match", words=words, size=(595, 842))


def result_at(payload, table, row, column):
    source = next(e for e in payload["evidence"] if e["field"] == "raw_value" and e.get("table") == table
                  and e.get("table_row") == row and e.get("table_column") == column)
    result = next(r for r in payload["results"] if r["result_id"] == source["entity_id"])
    config = next(c for c in payload["configurations"] if c["config_id"] == result["config_id"])
    dataset = next(d for d in payload["datasets"] if d["dataset_id"] == result["dataset_id"])
    return result, config, dataset, source


class OpinionSupplementTest(unittest.TestCase):
    def distance_page(self, final_delta_x=498):
        caption = "Table 3: Empirical W1 between corpus-level and retrieved sentiment distributions (k=20). Lower is better."
        return positioned_grid(6, caption, [
            (234, list(zip((317, 336, 365, 393, 400, 423, 468, 504, 520),
                           "Table 3: Empirical W 1 between corpus-level and retrieved".split(), strict=True))),
            (246, list(zip((320, 355, 402, 445, 477, 496, 514),
                           "sentiment distributions (k=20). Lower is better.".split()))),
            (283, [(336, "Dataset"), (387, "Raw"), (402, "W"), (409, "1"), (441, "Enriched"), (465, "W"),
                   (472, "1"), (498, "∆")]),
            (298, [(335, "Forums"), (394, "8.87"), (448, "7.24"), (498, "−18%")]),
            (308, [(338, "Silverton"), (394, "15.31"), (448, "9.17"), (498, "−40%")]),
            (318, [(344, "Ambassador"), (394, "19.15"), (448, "14.54"), (498, "−24%")]),
            (328, [(328, "Ala"), (349, "Moana"), (394, "12.88"), (448, "6.75"), (final_delta_x, "−48%")]),
        ])

    def test_distance_uses_signed_percent_hotel_row_and_cutoff(self):
        payload = extract_publication(PDF({"Title": "Opinion-Enriched RAG"}, [self.distance_page()]))
        self.assertEqual(len(payload["results"]), 12)
        result, config, dataset, source = result_at(payload, "Table 3", "Ala Moana", "∆")
        self.assertEqual((result["metric_name"], result["raw_value"], result["raw_unit"], result["metric_k"],
                          config["top_k"], dataset["dataset_name"], source["page"]),
                         ("W1 relative change", "−48%", "%", 20, 20, "OpinioBank: Ala Moana Honolulu", 6))
        self.assertEqual(source["bbox"], (496 / 595, 326 / 842, 500 / 595, 330 / 842))

    def test_distance_rejects_delta_that_moves_under_another_header(self):
        self.assertEqual(read_opinion_tables([self.distance_page(final_delta_x=465)], []), [])

    def hotel_page(self, data="+37.9pp"):
        caption = "Table 4: OpinioBank results (public hotel reviews). Entity match improvement and sentiment-intensity σ change vs.\nRaw KB."
        return positioned_grid(7, caption, [
            (77, list(zip((81, 98, 129, 168, 198, 224, 254, 287, 313, 354, 389, 436, 480, 499, 521),
                          caption.splitlines()[0].split(), strict=True))),
            (89, [(80, "Raw"), (100, "KB.")]),
            (114, [(166, "Property"), (256, "Reviews"), (298, "Entities"), (336, "Entity"), (363, "Match"),
                   (382, "∆"), (413, "Sent-Int"), (434, "σ"), (443, "∆")]),
            (129, [(165, "Silverton"), (189, "LV"), (256, "2,297"), (298, "4"), (355, data), (422, "+5.04")]),
            (139, [(171, "Ambassador"), (212, "Bangkok"), (256, "1,230"), (298, "4"), (355, "+37.2pp"),
                   (422, "+7.09")]),
            (149, [(155, "Ala"), (176, "Moana"), (208, "Honolulu"), (256, "2,729"), (298, "4"), (355, "+18.9pp"),
                   (422, "+3.00")]),
        ])

    def test_hotel_absolute_gain_preserves_percentage_points(self):
        payload = extract_publication(PDF({"Title": "Opinion-Enriched RAG"}, [self.hotel_page()]))
        self.assertEqual(len(payload["results"]), 6)
        result, config, dataset, source = result_at(payload, "Table 4", "Silverton LV", "Entity Match ∆")
        self.assertEqual((result["metric_name"], result["raw_value"], result["raw_unit"], result["reported_value"],
                          config["top_k"], dataset["dataset_name"], source["page"]),
                         ("Entity Match Rate absolute gain", "+37.9pp", "pp", 37.9, None,
                          "OpinioBank: Silverton Las Vegas", 7))
        self.assertEqual(source["bbox"], (353 / 595, 127 / 842, 357 / 595, 131 / 842))

    def test_hotel_gain_rejects_a_percent_substitution_for_percentage_points(self):
        self.assertEqual(read_opinion_tables([self.hotel_page(data="+37.9%")], []), [])


class OpinionMethodTest(unittest.TestCase):
    def ablation_page(self, misplaced=False, *, ambiguous=False):
        caption = ("Table 8: Entity-Split-Only ablation at k=20 (Mistral\n"
                   "Large). Bold = highest per metric, * = p<0.001 vs.\nEntity-Split.")
        page = positioned_grid(14, caption, [
            (419, list(zip((317, 335, 380, 435, 458, 477, 508), caption.splitlines()[0].split(), strict=True))),
            (431, list(zip((321, 352, 369, 390, 415, 439, 460, 469, 493, 521), caption.splitlines()[1].split(),
                           strict=True))),
            (443, [(331, "Entity-Split.")]),
            (467, list(zip((317, 338, 396, 416, 433, 456, 488, 515),
                           "KB Config Sent-Int σ Biz Impact Emot Marker".split(), strict=True))),
            (483, [(318, "Raw"), (400, "6.53"), (448, "9.24"), (504, "3.2")]),
            (493, [(331, "Entity-Split"), (363, "Only"), (400, "7.37"), (448, "10.27"), (504, "3.5")]),
            (503, [(317, "Full"), (341, "O-RAG"), (423 if misplaced else 400, "8.11*"), (448, "10.94*"), (504, "4.1*")]),
        ])
        method = "Significance is assessed with the Wilcoxon signed-rank test."
        return page if ambiguous else replace(page, reading_text=f"{method}\n{caption}")

    def retrieval_page(self, *, bad_group=False):
        caption = "Table 11: Semantic-only vs. hybrid retrieval at k=20.\nBold = p<0.05, * = p<0.001."
        rows = [
            (496, list(zip((81, 102, 142, 181, 203, 235, 259, 278), caption.splitlines()[0].split(), strict=True))),
            (508, list(zip((81, 96, 118, 139, 147, 171), caption.splitlines()[1].split(), strict=True))),
            (532, [(171, "Semantic"), (176 if bad_group else 250, "Hybrid")]),
            (544, list(zip((84, 143, 176, 189, 221, 254, 268), "Metric Raw Enr. (∆) Raw Enr. (∆)".split(), strict=True))),
        ]
        for index, (name, values) in enumerate([
            ("Sent-Int σ", "8.93 10.84 (+21.4%*) 9.12 11.56 (+26.8%*)"),
            ("Biz Impact σ", "10.47 11.63 (+11.1%) 10.55 12.08 (+14.5%)"),
            ("Revenue Band (%)", "79.1 68.7 (−13.1%*) 78.3 68.1 (−13.0%*)"),
            ("Tenure (%)", "81.5 76.8 (−5.8%) 81.2 75.6 (−6.9%)"),
            ("Evidence (%)", "44.3 43.8 (−1.1%) 44.3 46.4 (+4.7%)"),
            ("Emot. Markers", "7.1 8.4 (+18.3%) 7.2 8.9 (+23.6%*)"),
        ]):
            rows.append((557 + index * 8, [*zip((78, 94, 110), name.split()),
                                           *zip((143, 165, 191, 221, 243, 270), values.split(), strict=True)]))
        return positioned_grid(16, caption, rows)

    def test_entity_split_grid_keeps_model_separate_from_task_generator(self):
        generation = "Response generation from the retrieved context\nused Anthropic Claude Sonnet 4.5."
        payload = extract_publication(PDF({"Title": "Opinion-Enriched RAG"},
                                          [Page(13, generation, "text_match"), self.ablation_page()]))
        result, config, dataset, source = result_at(payload, "Table 8", "Full O-RAG", "Sent-Int σ")
        self.assertEqual((result["metric_name"], result["raw_value"], result["metric_k"], config["top_k"], config["note"],
                          config["generator_model"], config["generator_version"], dataset["dataset_name"]),
                         ("Sentiment-Intensity sigma", "8.11", 20, 20, "Opinion extraction: Mistral Large",
                          "Anthropic Claude Sonnet", "4.5", "E-commerce seller forums"))
        self.assertEqual(source["bbox"], (398 / 595, 501 / 842, 402 / 595, 505 / 842))

    def test_entity_split_grid_rejects_unaligned_ablation_score(self):
        self.assertEqual(read_opinion_tables([self.ablation_page(misplaced=True)], []), [])

    def test_entity_split_p_bound_uses_star_and_explicit_comparator(self):
        payload = extract_publication(PDF({"Title": "Opinion-Enriched RAG"}, [self.ablation_page()]))
        result, config, dataset, source = result_at(payload, "Table 8", "Full O-RAG", "Sent-Int σ / *")
        self.assertEqual((result["metric_name"], result["metric_variant"], result["raw_value"], result["reported_value"],
                          result["metric_k"], config["note"], dataset["dataset_name"], source["bbox"]),
                         ("p-value bound", "Wilcoxon signed-rank; Sentiment-Intensity sigma; Enriched vs Entity-Split Only",
                          "<0.001", None, 20, "Opinion extraction: Mistral Large", "E-commerce seller forums",
                          (398 / 595, 501 / 842, 402 / 595, 505 / 842)))

    def test_entity_split_star_without_identified_test_leaves_scores_only(self):
        tables = read_opinion_tables([self.ablation_page(ambiguous=True)], [])
        self.assertEqual(sum(len(table.columns) for table in tables), 9)

    def test_retrieval_grid_preserves_group_and_signed_change(self):
        payload = extract_publication(PDF({"Title": "Opinion-Enriched RAG"}, [self.retrieval_page()]))
        self.assertEqual(len(payload["results"]), 36)
        result, config, dataset, source = result_at(payload, "Table 11", "Semantic / Revenue Band (%)", "Semantic / ∆")
        self.assertEqual((result["metric_name"], result["raw_value"], result["raw_unit"], result["metric_k"],
                          config["retrieval_method"], config["top_k"], dataset["dataset_name"]),
                         ("Revenue Band Coverage relative change", "−13.1%", "%", 20, "semantic-only", 20,
                          "E-commerce seller forums"))
        self.assertEqual(source["bbox"], (189 / 595, 571 / 842, 193 / 595, 575 / 842))

    def test_retrieval_grid_rejects_unaligned_group_headings(self):
        self.assertEqual(read_opinion_tables([self.retrieval_page(bad_group=True)], []), [])


class OpinionReferenceDeltaTest(unittest.TestCase):
    model_list = "Mistral Large, Qwen 235B, and Claude Sonnet 4.5."
    appendix_link = ("Cross-model consistency. The Full O-RAG improvement over Raw KB is consistent across extraction "
                     "models: Mistral Large (+24.2% Sent-Int\nσ), Qwen 235B (+27.4%), and Claude Sonnet 4.5\n"
                     "(+26.8%), confirming the approach is not dependent on a specific LLM (detailed in Appendix B).")

    def main_page(self, *, misaligned=False):
        lines = MAIN.splitlines()[:3]
        rows = [
            (77, list(zip((317, 336, 363, 400, 423, 441, 471, 508), lines[0].split(), strict=True))),
            (89, list(zip((321, 355, 396, 443, 476, 505), lines[1].split(), strict=True))),
            (101, list(zip((329, 362, 387, 416, 430, 452, 474, 481, 505), lines[2].split(), strict=True))),
            (126, list(zip((332, 404, 421, 444, 461, 493), "Metric Raw KB Enr. KB ∆".split(), strict=True))),
        ]
        for y, printed, values in (
            (141, "Entity Match (%)", "24.8 35.4 +42.7%*"), (151, "Sent-Int σ", "9.12 11.56 +26.8%*"),
            (161, "Biz Impact σ", "10.55 12.08 +14.5%"), (171, "Emot. Markers", "7.2 8.9 +23.6%*"),
            (187, "Revenue Band (%)", "78.3 68.1 −13.0%*"), (197, "Tenure (%)", "81.2 75.6 −6.9%"),
            (207, "Evidence (%)", "44.3 46.4 +4.7%"),
        ):
            rows.append((y, [*zip((330, 354, 379), printed.split()),
                             *zip((411, 451, 480 if misaligned and y == 207 else 493), values.split(), strict=True)]))
        return positioned_grid(6, MAIN, rows)

    def test_reference_deltas_keep_cell_and_link_star_to_caption_bound(self):
        payload = extract_publication(PDF({"Title": "Opinion-Enriched RAG"}, [self.main_page(), depth_page(repeat=True)]))
        self.assertEqual(sum(bool(row.get("duplicate_of")) for row in payload["results"]), 14)
        result, config, dataset, source = result_at(payload, "Table 2", "Revenue Band (%)", "∆")
        self.assertEqual((result["metric_name"], result["raw_value"], result["raw_unit"], result["reported_value"],
                          config["top_k"], config["retrieval_method"], dataset["dataset_name"], source["bbox"]),
                         ("Revenue Band Coverage relative change", "−13.0%*", "%", -13, 20, "hybrid",
                          "E-commerce seller forums", (491 / 595, 185 / 842, 495 / 595, 189 / 842)))
        bound, _, _, bound_source = result_at(payload, "Table 2", "Revenue Band (%)", "∆ / *")
        self.assertEqual((bound["metric_name"], bound["metric_variant"], bound["raw_value"], bound["reported_value"],
                          bound["sample_size"], bound_source["bbox"]),
                         ("p-value bound", "Wilcoxon signed-rank; Revenue Band Coverage; Enriched vs Raw", "<0.001",
                          None, 48, source["bbox"]))

    def test_reference_delta_overlapping_wrong_column_does_not_change_original_rows(self):
        payload = extract_publication(PDF({"Title": "Opinion-Enriched RAG"}, [self.main_page(misaligned=True),
                                                                              depth_page(repeat=True)]))
        self.assertEqual(sum(bool(row.get("duplicate_of")) for row in payload["results"]), 14)
        self.assertEqual(len(payload["results"]), 28)

    def model_page(self, *, different=False):
        caption = ("Table 5: Retrieval diversity improvements (∆ vs. Raw\n"
                   "KB) across opinion extraction LLMs (k=20, hybrid).")
        rows = [
            (370, list(zip((82, 99, 125, 163, 212, 249, 263, 280), caption.splitlines()[0].split(), strict=True))),
            (382, list(zip((79, 103, 133, 171, 206, 235, 268), caption.splitlines()[1].split(), strict=True))),
            (406, [(97, "Metric"), (178, "Mistral"), (216, "Qwen"), (258, "Sonnet")]),
        ]
        for i, (label, values) in enumerate([
            ("Sent-Int σ", "+24.2% +27.4% +26.8%*"), ("Biz Impact σ", "+18.4% +12.4% +14.5%"),
            ("Revenue Band Cov.", "−17.5% −19.0% −13.0%*"), ("Tenure Cov.", "−3.9% −3.6% −6.9%"),
            ("Evidence Cov.", "+9.4% +8.9% +4.8%" if different else "+9.4% +8.9% +4.7%"),
            ("Emot. Markers", "+28.1% +20.0% +23.6%*"),
        ]):
            rows.append((422 + 10 * i, [*zip((91, 116, 143), label.split()),
                                        *zip((178, 216, 258), values.split(), strict=True)]))
        return positioned_grid(14, caption, rows)

    def test_extraction_models_keep_signs_and_link_only_the_complete_sonnet_repeat(self):
        payload = extract_publication(PDF({"Title": "Opinion-Enriched RAG"},
                                          [self.main_page(), Page(7, self.appendix_link, "text_match"),
                                           Page(13, self.model_list, "text_match"), self.model_page()]))
        result, config, dataset, source = result_at(payload, "Table 5", "Revenue Band Cov.", "Mistral")
        self.assertEqual((result["metric_name"], result["raw_value"], result["raw_unit"], result["metric_k"],
                          config["note"], config["retrieval_method"], dataset["dataset_name"], source["bbox"]),
                         ("Revenue Band Coverage relative change", "−17.5%", "%", 20, "Opinion extraction: Mistral Large",
                          "hybrid", "E-commerce seller forums", (176 / 595, 440 / 842, 180 / 595, 444 / 842)))
        model_source = next(e for e in payload["evidence"] if e["entity_id"] == config["config_id"] and e["field"] == "note")
        self.assertEqual((model_source["page"], model_source["fragment"], model_source["verification"]),
                         (13, "Mistral Large", "text_match"))
        original, _, _, _ = result_at(payload, "Table 2", "Revenue Band (%)", "∆")
        repeated, _, _, _ = result_at(payload, "Table 5", "Revenue Band Cov.", "Sonnet")
        self.assertEqual(repeated["duplicate_of"], original["result_id"])
        self.assertEqual(sum(bool(row.get("duplicate_of")) for row in payload["results"]), 6)

    def test_extraction_models_keep_a_changed_sonnet_grid_separate(self):
        payload = extract_publication(PDF({"Title": "Opinion-Enriched RAG"}, [self.main_page(),
                        Page(7, self.appendix_link, "text_match"), Page(13, self.model_list, "text_match"),
                        self.model_page(different=True)]))
        changed, _, _, _ = result_at(payload, "Table 5", "Evidence Cov.", "Sonnet")
        self.assertEqual(changed["raw_value"], "+4.8%")
        self.assertFalse(any(row.get("duplicate_of") for row in payload["results"]))

    def test_extraction_models_keep_coincident_cells_without_documented_linkage(self):
        payload = extract_publication(PDF({"Title": "Opinion-Enriched RAG"},
                                          [self.main_page(), Page(13, self.model_list, "text_match"), self.model_page()]))
        sonnet, _, data, _ = result_at(payload, "Table 5", "Sent-Int σ", "Sonnet")
        self.assertEqual((sonnet["raw_value"], sonnet.get("duplicate_of"), data["dataset_name"]), ("+26.8%*", None, None))


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
