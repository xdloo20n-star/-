import unittest

from pdf_pages import PDF, Page
from publication import Payload, add_table, extract_publication
from rag_tables import Column, Header, Table


class MetadataTest(unittest.TestCase):
    def test_native_front_matter_with_exact_evidence(self):
        title = "Retrieval Augmented Generation for\n     Question Answering"
        byline = ["Ana González1,2 , Min Chen2,∗", "Jo van Dijk3 and J. Doe3"]
        footer = "June 3–6, 2025 ©2025 Example Association"
        page = Page(
            1,
            "\n".join([title, *byline, "1", "Example University", "Abstract",
                       "Proceedings of the Example Conference, pages 1–5", footer]),
            "text_match",
        )
        payload = extract_publication(PDF({"CreationDate": "2026"}, [page]))
        publication = payload["publication"]
        self.assertEqual(publication["title"], "Retrieval Augmented Generation for Question Answering")
        self.assertEqual(publication["authors"], ["Ana González", "Min Chen", "Jo van Dijk", "J. Doe"])
        self.assertEqual(publication["year"], 2025)
        self.assertEqual([(item["field"], item["fragment"]) for item in payload["evidence"]],
                         [("title", title), ("authors", byline[0]), ("authors", byline[1]), ("year", footer)])

    def test_ocr_id_is_not_a_title(self):
        layout = "P 0 O 9\n\nRAG for Scientific Question Answering\n\nAbstract"
        page = Page(1, layout, "ocr_match")
        payload = extract_publication(PDF({"Title": "P 0 O 9.pdf"}, [page]))
        self.assertEqual(payload["publication"]["title"], "RAG for Scientific Question Answering")
        self.assertEqual(payload["evidence"][0]["verification"], "ocr_match")
        placeholder = extract_publication(PDF({"Title": "P009"}, [Page(1, "P 0 0 9", "ocr_match")]))
        self.assertIsNone(placeholder["publication"]["title"])

    def test_ambiguous_byline_retains_only_the_title(self):
        page = Page(1, "RAG for Scientific Question Answering\n"
                       "Ana González1, Min Chen1\n"
                       "Gao Liu Jing Wang John Smith\n"
                       "Example University\nAbstract", "text_match")
        publication = extract_publication(PDF({}, [page]))["publication"]
        self.assertEqual(publication["title"], "RAG for Scientific Question Answering")
        self.assertIsNone(publication["authors"])


class TableContextTest(unittest.TestCase):
    def test_wrapped_evaluation_definition_keeps_raw_score_and_unknown_scale(self):
        caption = "Table 1: Comparison on the Natural Questions test set."
        definition = "During evaluation, we use normalized exact\nmatch (EM) and word-level F1-score."
        page = Page(5, f"{caption}\nDirectGen                      22.27    17.11",
                    "text_match", f"{caption}\n\n{definition}")
        columns = [Column(Header(name, name, 0, None, None), "Natural Questions") for name in ("EM", "F1")]
        table = Table(page, "Table 1", caption, 0, columns, [("DirectGen", ["22.27", "17.11"])])
        payload = Payload({}, {})
        add_table(payload, table, True)
        em, f1 = payload.results
        self.assertEqual((em["raw_value"], em["reported_value"], em["reported_scale"]), ("22.27", 22.27, None))
        self.assertEqual((em["metric_variant"], f1["metric_variant"]), ("normalized exact match", "word-level"))
        self.assertEqual(payload.datasets[0]["data_split"], "test")
        self.assertIn(("data_split", caption), [(e["field"], e["fragment"]) for e in payload.evidence])
        self.assertIn(("metric_variant", definition), [(e["field"], e["fragment"]) for e in payload.evidence])

    def test_text_normalization_definition_is_attached_to_matching_metrics(self):
        caption = "Table 1: Main results on the HotpotQA test set."
        definition = ("For evaluation, we report Exact Match (EM) and F1 scores to measure absolute correctness "
                      "and token-level overlap, applying standard text normalization to both predictions and gold answers.")
        page = Page(6, f"{caption}\nMethod       EM     F1\nSystem       .42    .53", "text_match", definition)
        columns = [Column(Header(name, name, center, None, None), "HotpotQA")
                   for name, center in (("EM", 14), ("F1", 21))]
        table = Table(page, "Table 1", caption, 0, columns, [("System", [".42", ".53"])])
        payload = Payload({}, {})
        add_table(payload, table, True)
        self.assertEqual([result["metric_variant"] for result in payload.results], [
            "exact match after standard text normalization",
            "token-level overlap after standard text normalization",
        ])
        self.assertEqual([e["fragment"] for e in payload.evidence if e["field"] == "metric_variant"],
                         [definition, definition])

    def test_explicit_table_cutoff_is_attached_to_each_result(self):
        caption = "Table 2: Diversity metrics at k=20 (hybrid retrieval, forums dataset)."
        page = Page(6, f"{caption}\nRaw KB       24.8\nEnriched KB   35.4", "text_match")
        table = Table(page, "Table 2", caption, 0,
                      [Column(Header("Entity Match", "Entity Match", 0, None, "%"), "Forums")],
                      [("Raw KB", ["24.8"]), ("Enriched KB", ["35.4"])])
        payload = Payload({}, {})
        add_table(payload, table, True)
        self.assertEqual([result["metric_k"] for result in payload.results], [20, 20])
        self.assertEqual([result["metric_k_applicability"] for result in payload.results], ["applicable"] * 2)

    def test_abbreviated_metrics_expand_only_when_the_page_defines_them(self):
        caption = "Table 2: Diversity metrics at k=20 (hybrid retrieval, forums dataset)."
        definitions = ("Entity Match Rate: fraction of retrieved documents that mention the queried entity. "
                       "Sentiment-Intensity σ and Business Impact σ measure diversity. "
                       "Emotional Markers count distinct markers. Demographic Coverage includes Revenue Band, "
                       "Tenure and Evidence Type.")
        page = Page(6, f"{caption}\nRaw KB 24.8 9.12 10.55 7.2 78.3 81.2 44.3",
                    "text_match", f"{definitions}\n{caption}")
        labels = ["Entity Match", "Sent-Int σ", "Biz Impact σ", "Emot. Markers",
                  "Revenue Band", "Tenure", "Evidence"]
        columns = [Column(Header(label, label, 10 + 10 * i, None, None), "Forums")
                   for i, label in enumerate(labels)]
        table = Table(page, "Table 2", caption, 0, columns,
                      [("Raw KB", ["24.8", "9.12", "10.55", "7.2", "78.3", "81.2", "44.3"])])
        payload = Payload({}, {})
        add_table(payload, table, True)
        self.assertEqual([result["metric_name"] for result in payload.results], [
            "Entity Match Rate", "Sentiment-Intensity sigma", "Business Impact sigma",
            "Emotional Markers", "Revenue Band Coverage", "Tenure Coverage", "Evidence Type Coverage",
        ])
        self.assertTrue(all(result["metric_k"] == 20 for result in payload.results))
        self.assertEqual(sum(e["field"] == "metric_name" for e in payload.evidence), 7)

    def test_named_table_context_does_not_transfer_to_table_with_same_metric(self):
        captions = ["Table 1: Corpus comparison.", "Table 10: Corpus comparison."]
        evaluation = "In Table 1, macro-averaged F1 is reported as a percentage and evaluated using RAGAS."
        general = "For evaluation, we report word-level F1."
        layout = "\n\n".join([*captions, evaluation, general, "Method A    41.5"])
        page = Page(2, layout, "text_match")
        column = Column(Header("F1", "F1", 0, None, None), None)
        tables = [
            Table(page, label, caption, index, [column], [("Method A", ["41.5"])])
            for index, (label, caption) in enumerate(zip(("Table 1", "Table 10"), captions))
        ]
        payload = Payload({}, {})
        for table in tables:
            add_table(payload, table, True, tables)
        first, second = payload.results
        self.assertEqual((first["metric_variant"], first["raw_unit"], first["reported_scale"]),
                         ("macro", "percent", "percent"))
        self.assertEqual((second["metric_variant"], second["raw_unit"], second["reported_scale"]),
                         (None, None, None))
        self.assertEqual([row["evaluation_protocol"] for row in payload.experiments], ["RAGAS", None])
        for field in ("metric_variant", "raw_unit", "evaluation_protocol"):
            evidence = [row for row in payload.evidence if row["field"] == field]
            self.assertEqual(len(evidence), 1)
            self.assertEqual((evidence[0]["table"], evidence[0]["page"]), ("Table 1", 2))
            self.assertIn(evidence[0]["fragment"], page.text)


if __name__ == "__main__":
    unittest.main()
