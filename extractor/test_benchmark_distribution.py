import unittest

from pdf_pages import PDF, Page
from publication import extract_publication


LAYOUT = """                                   Enterprise                  Domain                         Files    Queries
                                   Aventro Motors              Automotive                       51        200
                                   Cendara University          Academia & Education             41        186
                                   Velvera Technologies        Enterprise Technology            39        200
                                   ZX Bank                     Banking & Finance                72        200
                                   Total                       —                               203        786
                Table 2: Composition of the RAG-Multi-Corpus benchmark across enterprises and domains.


                          Category          Count      Percentage        Description
                          Descriptive         138          17.6%         Factual descriptions or definitions
                          Analytical          122          15.5%         Analysis, interpretation, or inference
                          Comparative         139          17.7%         Comparison between entities or concepts
                          Boolean             108          13.7%         Yes/no factual questions
                          Temporal             24           3.1%         Time-based or sequence-oriented questions
                          Procedural          180          22.9%         Process-oriented or how-to questions
                          Open-Ended           75           9.5%         Multi-hop synthesis across sources
                          Total               786         100.0%         —
                        Table 3: Distribution of query categories in the RAG-Multi-Corpus benchmark.
"""


class BenchmarkDistributionTest(unittest.TestCase):
    def extract(self, text):
        return extract_publication(PDF({}, [Page(1, "Retrieval-Augmented Generation benchmark", "text_match"),
                                           Page(4, text, "text_match")]))

    def test_explicit_composition_counts_and_query_percentages_keep_their_scopes(self):
        payload = self.extract(LAYOUT)
        self.assertEqual(len(payload["results"]), 26)
        rows = {r["result_id"]: r for r in payload["results"]}
        datasets = {d["dataset_id"]: d["dataset_name"] for d in payload["datasets"]}
        configs = {c["config_id"]: c for c in payload["configurations"]}

        def locate(table, row, column):
            source, = [e for e in payload["evidence"] if e.get("field") == "raw_value"
                       and (e["table"], e["table_row"], e["table_column"]) == (table, row, column)]
            self.assertEqual((source["page"], source["verification"]), (4, "text_match"))
            return rows[source["entity_id"]]

        documents = locate("Table 2", "Aventro Motors", "Files")
        self.assertEqual((documents["metric_name"], documents["raw_value"], documents["raw_unit"],
                          datasets[documents["dataset_id"]]),
                         ("Documents", "51", None, "RAG-Multi-Corpus / Aventro Motors"))
        self.assertIsNone(configs[documents["config_id"]]["rag_variant"])
        share = locate("Table 3", "Temporal", "Percentage")
        self.assertEqual((share["metric_name"], share["raw_value"], share["raw_unit"], share["sample_size"],
                          datasets[share["dataset_id"]]),
                         ("Query proportion", "3.1%", "%", 786, "RAG-Multi-Corpus / Temporal"))
        self.assertEqual(locate("Table 3", "Total", "Count")["duplicate_of"],
                         locate("Table 2", "Total", "Queries")["result_id"])

    def test_rejects_categories_that_disagree_with_the_printed_total(self):
        payload = self.extract(LAYOUT.replace("Procedural          180", "Procedural          182"))
        self.assertEqual(len(payload["results"]), 10)
        self.assertFalse(any(evidence.get("table") == "Table 3" for evidence in payload["evidence"]))
