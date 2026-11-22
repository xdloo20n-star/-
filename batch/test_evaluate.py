import contextlib
import csv
import io
import json
import unittest
from unittest.mock import patch

from batch import evaluate as benchmark
from batch.result_validation import DUPLICATE_FIELDS


RESULT_COLUMNS = tuple(dict.fromkeys((*benchmark.RESULT_FIELDS, *DUPLICATE_FIELDS)))


def publication(publication_id, in_scope="true", reason=""):
    return {"publication_id": publication_id, "in_scope": in_scope, "exclusion_reason": reason}


def result(result_id, publication_id="D001", **overrides):
    return dict.fromkeys(RESULT_COLUMNS, "") | {
        "publication_id": publication_id,
        "result_id": result_id,
        "config_id": "config",
        "dataset_id": "dataset",
        "extraction_status": "extracted",
        "raw_value": "0.5",
        "metric_name": "Recall",
        "metric_value": "0.5",
        "source_page": "4",
        "source_table": "Table 2",
    } | overrides


def fixtures(publications, results, *, configurations=None, datasets=None):
    tables = {
        "publications": (("publication_id", "in_scope", "exclusion_reason"), publications),
        "results": (("publication_id", *RESULT_COLUMNS), results),
    }
    if configurations is not None:
        tables["configurations"] = (("publication_id", "config_id", *benchmark.CONFIG_FIELDS), configurations)
    if datasets is not None:
        tables["datasets"] = (("publication_id", "dataset_id", *benchmark.DATASET_FIELDS), datasets)

    def open_csv(table):
        fields, rows = tables[table]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.seek(0)
        return stream

    return benchmark.read_corpus(open_csv)


class SourceKeyTests(unittest.TestCase):
    def test_narrow_format_normalization(self):
        g = result("gold", raw_value="−.500 %", metric_name="Recall@05", metric_k="5",
                   source_page="04", source_table="TABLE II")
        c = result("candidate", raw_value="-0.5%", metric_name="recall", metric_k="05",
                   source_table="Tab. ii")
        self.assertIsNotNone(benchmark.source_key(g))
        self.assertEqual(benchmark.source_key(g), benchmark.source_key(c))
        self.assertEqual(benchmark.source_key(result("a", metric_name="ROUGE-1")),
                         benchmark.source_key(result("b", metric_name="ROUGE_1")))

    def test_percent_in_cell_and_percent_in_column_unit_have_same_source_value(self):
        cell = result("cell", raw_value="24.8%", raw_unit="%")
        headed = result("headed", raw_value="24.8", raw_unit="%")
        self.assertEqual(benchmark.source_key(cell), benchmark.source_key(headed))

    def test_preserves_sign_scale_cutoff_variant_and_source_type(self):
        for first, second in (
            ({"raw_value": "-0.0"}, {"raw_value": "0"}),
            ({"raw_value": "+0.5"}, {"raw_value": "0.5"}),
            ({"raw_value": "50%"}, {"raw_value": "0.50"}),
            ({"raw_value": "~0.5"}, {"raw_value": "0.50"}),
            ({"raw_value": "0.51"}, {"raw_value": "0.50"}),
            ({"raw_unit": "%"}, {"raw_unit": "fraction"}),
            ({"raw_unit": "mM"}, {"raw_unit": "mm"}),
            ({"metric_k": "0"}, {"metric_k": ""}),
            ({"metric_name": "Recall@5"}, {"metric_name": "Recall@10"}),
            ({"aggregation": "mean"}, {"aggregation": "std"}),
            ({"aggregation": "mean"}, {}),
            ({"metric_variant": "micro"}, {"metric_variant": "macro"}),
            ({"source_page": "9"}, {"source_page": "10"}),
            ({"source_table": "Table 2"}, {"source_table": "", "source_figure": "Fig. 2"}),
            ({"source_table": "Table II"}, {"source_table": "Table 2"}),
            ({"source_table": "", "source_figure": "Figure 2(a)"},
             {"source_table": "", "source_figure": "Figure 2"}),
        ):
            with self.subTest(first=first, second=second):
                first_key = benchmark.source_key(result("a", **first))
                second_key = benchmark.source_key(result("b", **second))
                self.assertIsNotNone(first_key)
                self.assertIsNotNone(second_key)
                self.assertNotEqual(first_key, second_key)

    def test_missing_fields_and_conflicting_cutoff_are_unmatchable(self):
        for overrides in ({"raw_value": ""}, {"raw_value": "NULL"}, {"raw_value": "—"},
                          {"metric_name": ""}, {"metric_name": "Recall@5", "metric_k": "10"},
                          {"source_page": ""}, {"source_page": "0"}, {"source_page": "4-5"},
                          {"source_table": ""}, {"source_table": "Table"}, {"source_table": "null"}):
            with self.subTest(overrides=overrides):
                self.assertIsNone(benchmark.source_key(result("a", **overrides)))
        self.assertIsNotNone(benchmark.source_key(result("zero", raw_value="0")))


class BenchmarkTests(unittest.TestCase):
    def test_one_to_one_count_with_missing_paper_and_table_scope(self):
        gold_publications = [publication("D001"), publication("D002"), publication("D003")]
        gold_results = [result("r1"), result("r2"), result("r3", source_table="", source_figure="Figure 1"),
                        result("r4", "D002"), result("r5", "D002"), result("r6", "D002"),
                        result("r7", "D003", source_table="", source_figure="Figure 3")]
        gold = fixtures(gold_publications, gold_results)
        candidate_publications = [publication("D001"), publication("P001")]
        candidate_results = [result("c1"), result("c2"), result("c3"), result("c4", "P001")]
        report = benchmark.evaluate(gold, fixtures(candidate_publications, candidate_results))
        self.assertEqual(report, benchmark.evaluate(fixtures(gold_publications, reversed(gold_results)),
                                                    fixtures(candidate_publications, reversed(candidate_results))))
        coverage = report["corpus"]["coverage"]
        self.assertEqual(coverage["pdf_coverage"], 1 / 3)
        self.assertEqual(coverage["missing_pdfs"], ["D002", "D003"])
        self.assertEqual(coverage["outside_gold_pdfs"], ["P001"])
        all_results = report["corpus"]["scopes"]["all_results"]
        table_only = report["corpus"]["scopes"]["table_only"]
        self.assertEqual(all_results["candidate_outside_gold"]["extracted"], 1)
        self.assertIsNone(all_results["context_agreement"])
        self.assertEqual(all_results["source_agreement"]["micro"], benchmark.agreement(2, 7, 3))
        self.assertEqual(table_only["source_agreement"]["micro"], benchmark.agreement(2, 5, 3))
        self.assertEqual(all_results["source_agreement"]["macro"], {
            "papers": 3, "precision": (2 / 3) / 3, "recall": (2 / 3) / 3, "f1": (4 / 6) / 3,
        })
        self.assertEqual(table_only["source_agreement"]["macro"], {
            "papers": 2, "precision": (2 / 3) / 2, "recall": 1 / 2, "f1": (4 / 5) / 2,
        })
        missing_paper = report["documents"]["D002"]
        self.assertEqual(missing_paper["candidate"]["decision"], "missing")
        self.assertIsNone(missing_paper["scopes"]["table_only"]["source_agreement"]["precision"])
        self.assertIsNone(report["documents"]["P001"]["scopes"]["all_results"]["source_agreement"])

    def test_counts_status_duplicates_exclusions_and_unkeyable_rows(self):
        gold = fixtures([publication("D001"), publication("D002"), publication("D003", "false", "Out of scope")], [
            result("r1"), result("r2", duplicate_of="r1"), result("r3", extraction_status="absent", raw_value="0"),
            result("r4", extraction_status="present_not_extractable"), result("r5", "D002"),
        ])
        candidate = fixtures([publication("D001"), publication("D002", "false", "Excluded"), publication("D003")], [
            result("c1"), result("c2", duplicate_of="c1"), result("c3", source_page=""),
            result("c4", "D002"), result("c5", "D003"),
        ])
        report = benchmark.evaluate(gold, candidate)
        self.assertEqual(report["corpus"]["coverage"]["scope_disagreement_pdfs"], ["D002", "D003"])
        self.assertEqual(report["documents"]["D002"]["candidate"], {"decision": "excluded", "exclusion_reason": "Excluded"})
        scope = report["corpus"]["scopes"]["all_results"]
        self.assertEqual(scope["gold"], {
            "rows": 5, "extracted": 3, "absent": 1, "present_not_extractable": 1, "marked_duplicate": 1,
            "excluded_rows": 0, "eligible": 2, "source_keyed": 2, "context_keyed": 0,
        })
        self.assertEqual(scope["candidate"], {
            "rows": 5, "extracted": 5, "absent": 0, "present_not_extractable": 0, "marked_duplicate": 1,
            "excluded_rows": 1, "eligible": 3, "source_keyed": 2, "context_keyed": 0,
        })
        self.assertEqual(scope["source_agreement"]["micro"], benchmark.agreement(1, 2, 3))

    def test_figure_match_contributes_only_to_all_results(self):
        publications = [publication("D001")]
        gold = fixtures(publications, [result("r1", source_table="", source_figure="Figure 2(a)")])
        candidate = fixtures(publications, [result("c1", source_table="", source_figure="Fig. 2(a)")])
        scopes = benchmark.evaluate(gold, candidate)["corpus"]["scopes"]
        self.assertEqual(scopes["all_results"]["source_agreement"]["micro"], benchmark.agreement(1, 1, 1))
        self.assertEqual(scopes["table_only"]["source_agreement"]["micro"], benchmark.agreement(0, 0, 0))
        self.assertEqual(scopes["table_only"]["source_agreement"]["macro"], {
            "papers": 0, "precision": None, "recall": None, "f1": None,
        })

    def test_context_uses_local_joins_and_requires_field_agreement(self):
        pub = publication("D001")
        gold = fixtures([pub], [result("r1"), result("r2", raw_value="0.7")], configurations=[{
            "publication_id": "D001", "config_id": "config", "rag_variant": "RAG A", "generator_model": "Model 1",
        }], datasets=[{"publication_id": "D001", "dataset_id": "dataset", "dataset_name": "Corpus 1", "data_split": "test"}])
        candidate = fixtures([pub, publication("D002")], [
            result("c1", config_id="different-c", dataset_id="different-d"),
            result("c2", raw_value="0.7", config_id="config", dataset_id="different-d"),
        ], configurations=[
            {"publication_id": "D001", "config_id": "different-c", "rag_variant": "rag a", "generator_model": "model 1"},
            {"publication_id": "D001", "config_id": "config", "rag_variant": "rag a", "generator_model": "model 2"},
            {"publication_id": "D002", "config_id": "config", "rag_variant": "rag a", "generator_model": "model 1"},
        ], datasets=[{"publication_id": "D001", "dataset_id": "different-d", "dataset_name": "corpus 1", "data_split": "test"}])
        scope = benchmark.evaluate(gold, candidate)["corpus"]["scopes"]["table_only"]
        self.assertEqual(scope["source_agreement"]["micro"], benchmark.agreement(2, 2, 2))
        self.assertEqual(scope["context_agreement"]["micro"], benchmark.agreement(1, 2, 2))

    def test_invalid_duplicate_reference_cannot_remove_candidate_from_score(self):
        with self.assertRaisesRegex(ValueError, "D001/r1: duplicate_of target 'r2' does not exist"):
            fixtures([publication("D001"), publication("D002")], [
                result("r1", duplicate_of="r2"), result("r2", "D002"),
            ])

    def test_invalid_duplicate_conditions_and_values_are_rejected(self):
        for field in DUPLICATE_FIELDS:
            with self.subTest(field=field):
                mismatch = "0.56" if field == "metric_value" else "different"
                with self.assertRaisesRegex(ValueError, f"D001/r2: duplicate_of differs in {field}"):
                    fixtures([publication("D001")], [
                        result("r1"), result("r2", duplicate_of="r1", **{field: mismatch}),
                    ])
        for rows in (
            [result("r1", extraction_status="absent"), result("r2", duplicate_of="r1")],
            [result("r1"), result("r2", duplicate_of="r1", extraction_status="present_not_extractable")],
        ):
            with self.assertRaisesRegex(ValueError, "D001/r2: duplicate_of requires two extracted rows"):
                fixtures([publication("D001")], rows)

    def test_duplicate_cycle_is_rejected(self):
        for rows in (
            [result("r1", duplicate_of="r1")],
            [result("r1", duplicate_of="r2"), result("r2", duplicate_of="r1")],
        ):
            with self.assertRaisesRegex(ValueError, "D001/r1: duplicate_of cycle"):
                fixtures([publication("D001")], rows)

    def test_duplicate_chain_compares_each_link_at_reported_precision(self):
        publications = [publication("D001")]
        gold = fixtures(publications, [result("r1")])
        candidate = fixtures(publications, [
            result("leaf", duplicate_of=" middle ", metric_value="0.449", raw_value=".449"),
            result("middle", duplicate_of="root", metric_value="0.45", raw_value=".45"),
            result("root"),
        ])
        scope = benchmark.evaluate(gold, candidate)["corpus"]["scopes"]["all_results"]
        self.assertEqual(2, scope["candidate"]["marked_duplicate"])
        self.assertEqual(1, scope["candidate"]["eligible"])
        self.assertEqual(benchmark.agreement(1, 1, 1), scope["source_agreement"]["micro"])

    def test_diagnostics_keep_exact_matches_and_count_each_projection_one_to_one(self):
        publications = [publication("D001"), publication("D002")]
        gold = fixtures(publications, [
            result("r1", aggregation="mean"), result("r2", aggregation="std"),
            result("r3", aggregation="std"),
            result("r4", "D002", raw_value="0.7", source_table="Table 8"),
            result("r5", raw_value="0.8", raw_unit="%", metric_variant="micro"),
        ])
        candidate = fixtures(publications + [publication("P001")], [
            result("c1", aggregation="mean"), result("c2", aggregation="median"),
            result("c3", "D002", raw_value="0.7", source_table="Table 9"),
            result("c4", raw_value="0.8", raw_unit="fraction", metric_variant="macro"),
            result("c5", "P001", aggregation="std"),
        ])
        strict = benchmark.evaluate(gold, candidate)
        report = benchmark.evaluate(gold, candidate, diagnostics=True)
        for name, scope in report["corpus"]["scopes"].items():
            self.assertEqual(strict["corpus"]["scopes"][name]["source_agreement"], scope["source_agreement"])
            gains = dict.fromkeys(benchmark.DIAGNOSTIC_FIELDS, 0) | {"aggregation": 1, "source_table": 1}
            self.assertEqual(gains, scope["additional_matches_by_field"])
        scopes = {pdf: item["scopes"]["table_only"] for pdf, item in report["documents"].items()}
        self.assertEqual(1, scopes["D001"]["additional_matches_by_field"]["aggregation"])
        self.assertEqual(1, scopes["D002"]["additional_matches_by_field"]["source_table"])
        self.assertIsNone(scopes["P001"]["additional_matches_by_field"])

    def test_diagnostics_can_project_away_one_missing_required_field(self):
        publications = [publication("D001")]
        gold = fixtures(publications, [
            result("match", raw_value="0.4"),
            result("table"), result("metric", raw_value="0.6"),
            result("page", raw_value="0.7"), result("two_missing", raw_value="0.8"),
        ])
        candidate = fixtures(publications, [
            result("match", raw_value="0.4"), result("match_extra", raw_value="0.4", source_table=""),
            result("missing_table", source_table=""), result("missing_table_extra", source_table=""),
            result("missing_metric", raw_value="0.6", metric_name=""),
            result("missing_page", raw_value="0.7", source_page=""),
            result("two_missing", raw_value="0.8", source_page="", source_table=""),
        ])
        report = benchmark.evaluate(gold, candidate, diagnostics=True)
        self.assertTrue(report["scoring"].pop("additional_matches_by_field"))
        for scopes in (report["corpus"]["scopes"], report["documents"]["D001"]["scopes"]):
            for scope_name, scope in scopes.items():
                gains = scope.pop("additional_matches_by_field")
                expected = dict.fromkeys(benchmark.DIAGNOSTIC_FIELDS, 0) | {
                    "metric_name": 1, "source_page": 1, "source_table": int(scope_name == "all_results"),
                }
                self.assertEqual(expected, gains)
        self.assertEqual(benchmark.evaluate(gold, candidate), report)

    def test_cli_json_stdout(self):
        gold = fixtures([publication("D001")], [result("r1")])
        candidate = fixtures([publication("D001")], [])
        with (patch.object(benchmark, "read_gold_zip", return_value=gold),
              patch.object(benchmark, "read_directory", return_value=candidate),
              contextlib.redirect_stdout(io.StringIO()) as stdout):
            self.assertEqual(benchmark.main(["candidate", "--corpus-zip", "archive.zip", "--diagnostics",
                                             "--json-report", "-"]), 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["corpus"]["coverage"]["pdf_coverage"], 1)
        self.assertEqual(payload["corpus"]["scopes"]["all_results"]["source_agreement"]["micro"],
                         benchmark.agreement(0, 1, 0))
        self.assertEqual(dict.fromkeys(benchmark.DIAGNOSTIC_FIELDS, 0),
                         payload["corpus"]["scopes"]["all_results"]["additional_matches_by_field"])


if __name__ == "__main__":
    unittest.main()
