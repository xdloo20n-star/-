import csv
from pathlib import Path
import tempfile
import unittest

from batch.submission import write_submission
from batch.test_batch import TABLES, detail


def summary(result_id, value=0.5, **fields):
    return {
        "extraction_mode": "python",
        "normalized": {"value": value, "unit": "fraction"},
        "result": {
            "result_id": result_id, "extraction_status": "extracted", "result_origin": "primary",
            "metric_name": "Recall", "raw_value": str(value),
        } | fields,
    }


def evidence(evidence_id, result_id, **fields):
    return {
        "evidence_id": evidence_id, "entity_type": "result", "entity_id": result_id,
        "field": "raw_value", "page": 2,
    } | fields


class SubmissionValidationTests(unittest.TestCase):
    def export(self, results, sources):
        document = detail(included=True)
        document["data"].update(results=results, evidence=sources)
        return write_submission([document], TABLES, self.output)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "submission"

    def test_cited_result_requires_citation_identifier(self):
        for citation in (None, "", "  "):
            with self.subTest(citation=citation):
                with self.assertRaisesRegex(ValueError, "r1.*cited_publication_id"):
                    self.export([summary("r1", result_origin="cited", cited_publication_id=citation)],
                                [evidence("ev", "r1", fragment="Recall 0.5")])
                self.assertFalse(self.output.exists())

    def test_primary_result_cannot_cite_another_publication(self):
        with self.assertRaisesRegex(ValueError, "r1.*primary result has cited_publication_id"):
            self.export([summary("r1", cited_publication_id="[3]")],
                        [evidence("ev", "r1", table="Table 1")])
        self.assertFalse(self.output.exists())

    def test_extracted_result_requires_representable_source_locator(self):
        for sources in (
            [],
            [evidence("ev", "r1", bbox=[1, 2, 3, 4], block_id="block")],
            [evidence("ev", "r1", fragment="  ", figure="\t", table="\n")],
            [evidence("ev", "r1", page=None, table="Table 1")],
            [evidence("ev", "r1", page=-2, table="Table 1")],
            [evidence("ev", "r1", page="00", table="Table 1")],
            [evidence("ev", "r1", page="2.5", table="Table 1")],
            [evidence("ev", "r1", page=3, section="Results")],
        ):
            with self.subTest(sources=sources):
                with self.assertRaisesRegex(ValueError, "r1.*extracted result without a CSV source locator"):
                    self.export([summary("r1")], sources)
                self.assertFalse(self.output.exists())

    def test_representable_whole_result_evidence_beats_raw_value_bbox(self):
        self.export([summary("r1")], [
            evidence("bbox", "r1", bbox=[1, 2, 3, 4]),
            evidence("whole", "r1", field=None, table="Table 1"),
        ])
        with (self.output / "results.csv").open(encoding="utf-8", newline="") as stream:
            [row] = list(csv.DictReader(stream))
        self.assertEqual("Table 1", row["source_table"])

    def test_checks_projected_duplicate_values_and_conditions_before_writing(self):
        sources = [evidence("ev1", "r1", table="Table 1"), evidence("ev2", "r2", table="Table 1")]
        for field, row in (
            ("metric_value", summary("r2", 0.56, duplicate_of="r1", raw_value="0.5")),
            ("config_id", summary("r2", duplicate_of="r1", config_id="another")),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, f"r2: duplicate_of differs in {field}"):
                    self.export([summary("r1"), row], sources)
                self.assertFalse(self.output.exists())

    def test_exports_acyclic_duplicate_chain_with_reported_rounding(self):
        results = [summary("r1"), summary("r2", 0.45, duplicate_of="r1"),
                   summary("r3", 0.449, duplicate_of="r2")]
        sources = [evidence(f"ev_{key}", key, table="Table 1") for key in ("r1", "r2", "r3")]
        self.assertEqual(3, self.export(results, sources)["results"])
        with (self.output / "results.csv").open(encoding="utf-8", newline="") as stream:
            self.assertEqual(
                [("", "0.5"), ("r1", "0.45"), ("r2", "0.449")],
                [(row["duplicate_of"], row["metric_value"]) for row in csv.DictReader(stream)],
            )
