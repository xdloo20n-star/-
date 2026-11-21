import contextlib
import csv
import io
import json
import tempfile
import threading
import unittest
import zipfile
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from batch.__main__ import main
from batch.submission import SCHEMA_MEMBER


# A small synthetic schema exercises the projection and header order.
TABLES = {
    "publications": "title publication_id DOI authors in_scope year exclusion_reason",
    "configurations": "config_id publication_id top_k chunk_overlap",
    "datasets": "publication_id dataset_id dataset_name",
    "tasks": "publication_id task_id task_type",
    "experiments": "publication_id experiment_id latency_ms",
    "results": ("publication_id result_id experiment_id config_id dataset_id task_id duplicate_of metric_name "
                "metric_k raw_value raw_unit metric_value metric_unit normalized_value normalized_unit baseline_value "
                "source_page source_section source_table source_figure source_fragment note result_origin "
                "cited_publication_id extraction_status"),
    "issues": "publication_id issue_id record_ids check_name description source_page source_fragment",
}
TABLES = {name: fields.split() for name, fields in TABLES.items()}


def detail(corpus_id="D001", *, document_id=None, mode="python", status="completed", included=False):
    document = {
        "corpus_id": corpus_id, "document_id": document_id or "doc_" + corpus_id,
        "status": status, "extraction_mode": mode,
    }
    publication = {
        "publication_id": "internal", "title": 'Synthetic, "test"\nTitle',
        "year": None, "in_scope": included, "exclusion_reason": None if included else "Outside scope",
        "authors": ["Doe, Jane", "А. Тест"], "doi": "10.1234/example",
    }
    data = {table: [] for table in ("configurations", "datasets", "tasks", "experiments", "results", "issues", "evidence")}
    return {"document": document, "data": dict(data, publication=publication) if status == "completed" else None}


class BatchTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.schema = self.root / "schema.json"
        self.schema.write_text(json.dumps({"tables": TABLES}), encoding="utf-8")
        self.output = self.root / "submission"

    def cli(self, *arguments, schema=None):
        schema_args = schema or ("--schema", self.schema)
        stderr, stdout = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
            code = main([str(arg) for arg in (*arguments, *schema_args, "--output", self.output)])
        return code, stdout.getvalue(), stderr.getvalue()

    def export(self, *details):
        path = self.root / "export.json"
        path.write_text(json.dumps({"schema_version": "1.0", "documents": details}), encoding="utf-8")
        return path

    def csv(self, table):
        with (self.output / f"{table}.csv").open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            self.assertEqual(TABLES[table], reader.fieldnames)
            return list(reader)

    def api(self, route):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def respond(self):
                size = int(self.headers.get("Content-Length", 0))
                status, body = route(self.command, urlsplit(self.path), self.headers, self.rfile.read(size))
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = do_POST = respond

        server = HTTPServer(("127.0.0.1", 0), Handler)
        self.addCleanup(server.server_close)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}"

    def test_offline_csv_projection(self):
        included = detail(included=True)
        data = included["data"]
        result = {
            "result_id": "r1", "experiment_id": "e1", "config_id": "c1", "dataset_id": "d1", "task_id": "t1",
            "metric_name": "Recall@0", "metric_k": 0, "raw_value": "0%", "raw_unit": "%",
            "result_origin": "primary", "extraction_status": "extracted", "note": "Existing note",
        }
        data.update(
            configurations=[{"config_id": "c1", "top_k": 0, "chunk_overlap": 0}],
            datasets=[{"dataset_id": "d1", "dataset_name": None}],
            tasks=[{"task_id": "t1", "task_type": "question answering"}],
            experiments=[{"experiment_id": "e1", "latency_ms": 0}],
            results=[
                {"extraction_mode": "python", "result": result,
                 "normalized": {"value": 0, "unit": "fraction"}, "baseline": {"value": 0, "unit": "fraction"}},
                {"extraction_mode": "python", "result": dict(result, result_id="r2", duplicate_of="r1"),
                 "normalized": {"value": 0, "unit": "fraction"}},
                {"extraction_mode": "python", "result": {"result_id": "r3", "result_origin": "cited", "cited_publication_id": "[1]",
                            "raw_value": "7", "reported_value": 7}, "normalized": None},
            ],
            evidence=[
                {"evidence_id": "metric", "entity_type": "result", "entity_id": "r1", "field": "metric_name",
                 "page": 2, "fragment": "Recall@0"},
                {"evidence_id": "whole", "entity_type": "result", "entity_id": "r1", "field": None,
                 "page": 3, "figure": "Figure 1", "fragment": "Whole record"},
                {"evidence_id": "value", "entity_type": "result", "entity_id": "r1", "field": "raw_value",
                 "page": 4, "table": "Table 2", "table_row": 'Run "A"', "table_column": "Recall@0",
                 "fragment": 'Run "A", 0%\nEnd', "section": "Evaluation"},
                {"evidence_id": "other", "entity_type": "result", "entity_id": "r2", "page": 5, "figure": "Figure 2"},
            ],
            issues=[
                {"issue_id": "i1", "record_ids": "unrelated", "result_ids": ["r2", "r1"], "evidence_ids": ["other", "value"],
                 "check_name": "source_unverified", "description": 'Review, "please"'},
                {"issue_id": "i2", "result_ids": [], "evidence_ids": [], "check_name": "unreadable_pages"},
            ],
        )
        excluded = detail("D002")
        excluded["data"]["issues"] = [{"issue_id": "document_level"}]
        archive = self.root / "corpus.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.write(self.schema, SCHEMA_MEMBER)
        self.output.mkdir()  # Existing empty destination.
        code, _, err = self.cli("convert", self.export(excluded, included), schema=("--corpus-zip", archive))
        self.assertEqual(0, code, err)
        self.assertEqual({name + ".csv" for name in TABLES}, {path.name for path in self.output.iterdir()})
        first, second = self.csv("publications")
        self.assertEqual(("D001", 'Synthetic, "test"\nTitle', "Doe, Jane; А. Тест", "10.1234/example", "true", ""),
                         tuple(first[key] for key in ("publication_id", "title", "authors", "DOI", "in_scope", "year")))
        self.assertEqual(("D002", "false", "Outside scope"),
                         tuple(second[key] for key in ("publication_id", "in_scope", "exclusion_reason")))
        self.assertEqual("0", self.csv("configurations")[0]["top_k"])
        self.assertEqual("0", self.csv("configurations")[0]["chunk_overlap"])
        self.assertEqual("0", self.csv("experiments")[0]["latency_ms"])
        self.assertEqual("", self.csv("datasets")[0]["dataset_name"])
        self.assertEqual("t1", self.csv("tasks")[0]["task_id"])
        first, second, third = self.csv("results")
        self.assertEqual(("0", "0", "fraction", "0", "fraction", "0", "0%"),
                         tuple(first[key] for key in ("metric_k", "metric_value", "metric_unit", "normalized_value",
                                                       "normalized_unit", "baseline_value", "raw_value")))
        self.assertEqual(("4", "Evaluation", "Table 2", "", 'Run "A", 0%\nEnd'),
                         tuple(first[f"source_{field}"] for field in ("page", "section", "table", "figure", "fragment")))
        self.assertEqual('Existing note\nTable row: Run "A"\nTable column: Recall@0', first["note"])
        self.assertEqual(("e1", "c1", "d1", "t1", "r1", "Figure 2"),
                         tuple(second[key] for key in ("experiment_id", "config_id", "dataset_id", "task_id",
                                                        "duplicate_of", "source_figure")))
        self.assertEqual(("7", "", "", "[1]"),
                         tuple(third[key] for key in ("raw_value", "metric_value", "metric_unit", "cited_publication_id")))
        first, second = self.csv("issues")
        self.assertEqual(("r2; r1", "4", 'Run "A", 0%\nEnd'),
                         tuple(first[key] for key in ("record_ids", "source_page", "source_fragment")))
        self.assertEqual(("", ""), (second["record_ids"], second["source_page"]))

    def test_offline_reports_failed_missing_mixed_and_duplicate(self):
        for name in ("D001.pdf", "D002.pdf", "D003.pdf"):
            (self.root / name).touch()
        failed = detail(status="failed")
        failed["document"]["failure"] = {"code": "extraction_failed", "message": "example failure"}
        extra = detail("D004")
        extra["data"] = None
        path = self.export(failed, detail("D002", mode="mock"), detail("D002", document_id="doc_duplicate"), extra)
        code, _, err = self.cli("convert", path, "--pdf-dir", self.root)
        self.assertEqual(1, code)
        for message in ("Mixed extraction modes", "extraction_failed: example failure", "extraction_mode='mock'",
                        "D003: missing document", "duplicate corpus_id", "D004: unexpected document", "missing data"):
            self.assertIn(message, err)
        self.assertFalse(self.output.exists())

    def test_convert_requires_explicit_python_mode_on_each_result(self):
        included = detail(included=True)
        for mode in (None, "mock"):
            with self.subTest(mode=mode):
                summary = {"result": {"result_id": "r1"}}
                if mode is not None:
                    summary["extraction_mode"] = mode
                included["data"]["results"] = [summary]
                code, _, err = self.cli("convert", self.export(included))
                self.assertEqual(1, code)
                self.assertIn(f"result with extraction_mode={mode!r}", err)
                self.assertFalse(self.output.exists())

    def test_convert_rejects_result_rows_for_excluded_publications(self):
        for mode in (None, "mock", "python"):
            with self.subTest(mode=mode):
                self.output = self.root / f"excluded-{mode}"
                excluded = detail("D002")
                summary = {"result": {"result_id": "r1"}}
                if mode is not None:
                    summary["extraction_mode"] = mode
                excluded["data"]["results"] = [summary]
                code, _, err = self.cli("convert", self.export(excluded))
                self.assertEqual(1, code)
                self.assertIn("D002 (doc_D002)", err)
                self.assertIn("excluded publication has result rows", err)
                self.assertFalse(self.output.exists())

    def test_single_document_detail_and_existing_output(self):
        path = self.root / "detail.json"
        path.write_text(json.dumps(detail()), encoding="utf-8")
        code, _, err = self.cli("convert", path)
        self.assertEqual(0, code, err)
        self.assertEqual("D001", self.csv("publications")[0]["publication_id"])
        code, _, err = self.cli("convert", path)
        self.assertEqual(1, code)
        self.assertIn("output must be a new or empty directory", err)

    def test_upload_then_poll_and_resume_without_upload(self):
        pdf = b"%PDF-1.7\nsynthetic bytes"
        (self.root / "D001.pdf").write_bytes(pdf)
        requests = []
        finished = detail()

        def route(method, url, headers, body):
            requests.append((method, url, headers, body))
            if method == "POST":
                return 202, dict(finished["document"], status="queued")
            if url.path == "/api/v1/documents":
                page = parse_qs(url.query)["page"]
                items = [finished["document"]] if page == ["2"] else [{"corpus_id": None}] * 100
                return 200, {"items": items, "total": 101}
            if len(requests) == 2:
                return 200, detail(status="processing")
            return 200, finished

        api = self.api(route)
        code, _, err = self.cli("run", self.root, "--api-url", api, "--poll-interval", 0.01)
        self.assertEqual(0, code, err)
        self.assertIn("queued [python]", err)
        self.assertIn("processing [python]", err)
        self.assertIn("completed [python]", err)
        method, url, headers, body = requests[0]
        self.assertEqual(("POST", "/api/v1/documents"), (method, url.path))
        multipart = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {headers['Content-Type']}\r\n\r\n".encode() + body)
        [part] = list(multipart.iter_parts())
        self.assertEqual("D001.pdf", part.get_filename())
        self.assertEqual("file", part.get_param("name", header="content-disposition"))
        self.assertEqual(pdf, part.get_payload(decode=True))
        self.assertEqual("D001", self.csv("publications")[0]["publication_id"])
        requests.clear()
        self.output = self.root / "resumed"
        code, _, err = self.cli("run", self.root, "--resume", "--api-url", api + "/api/v1")
        self.assertEqual(0, code, err)
        self.assertEqual(["GET"] * 3, [method for method, *_ in requests])
        self.assertEqual("D001", self.csv("publications")[0]["publication_id"])

    def test_ambiguous_resume_and_pinned_job(self):
        selected = detail(document_id="doc_chosen")

        def route(method, url, headers, body):
            if method == "GET" and url.path == "/api/v1/documents":
                return 200, {"total": 2, "items": [selected["document"], detail()["document"]]}
            if method == "GET" and url.path == "/api/v1/documents/doc_chosen":
                return 200, selected
            return 404, {"error": {"code": "not_found", "message": "Missing job"}}

        api = self.api(route)
        code, _, err = self.cli("resume", "D001", "--api-url", api)
        self.assertEqual(1, code)
        self.assertIn("multiple jobs (doc_D001, doc_chosen)", err)
        code, _, err = self.cli("resume", "D002", "--api-url", api)
        self.assertEqual(1, code)
        self.assertIn("D002: missing document", err)
        code, _, err = self.cli("resume", "D001", "--document-id", "D001=doc_missing", "--api-url", api)
        self.assertEqual(1, code)
        self.assertIn("HTTP 404", err)
        self.assertIn("not_found: Missing job", err)
        code, _, err = self.cli("resume", "D001", "--document-id", "D001=doc_chosen", "--api-url", api)
        self.assertEqual(0, code, err)

    def test_timeout_includes_corpus_job_and_last_status(self):
        processing = detail(status="processing")
        api = self.api(lambda *_: (200, processing))
        code, _, err = self.cli("resume", "D001", "--document-id", "D001=doc_D001", "--api-url", api,
                                "--timeout", 0.1, "--poll-interval", 0.2)
        self.assertEqual(1, code)
        self.assertIn("Timed out waiting for jobs after 0.1s", err)
        self.assertIn("D001 (doc_D001): last status=processing", err)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
