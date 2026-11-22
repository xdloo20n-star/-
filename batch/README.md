# Sverka corpus batch CLI

Run these commands from the repository root with Python 3.10 or later. The CLI uses the Python standard library. Supply the organizer's archive with `--corpus-zip` or its extracted `submission_template/schema.json` with `--schema`. Each export requires a new or empty output directory. The examples write under `qa-output/`.

## Submit a directory

Start the Go API in Python mode as described in the repository README. Pass a single directory of corpus PDFs. The API recognizes filenames such as `D019.pdf` and `P090.pdf`.

```bash
python3 -m batch run /path/to/test \
  --corpus-zip ../drive-download-20261120T125610Z-1-001.zip \
  --api-url http://127.0.0.1:8080 --timeout 3600 --output qa-output/batch-test-run
```

The CLI reports job IDs and status changes on stderr. `--timeout` defaults to 3600 seconds. A run that uploads uses this budget for job selection and checks, then gets a fresh wait budget after uploads. A run that reuses every job and the `resume` subcommand share one budget across selection and polling. `--request-timeout` caps each complete HTTP exchange, including slow responses, and defaults to 30 seconds. `--poll-interval` defaults to 2 seconds. If a job fails or the wait expires, the command returns 1 with available job IDs and error details. Already queued jobs continue on the server.

The CLI retries reads of the document list, PDF digest, or job status up to twice after a temporary network failure or HTTP 408, 502, 503 or 504. Retries use the remaining budget. Each upload POST is attempted once. After an upload connection fails, use `run --resume` with the same PDFs to recover.

## Resume jobs

Use the same API instance. `run --resume` looks up each unpinned corpus ID using the API's exact filter. Each selected job's PDF must have the same SHA-256 digest as the local PDF. For a missing job, the CLI sends `POST /api/v1/documents?reuse=true`. If another uploader fills the gap, the API can return that job when it is the sole match and has identical PDF bytes and extraction mode. A conflict stops the run.

The `resume` subcommand waits for exactly the requested corpus IDs and requires all of them to exist. It uses the exact filter or explicit job IDs, with no local PDFs required.

```bash
python3 -m batch run /path/to/test --resume --schema /path/to/schema.json --output qa-output/batch-test-resumed
python3 -m batch resume D019 D020 --schema /path/to/schema.json --output qa-output/batch-dev
```

If the API holds multiple jobs for a corpus ID, choose the server job explicitly. Repeat `--document-id` for each ID you need to disambiguate. With `run`, pins still require a matching local PDF digest and can be combined with uploads for the rest of the directory. To retry failed extraction after addressing the cause, use plain `run` on the affected PDFs and pin the new jobs in subsequent runs.

```bash
python3 -m batch resume D019 --document-id D019=doc_example \
  --schema /path/to/schema.json --output qa-output/batch-dev-pinned
```

Filtered discovery and atomic uploads require the updated Go API. The CLI requires each filtered response page to echo the requested `corpus_id` before it can use the results, including an empty page. A pinned job with a local PDF requires the digest endpoint. Existing jobs can be verified if the server still has their original PDFs. Plain unpinned `run`, fully pinned `resume`, and offline `convert` work with the older API contract.

## Convert saved JSON

The converter accepts a saved `GET /api/v1/export` response with schema version `1.0` or one `GET /api/v1/documents/{id}` response. To check that the export covers exactly the intended corpus, pass its PDF directory.

```bash
curl -fsS http://127.0.0.1:8080/api/v1/export -o export.json
python3 -m batch convert export.json --pdf-dir /path/to/test \
  --schema /path/to/schema.json --output qa-output/batch-test-converted
```

A successful conversion writes the seven CSVs together. Every selected job must be completed in Python mode with its own corpus ID. A failed or missing job, a duplicate corpus ID, a selection containing mock data, or an excluded publication with result rows stops the export. A valid excluded article contributes only its publication row.

The exporter uses Go's normalized pair and comparable baseline value. Each result uses a single source, preferring raw-value evidence that fits the organizer's locator fields. Whole-result evidence is the fallback. An extracted result needs a positive page and a table, figure or quote. Table coordinates and any block or bounding-box locator are carried in `note`. A cited result needs `cited_publication_id`. Keep the JSON for the full evidence list. Run the organizer's checker on the output and review any findings before submission.

```bash
python3 /path/to/tools/check_submission.py qa-output/batch-test-converted /path/to/test --schema /path/to/schema.json
python3 -m unittest discover -s batch -v
```

## Evaluate against the local dev gold

```bash
python3 -m batch.evaluate /path/to/dev-submission \
  --corpus-zip ../drive-download-20261120T125610Z-1-001.zip

python3 -m batch.evaluate /path/to/dev-submission \
  --gold /path/to/dev_gold --diagnostics --json-report report.json

python3 -m unittest batch.test_evaluate -v
```

The evaluator reads the CSVs locally with the standard library. `--json-report -` emits JSON on stdout for piping. A report file adds JSON alongside the usual text output. The same CSVs produce the same report through the archive and directory inputs.

`--diagnostics` adds counts of matches gained by ignoring one source field at a time. The report compares unmatched rows in the same PDF and keeps each row eligible for at most one match per field. Counts for different fields may overlap. Precision, recall and F1 continue to use the strict keys.

PDF coverage comes from `publications.csv` and includes excluded publications. Missing gold PDFs contribute zero candidate records. Candidate PDFs outside gold are listed with their row counts and sit outside the agreement denominators. Each paper shows its inclusion decision and counts of extracted, absent, present-but-unextractable and marked duplicate rows. The denominators use canonical `extracted` rows on each side's included PDFs within the gold set. The exporter and evaluator require each `duplicate_of` link to connect two compatible extracted rows in the same publication. A chain must end at a canonical row. Rows on excluded publications are counted separately. Unmarked rows with the same key retain their multiplicity.

All-results agreement counts every eligible row. Table-only agreement uses eligible rows with a table locator on each side. A source match requires a positive PDF page and a table or figure locator, and compares the raw value and unit, metric label, cutoff, metric variant, aggregation and locator. Table and figure locators remain distinct. Eligible rows lacking the fields needed for a key remain unmatched in their scope. The report counts how many rows can form source and context keys.

The keys collapse whitespace, normalize Unicode NFC and lowercase labels. For plain raw decimals they normalize leading and trailing zeros and mathematical minus while retaining explicit signs. If the raw value includes `%` and `raw_unit` independently says `%` or `percent`, the redundant marker is folded into the unit key; a percent sign with no explicit percent unit is retained. Raw units otherwise retain case. Cutoffs can be expressed as a terminal `@k` or in `metric_k`. The metric name key treats underscore, hyphen and space as equivalent separators in numbered BLEU and ROUGE labels and in ROUGE-L. Locator prefixes such as `Table` or `Fig.` are optional and locator suffixes are retained. The evaluator uses no rounding tolerance, scale conversion, inferred cutoffs or semantic aliases. Blank values and zero are distinct.

The contextual score adds exact lexical agreement on `rag_variant`, `generator_model`, `generator_version`, `top_k`, `dataset_name`, `dataset_version` and `data_split`. It resolves configuration and dataset IDs separately within each input and publication. Both entries must resolve and contain a variant and dataset name. The score uses the same denominators as source agreement. If either input lacks the configuration or dataset CSV, context agreement is unavailable.

Within each PDF and scope, each exact key can match up to the smaller of its two row counts. Micro scores pool counts across PDFs. Macro scores average each ratio over papers with eligible gold rows in that scope and use zero precision for papers with no eligible candidate rows. Undefined per-paper or micro ratios appear as `n/a` or JSON `null`. Precision, recall and F1 quantify agreement under these local matching criteria. They do not establish scientific validity or verify evidence against the PDFs.
