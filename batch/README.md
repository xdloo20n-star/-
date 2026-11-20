# Sverka corpus batch CLI

Run these commands from the repository root with Python 3.10 or later. The CLI uses the Python standard library. Supply the organizer's archive with `--corpus-zip` or its extracted `submission_template/schema.json` with `--schema`.

## Submit a directory

Start the Go API in Python mode as described in the repository README. Pass a single directory of corpus PDFs. The API recognizes filenames such as `D019.pdf` and `P090.pdf`.

```bash
python3 -m batch run /path/to/test \
  --corpus-zip ../drive-download-20261120T125610Z-1-001.zip \
  --api-url http://127.0.0.1:8080 --timeout 3600 --output submission
```

The CLI reports job IDs and status changes on stderr. `--timeout` is the wait budget in seconds after uploads. `--request-timeout` and `--poll-interval` default to 30 and 2 seconds. If a job fails or the wait expires, the command returns 1 with the affected IDs and any error the API supplied. Already queued jobs continue on the server. The Go logs include the extractor's error details.

## Resume jobs

Use the same API instance. `run --resume` looks for existing jobs by corpus ID and uploads only inputs without a match. The `resume` subcommand waits for exactly the requested corpus IDs and requires all of them to exist.

```bash
python3 -m batch run /path/to/test --resume --schema /path/to/schema.json --output submission
python3 -m batch resume D019 D020 --schema /path/to/schema.json --output dev-submission
```

If the API holds multiple jobs for a corpus ID, choose the server job explicitly. Repeat `--document-id` for each ID you need to disambiguate. With `run` it can also pin selected jobs while uploading the rest of the directory. To retry failed extraction after addressing the cause, use `run` on the affected PDFs and pin the new jobs in subsequent runs.

```bash
python3 -m batch resume D019 --document-id D019=doc_example \
  --schema /path/to/schema.json --output dev-submission
```

## Convert saved JSON

The converter accepts a saved `GET /api/v1/export` response with schema version `1.0` or one `GET /api/v1/documents/{id}` response. To check that the export covers exactly the intended corpus, pass its PDF directory.

```bash
curl -fsS http://127.0.0.1:8080/api/v1/export -o export.json
python3 -m batch convert export.json --pdf-dir /path/to/test \
  --schema /path/to/schema.json --output submission
```

All commands require a new or empty output directory. A successful conversion writes the seven CSVs together. Every selected job must be completed in Python mode with its own corpus ID. A failed or missing job, a duplicate corpus ID, or a selection containing mock data stops the export. The CSVs retain one publication per PDF and only the publication row for an excluded article.

The exporter uses Go's normalized pair and comparable baseline value. Each result uses a single source, preferring raw-value evidence that fits the organizer's locator fields. Whole-result evidence is the fallback. Table coordinates and any block or bounding-box locator are carried in `note`. Keep the JSON for the full evidence list. Run the organizer's checker on the output and review any findings before submission.

```bash
python3 /path/to/tools/check_submission.py submission /path/to/test --schema /path/to/schema.json
python3 -m unittest discover -s batch -v
```
