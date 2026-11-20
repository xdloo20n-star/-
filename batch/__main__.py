"""Submit and resume Sverka corpus jobs, or convert a saved Go export."""

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

from .api import APIError, SverkaAPI
from .submission import (
    CORPUS_ID,
    DOCUMENT_ID,
    check_output,
    document_metadata,
    export_details,
    job_label,
    mode_errors,
    pdf_inputs,
    read_schema,
    status_error,
    write_submission,
)


ACTIVE = ("queued", "processing")


def progress(message):
    print(message, file=sys.stderr, flush=True)


def duration(raw):
    try:
        number = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected positive seconds") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("expected finite, positive seconds")
    return number


def parser():
    root = argparse.ArgumentParser(prog="python3 -m batch", description=__doc__)
    subcommands = root.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run", help="submit a directory of corpus PDFs")
    run.add_argument("pdf_dir", type=Path)
    run.add_argument("--resume", action="store_true",
                     help="reuse unique jobs by corpus ID and upload PDFs without a job")
    resume = subcommands.add_parser("resume", help="wait for existing jobs by corpus ID without uploading")
    resume.add_argument("corpus_ids", nargs="+", metavar="CORPUS_ID")
    convert = subcommands.add_parser("convert", help="convert a saved Go export or single DocumentDetail offline")
    convert.add_argument("export", type=Path)
    convert.add_argument("--pdf-dir", type=Path, help="require the same corpus IDs as this directory")
    for command in (run, resume, convert):
        command.add_argument("--output", required=True, type=Path, help="new or empty directory for the seven CSVs")
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--schema", type=Path, help="organizer submission_template/schema.json")
        source.add_argument("--corpus-zip", type=Path, help="organizer ZIP containing submission_template/schema.json")
    for command in (run, resume):
        command.add_argument("--api-url", default="http://127.0.0.1:8080", help="Go API base URL")
        command.add_argument("--document-id", action="append", default=[], metavar="CORPUS_ID=DOCUMENT_ID",
                             help="pick an exact server job for an input (repeatable)")
        command.add_argument("--timeout", type=duration, default=3600,
                             help="seconds to wait after uploading (default 3600)")
        command.add_argument("--request-timeout", type=duration, default=30,
                             help="HTTP request timeout in seconds (default 30)")
        command.add_argument("--poll-interval", type=duration, default=2,
                             help="seconds between polling rounds (default 2)")
    return root


def requested_jobs(api, corpus_ids, pins, *, discover, require_existing):
    selected = {}
    for pin in pins:
        corpus_id, _, document_id = pin.partition("=")
        if corpus_id not in corpus_ids or not DOCUMENT_ID.fullmatch(document_id):
            raise ValueError(f"invalid --document-id {pin!r}; use CORPUS_ID=DOCUMENT_ID for a selected PDF")
        if corpus_id in selected:
            raise ValueError(f"multiple --document-id mappings for {corpus_id}")
        selected[corpus_id] = document_id

    errors = []
    to_find = set(corpus_ids) - selected.keys()
    if discover and to_find:
        candidates = defaultdict(set)
        for document in api.documents():
            corpus_id, document_id = document.get("corpus_id"), document.get("document_id")
            if isinstance(corpus_id, str) and corpus_id in to_find:
                if not isinstance(document_id, str) or not DOCUMENT_ID.fullmatch(document_id):
                    raise APIError(f"{corpus_id}: invalid or missing document_id in GET /documents")
                candidates[corpus_id].add(document_id)
        for corpus_id in sorted(to_find):
            matches = sorted(candidates[corpus_id])
            if len(matches) == 1:
                selected[corpus_id] = matches[0]
            elif matches:
                errors.append(f"{corpus_id}: multiple jobs ({', '.join(matches)}). Select one with --document-id.")
            elif require_existing:
                errors.append(f"{corpus_id}: missing document in the API")
    return selected, errors


def observe(states, corpus_id, document, document_id=None):
    actual_id = document.get("document_id")
    if not isinstance(actual_id, str) or not DOCUMENT_ID.fullmatch(actual_id):
        raise ValueError(f"missing or invalid document_id for {corpus_id}")
    if document_id is not None and actual_id != document_id:
        raise ValueError(f"API returned document_id {actual_id!r}, expected {document_id!r}")
    if document.get("corpus_id") != corpus_id:
        raise ValueError(f"{actual_id}: API returned corpus_id {document.get('corpus_id')!r}, expected {corpus_id!r}")
    previous = states.get(corpus_id, {})
    status = document.get("status")
    mode = document.get("extraction_mode")
    if previous.get("status") != status or previous.get("extraction_mode") != mode:
        progress(f"{job_label(document)}: {status} [{mode}]")
    states[corpus_id] = document
    if status not in ("queued", "processing", "completed"):
        raise ValueError(status_error(document))


def collect_jobs(api, selected, states, complete, *, deadline=None):
    errors = []
    for corpus_id, document_id in sorted(selected.items()):
        if corpus_id in complete:
            continue
        remaining = deadline - time.monotonic() if deadline is not None else None
        if remaining is not None and remaining <= 0:
            break
        try:
            detail = api.detail(document_id, timeout=remaining)
            document = document_metadata(detail)
            observe(states, corpus_id, document, document_id)
            if document["status"] == "completed":
                if not isinstance(detail.get("data"), dict):
                    raise ValueError("completed job is missing data")
                complete[corpus_id] = detail
        except (APIError, ValueError) as exc:
            label = f"{corpus_id} ({document_id})"
            message = str(exc)
            errors.append(message if message.startswith(label) else f"{label}: {message}")
    return errors


def wait_for_jobs(api, selected, states, complete, args):
    deadline = time.monotonic() + args.timeout
    while len(complete) < len(selected):
        errors = collect_jobs(api, selected, states, complete, deadline=deadline)
        errors.extend(mode_errors(states.values()))
        if len(complete) == len(selected) and not errors:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pending = [
                f"{corpus_id} ({document_id}): last status={states.get(corpus_id, {}).get('status', 'unknown')}"
                for corpus_id, document_id in sorted(selected.items())
                if states.get(corpus_id, {}).get("status") in ACTIVE or corpus_id not in states
            ]
            if pending:
                errors.extend([f"Timed out waiting for jobs after {args.timeout:g}s.", *pending])
        if errors:
            raise ValueError("\n".join(errors))
        time.sleep(min(args.poll_interval, remaining))


def run_batch(args, inputs, corpus_ids):
    api = SverkaAPI(args.api_url, args.request_timeout)
    resume_only = args.command == "resume"
    if (len(set(corpus_ids)) != len(corpus_ids)
            or any(not CORPUS_ID.fullmatch(corpus_id) for corpus_id in corpus_ids)):
        raise ValueError("pass distinct corpus IDs such as D001 or P090")
    selected, errors = requested_jobs(api, corpus_ids, args.document_id,
                                      discover=resume_only or args.resume, require_existing=resume_only)
    states, complete = {}, {}
    uploads = inputs.keys() - selected.keys()
    if errors or uploads:
        preflight_deadline = time.monotonic() + args.timeout
        errors.extend(collect_jobs(api, selected, states, complete, deadline=preflight_deadline))
        errors.extend(mode_errors(states.values()))
        if not errors:
            errors.extend(f"{corpus_id} ({selected[corpus_id]}): timed out checking job status before upload"
                          for corpus_id in sorted(selected.keys() - states.keys()))
        if errors:
            raise ValueError("\n".join(errors))

    for corpus_id in sorted(uploads):
        try:
            document = api.upload(inputs[corpus_id])
            observe(states, corpus_id, document)
        except (APIError, OSError, ValueError) as exc:
            raise ValueError(f"{corpus_id}: upload failed: {exc}\n"
                             "Check the API or use run --resume before uploading again.") from exc
        selected[corpus_id] = document["document_id"]
        errors = mode_errors(states.values())
        if errors:
            raise ValueError("\n".join(errors))

    wait_for_jobs(api, selected, states, complete, args)
    return complete.values()


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        check_output(args.output)
        tables = read_schema(args.schema, args.corpus_zip)
        if args.command == "convert":
            details = export_details(args.export)
            expected_ids = pdf_inputs(args.pdf_dir) if args.pdf_dir is not None else None
        else:
            inputs = pdf_inputs(args.pdf_dir) if args.command == "run" else {}
            expected_ids = list(inputs) if args.command == "run" else args.corpus_ids
            details = run_batch(args, inputs, expected_ids)
        counts = write_submission(details, tables, args.output, expected_ids=expected_ids)
    except (APIError, OSError, ValueError) as exc:
        progress(f"ERROR: {exc}")
        return 1
    except KeyboardInterrupt:
        progress("Interrupted. Already submitted jobs remain on the API.")
        return 130
    print(f"Wrote {len(counts)} CSVs to {args.output} ({', '.join(f'{key}={count}' for key, count in counts.items())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
