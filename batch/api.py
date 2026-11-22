"""HTTP client for Sverka's document queue."""

import errno
import http.client
import itertools
import json
import os
import re
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import Future
from contextlib import suppress


class APIError(Exception):
    pass


class TemporaryAPIError(APIError):
    pass


TEMPORARY_HTTP_STATUSES = {408, 502, 503, 504}
TEMPORARY_NETWORK_ERRORS = {errno.ENETDOWN, errno.ENETUNREACH, errno.EHOSTUNREACH, socket.EAI_AGAIN}


def problem_text(problem):
    if isinstance(problem, dict):
        return ": ".join(str(problem[key]) for key in ("code", "message") if problem.get(key))
    return ""


class DeadlineHandler(urllib.request.HTTPHandler, urllib.request.HTTPSHandler):
    def __init__(self, context):
        super().__init__(context=context)
        self.lock = threading.Lock()
        self.socket = None
        self.aborted = False

    def do_open(self, connection_class, request, **kwargs):
        track_socket = self.track_socket

        class Connection(connection_class):
            def connect(self):
                super().connect()
                track_socket(self.sock)

        return super().do_open(Connection, request, **kwargs)

    def track_socket(self, sock):
        with self.lock:
            if self.aborted:
                raise TimeoutError("request deadline exceeded")
            self.socket = sock

    def abort(self):
        with self.lock:
            self.aborted = True
            if self.socket is not None:
                with suppress(OSError):
                    self.socket.shutdown(socket.SHUT_RDWR)


class SverkaAPI:
    def __init__(self, url, request_timeout):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment:
            raise ValueError("--api-url must be an HTTP(S) URL without a query or fragment")
        self.url = url.rstrip("/")
        if not self.url.endswith("/api/v1"):
            self.url += "/api/v1"
        self.request_timeout = request_timeout
        self.tls_context = ssl.create_default_context()
        self.tls_context.set_alpn_protocols(["http/1.1"])

    def request(self, method, path, *, body=None, headers=None, deadline=None):
        started = time.monotonic()
        limit = started + self.request_timeout
        if deadline is not None:
            limit = min(limit, deadline)
        timeout = limit - started
        label = f"{method} {path}"
        if timeout <= 0:
            raise TemporaryAPIError(f"{label}: request deadline exceeded")
        handler = DeadlineHandler(self.tls_context)
        result = Future()

        def perform():
            try:
                opener = urllib.request.build_opener(handler)
                request = urllib.request.Request(self.url + path, data=body, headers=headers or {}, method=method)
                result.set_result(self._read_response(opener, request, label, timeout))
            except Exception as exc:
                result.set_exception(exc)

        # Socket timeouts limit inactivity. Bound the whole exchange, including
        # headers and error bodies, and interrupt an active socket on expiry.
        try:
            threading.Thread(target=perform, daemon=True).start()
            response = result.result(timeout=max(0, limit - time.monotonic()))
            if time.monotonic() >= limit:
                raise TimeoutError
            return response
        except TimeoutError as exc:
            raise TemporaryAPIError(f"{label}: request timed out after {timeout:.2f}s") from exc
        finally:
            if not result.done():
                handler.abort()

    @staticmethod
    def _read_response(opener, request, label, timeout):
        try:
            with opener.open(request, timeout=timeout) as response:
                expected = 202 if request.method == "POST" else 200
                if response.status != expected:
                    raise APIError(f"{label}: expected HTTP {expected}, got {response.status}")
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            with exc:
                try:
                    payload = json.loads(exc.read(8192))
                    reason = problem_text(payload.get("error")) if isinstance(payload, dict) else ""
                except (ValueError, OSError, http.client.HTTPException):
                    reason = ""
            missing = " (missing document or endpoint)" if exc.code == 404 else ""
            error = TemporaryAPIError if exc.code in TEMPORARY_HTTP_STATUSES else APIError
            raise error(f"{label}: HTTP {exc.code}{missing}: {reason or exc.reason}") from exc
        except (OSError, http.client.HTTPException) as exc:
            reason = getattr(exc, "reason", exc)
            temporary = isinstance(reason, (TimeoutError, ConnectionError, http.client.IncompleteRead)) or (
                isinstance(reason, OSError) and reason.errno in TEMPORARY_NETWORK_ERRORS
            )
            error = TemporaryAPIError if temporary else APIError
            if isinstance(reason, TimeoutError):
                reason = f"request timed out after {timeout:.2f}s"
            raise error(f"{label}: {reason}") from exc
        except ValueError as exc:
            raise APIError(f"{label}: {exc}") from exc
        if not isinstance(result, dict):
            raise APIError(f"{label}: expected a JSON object")
        return result

    def get(self, path, *, deadline=None):
        if deadline is None:
            deadline = time.monotonic() + self.request_timeout
        last_error = None
        for retry_delay in (0.2, 0.4, None):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise APIError(f"GET {path}: read deadline exceeded") from last_error
            try:
                return self.request("GET", path, deadline=deadline)
            except TemporaryAPIError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise APIError(f"GET {path}: read deadline exceeded") from exc
                if retry_delay is None or retry_delay >= remaining:
                    raise
                last_error = exc
                time.sleep(retry_delay)

    def upload(self, pdf, *, reuse=False):
        boundary = uuid.uuid4().hex
        # The CLI validates the filename against the API's corpus naming rule.
        prefix = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{pdf.name}"\r\n'
            "Content-Type: application/pdf\r\n\r\n"
        ).encode("ascii")
        suffix = f"\r\n--{boundary}--\r\n".encode("ascii")
        with pdf.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            body = itertools.chain((prefix,), iter(lambda: stream.read(64 * 1024), b""), (suffix,))
            path = "/documents?reuse=true" if reuse else "/documents"
            return self.request("POST", path, body=body, headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(prefix) + size + len(suffix)),
            })

    def detail(self, document_id, *, deadline):
        path = "/documents/" + urllib.parse.quote(document_id, safe="")
        return self.get(path, deadline=deadline)

    def digest(self, document_id, *, deadline):
        path = "/documents/" + urllib.parse.quote(document_id, safe="") + "/digest"
        sha256 = self.get(path, deadline=deadline).get("sha256")
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise APIError(f"GET {path}: invalid SHA-256 digest")
        return sha256

    def documents(self, corpus_id, *, deadline):
        page_size = 100
        page = 1
        previous_total = None
        while True:
            query = urllib.parse.urlencode({"corpus_id": corpus_id, "page": page, "page_size": page_size})
            response = self.get(f"/documents?{query}", deadline=deadline)
            if response.get("corpus_id") != corpus_id:
                raise APIError(f"{corpus_id}: API did not confirm the exact corpus filter. Upgrade the Go API.")
            items, total = response.get("items"), response.get("total")
            if (not isinstance(items, list) or any(not isinstance(item, dict) for item in items)
                    or type(total) is not int or total < 0):
                raise APIError(f"GET /documents: invalid pagination data on page {page}")
            if (previous_total is not None and previous_total != total
                    or len(items) != min(page_size, max(0, total - (page - 1) * page_size))):
                raise APIError(f"{corpus_id}: document list changed or returned incomplete pagination. Run again.")
            yield from items
            if page * page_size >= total:
                return
            previous_total = total
            page += 1
