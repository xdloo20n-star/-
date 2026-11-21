"""HTTP client for Sverka's document queue."""

import errno
import http.client
import itertools
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


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


class SverkaAPI:
    def __init__(self, url, request_timeout):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment:
            raise ValueError("--api-url must be an HTTP(S) URL without a query or fragment")
        self.url = url.rstrip("/")
        if not self.url.endswith("/api/v1"):
            self.url += "/api/v1"
        self.request_timeout = request_timeout

    def request(self, method, path, *, body=None, headers=None, timeout=None):
        request = urllib.request.Request(self.url + path, data=body, headers=headers or {}, method=method)
        timeout = min(timeout, self.request_timeout) if timeout is not None else self.request_timeout
        label = f"{method} {path}"
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                expected = 202 if method == "POST" else 200
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

    def get(self, path, *, timeout=None):
        deadline = time.monotonic() + (self.request_timeout if timeout is None else timeout)
        last_error = None
        for retry_delay in (0.2, 0.4, None):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise APIError(f"GET {path}: read deadline exceeded") from last_error
            try:
                return self.request("GET", path, timeout=remaining)
            except TemporaryAPIError as exc:
                if retry_delay is None or retry_delay >= deadline - time.monotonic():
                    raise
                last_error = exc
                time.sleep(retry_delay)

    def upload(self, pdf):
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
            return self.request("POST", "/documents", body=body, headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(prefix) + size + len(suffix)),
            })

    def detail(self, document_id, *, timeout=None):
        path = "/documents/" + urllib.parse.quote(document_id, safe="")
        return self.get(path, timeout=timeout)

    def documents(self):
        page_size = 100
        page = 1
        while True:
            response = self.get(f"/documents?page={page}&page_size={page_size}")
            items, total = response.get("items"), response.get("total")
            if (not isinstance(items, list) or any(not isinstance(item, dict) for item in items)
                    or type(total) is not int or total < 0):
                raise APIError(f"GET /documents: invalid pagination data on page {page}")
            yield from items
            if page * page_size >= total:
                return
            page += 1
