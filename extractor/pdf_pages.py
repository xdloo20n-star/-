"""Extract page text with Poppler and use optical recognition for scanned pages."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from xml.etree import ElementTree

TABLE_HEADING = re.compile(r"^\s*table\s+(?:\d+|[ivxlcdm]+)\b", re.I | re.M)
CONTROL_GLYPHS = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")
FAILURE_RATE_HEADINGS = re.compile(r"\bLeak\s*(?:↓\s*)?Stale\s*(?:↓\s*)?Unsup\.", re.I)


@dataclass(frozen=True)
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def x(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def y(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass(frozen=True)
class Page:
    number: int
    text: str
    verification: str
    reading_text: str = ""
    words: tuple[Word, ...] = ()

    @property
    def prose(self) -> str:
        return " ".join((self.reading_text or self.text).split())


@dataclass(frozen=True)
class PDF:
    metadata: dict[str, str]
    pages: list[Page]


class ExtractionError(Exception):
    """The PDF or a tool could not be used."""


def run(command: list[str], *, data: bytes | None = None, timeout: int = 40,
        env: dict[str, str] | None = None) -> bytes:
    try:
        result = subprocess.run(
            command, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True, timeout=timeout, env=env,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace").strip()[:1000]
        raise ExtractionError(f"{Path(command[0]).name}: {detail or error.returncode}") from error
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ExtractionError(str(error)) from error
    return result.stdout


def as_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def usable(text: str) -> bool:
    # A page number or a stray invisible character does not constitute a text layer.
    return len(re.findall(r"[^\W\d_]{2,}", text, re.UNICODE)) >= 8


def ocr_page(path: Path, number: int, engine: str, dpi: int = 200) -> tuple[str, str]:
    # Keep Poppler's RGB pixels and skip PNG compression before local OCR.
    image = run([
        "pdftoppm", "-f", str(number), "-l", str(number), "-singlefile",
        "-r", str(dpi), str(path),
    ], timeout=60)
    env = dict(os.environ)
    env.setdefault("OMP_THREAD_LIMIT", "1")
    recognized_pdf = run([
        engine, "stdin", "stdout", "-l", env.get("OCR_LANG", "eng"),
        "--dpi", str(dpi), "--psm", "3", "-c", "tessedit_create_pdf=1", "-c", "textonly_pdf=1",
    ], data=image, timeout=75, env=env)
    layout = as_text(run(["pdftotext", "-layout", "-enc", "UTF-8", "-", "-"], data=recognized_pdf))
    reading = as_text(run(["pdftotext", "-enc", "UTF-8", "-", "-"], data=recognized_pdf))
    return layout.rstrip("\f"), reading.rstrip("\f")


def poppler_text(path: Path, count: int, *flags: str) -> list[str]:
    command = ["pdftotext", "-enc", "UTF-8", *flags]
    bulk = as_text(run([*command, str(path), "-"])).split("\f")
    if len(bulk) == count + 1 and not bulk[-1].strip():
        return [CONTROL_GLYPHS.sub(" ", page) for page in bulk[:-1]]
    # Embedded control glyphs in some figures are indistinguishable from bulk separators.
    return [
        CONTROL_GLYPHS.sub(" ", as_text(run([
            *command, "-f", str(number), "-l", str(number), "-nopgbrk", str(path), "-",
        ])))
        for number in range(1, count + 1)
    ]


def positioned_words(path: Path, number: int) -> tuple[Word, ...]:
    data = run(["pdftotext", "-f", str(number), "-l", str(number), "-enc", "UTF-8", "-bbox", str(path), "-"])
    document = ElementTree.fromstring(CONTROL_GLYPHS.sub(" ", as_text(data)))
    return tuple(Word(word.text, float(word.attrib["xMin"]), float(word.attrib["yMin"]),
                      float(word.attrib["xMax"]), float(word.attrib["yMax"]))
                 for word in document.findall(".//{*}word") if word.text)


def scanned_page(path: Path, number: int, engine: str) -> Page:
    try:
        layout, reading = ocr_page(path, number, engine)
    except ExtractionError as error:
        print(f"page {number}: {error}", file=sys.stderr)
        return Page(number, "", "unverified")
    if TABLE_HEADING.search(layout):
        try:
            precise, precise_reading = ocr_page(path, number, engine, dpi=360)
            if usable(precise):
                layout, reading = precise, precise_reading
        except ExtractionError as error:
            print(f"page {number}, high resolution: {error}", file=sys.stderr)
    if usable(layout):
        return Page(number, layout, "ocr_match", reading)
    return Page(number, "", "unverified")


def read_pdf(path: Path) -> PDF:
    info = as_text(run(["pdfinfo", "-enc", "UTF-8", str(path)]))
    metadata = {key: value.strip() for key, value in re.findall(r"^([\w ]+):[ \t]*(.*)$", info, re.M)}
    page_count = int(metadata.get("Pages", "0"))
    if not page_count:
        raise ExtractionError("PDF page count is unavailable")
    text = poppler_text(path, page_count, "-layout")
    reading = poppler_text(path, page_count)
    engine = shutil.which(os.environ.get("TESSERACT_BIN", "tesseract"))
    pages = [Page(number, native, "text_match", reading[number - 1]) if usable(native)
             else Page(number, "", "unverified")
             for number, native in enumerate(text, start=1)]
    for i, page in enumerate(pages):
        if page.verification != "text_match" or not FAILURE_RATE_HEADINGS.search(page.text):
            continue
        try:
            pages[i] = replace(page, words=positioned_words(path, page.number))
        except (ExtractionError, ElementTree.ParseError, KeyError, ValueError) as error:
            print(f"page {page.number}, table coordinates: {error}", file=sys.stderr)
    missing = [page.number for page in pages if not page.text]
    if missing and not engine:
        print("Tesseract is unavailable, scanned pages will be marked unreadable", file=sys.stderr)
    if missing and engine:
        with ThreadPoolExecutor(max_workers=3) as workers:
            pending = {number: workers.submit(scanned_page, path, number, engine) for number in missing}
            for number, task in pending.items():
                pages[number - 1] = task.result()
    if not any(page.text for page in pages):
        raise ExtractionError("No PDF pages could be read")
    return PDF(metadata, pages)
