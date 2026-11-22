"""Read bounded regions of scanned tables while retaining page coordinates."""

from dataclasses import dataclass, replace
from itertools import groupby
import math
import os
import re

from pdf_pages import ExtractionError, ScanSource, Word, as_text, run


@dataclass(frozen=True)
class ScanImage:
    width: int
    height: int
    pixels: bytes
    dpi: int
    page_width: int
    page_height: int
    left: int = 0
    top: int = 0
    slope: float = 0.0
    offset: float = 0.0

    def crop(self, x0: int, y0: int, x1: int, y1: int) -> "ScanImage":
        if not (0 <= x0 < x1 <= self.width and 0 <= y0 < y1 <= self.height):
            raise ValueError("OCR crop outside the raster")
        pixels = b"".join(self.pixels[y*self.width+x0:y*self.width+x1] for y in range(y0, y1))
        return ScanImage(x1 - x0, y1 - y0, pixels, self.dpi, self.page_width, self.page_height,
                         self.left + x0, self.top + y0, self.slope, self.offset + x0 * self.slope)

    def source_bbox(self, x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        first = self.offset + x0 * self.slope
        last = self.offset + x1 * self.slope
        return (
            max(0, self.left + x0) / self.page_width,
            max(0, self.top + y0 + min(first, last) - 1) / self.page_height,
            min(self.page_width, self.left + x1) / self.page_width,
            min(self.page_height, self.top + y1 + max(first, last) + 1) / self.page_height,
        )


def scan_region(source: ScanSource, bounds: tuple[float, float, float, float], dpi: int = 480) -> ScanImage:
    number = str(source.number)
    data = run(["pdftoppm", "-gray", "-f", number, "-l", number, "-singlefile",
                "-r", str(dpi), str(source.path)], timeout=60)
    header = re.match(rb"P5\s+(\d+)\s+(\d+)\s+255\n", data)
    if header is None:
        raise ExtractionError("Unexpected grayscale raster from Poppler")
    width, height = map(int, header.groups())
    pixels = data[header.end():]
    if len(pixels) != width * height:
        raise ExtractionError("Incomplete grayscale raster from Poppler")
    page = ScanImage(width, height, pixels, dpi, width, height)
    x0, y0, x1, y1 = bounds
    return page.crop(math.floor(x0 * width), math.floor(y0 * height),
                     math.ceil(x1 * width), math.ceil(y1 * height))


def deskew(image: ScanImage) -> ScanImage:
    """Align horizontal rules using a small shear, keeping the inverse transform."""
    width, height = image.width, image.height
    if image.slope:
        raise ValueError("The scan region is already deskewed")
    pixels = image.pixels
    samples = [(x, [y for y in range(0, height, 2) if pixels[y * width + x] < 120])
               for x in range(0, width, max(1, width // 140))]
    margin = math.ceil(width * 0.025) + 3

    def score(step: int) -> int:
        counts = [0] * (height + 2 * margin)
        for x, ys in samples:
            offset = margin - round((x - width / 2) * step / 1000)
            for y in ys:
                counts[y + offset] += 1
        return sum(sorted((sum(counts[y:y+5]) for y in range(len(counts) - 4)), reverse=True)[:8])

    # Prefer the unrotated raster when the horizontal projection ties.
    step = max(range(-50, 51), key=lambda candidate: (score(candidate), -abs(candidate)))
    if step == 0:
        return image
    slope = step / 1000
    offset = -width / 2 * slope
    shifted = bytearray(b"\xff") * len(pixels)
    for delta, group in groupby(range(width), key=lambda x: round(slope * x + offset)):
        start = next(group)
        end = max(group, default=start) + 1
        for y in range(max(0, -delta), min(height, height - delta)):
            source = (y + delta) * width
            target = y * width
            shifted[target+start:target+end] = pixels[source+start:source+end]
    return ScanImage(width, height, bytes(shifted), image.dpi, image.page_width, image.page_height,
                     image.left, image.top, slope, offset)


def contrast(image: ScanImage, white: int) -> ScanImage:
    palette = bytes(min(255, max(0, round((value - 45) * 255 / (white - 45)))) for value in range(256))
    return replace(image, pixels=image.pixels.translate(palette))


def recognize(image: ScanImage, engine: str, psm: int = 6, digits_only: bool = False) -> tuple[Word, ...]:
    env = dict(os.environ)
    env.setdefault("OMP_THREAD_LIMIT", "1")
    command = [engine, "stdin", "stdout", "-l", env.get("OCR_LANG", "eng"),
               "--dpi", str(image.dpi), "--psm", str(psm)]
    if digits_only:
        command += ["-c", "tessedit_char_whitelist=0123456789.,%+-()"]
    data = f"P5\n{image.width} {image.height}\n255\n".encode("ascii") + image.pixels
    words = []
    for line in as_text(run([*command, "tsv"], data=data, timeout=60, env=env)).splitlines():
        fields = line.split("\t", 11)
        if len(fields) != 12 or fields[0] != "5" or not fields[-1].strip():
            continue
        x, y, width, height = map(int, fields[6:10])
        if width > 0 and height > 0:
            words.append(Word(fields[-1].strip(), x, y, x + width, y + height))
    return tuple(words)
