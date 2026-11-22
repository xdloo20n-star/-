"""Read source-grounded histogram bars and legends from a PDF vector surface."""

from dataclasses import dataclass, replace
from pathlib import Path
import re
import sys
from statistics import median
from xml.etree import ElementTree

from pdf_pages import ExtractionError, Page, Word, positioned_words, run
from vector_scatter import (
    Affine, Box, DECIMAL, POSITION_TOLERANCE, SVG_EXPANSION, TICK_TOLERANCE, calibration, figure_region,
    paired_ticks, rectangle_path, surface_scale, tick_segment, visible_paths, white_rectangle,
)


CAPTION = re.compile(
    r"\b(Figure \d+): Histogram of answer F1 distributions of Wizard of Wikipedia and HotpotQA "
    r"by feeding each individual candidate sentence to the ([\w.-]+) generator\.", re.I,
)
CHART_TITLE = "Answer F1 Distribution Given Single Knowledge Sentence"
METRIC = "Fraction of individual-sentence outputs in F1 bin"
LABELS = (("Wizard of Wikipedia", "WoW"), ("HotpotQA", "HotpotQA"))


@dataclass(frozen=True)
class HistogramBar:
    dataset: str
    number: int
    count: int
    lower: float
    upper: float
    value: float
    bbox: tuple[float, float, float, float]

    @property
    def metric_variant(self) -> str:
        closer = "]" if self.number == self.count else ")"
        return f"bin{self.number} [{self.lower:.5f}, {self.upper:.5f}{closer}"


@dataclass(frozen=True)
class HistogramFigure:
    page: Page
    label: str
    generator: str
    bars: tuple[HistogramBar, ...]


def read_vector_histograms(path: Path, pages: list[Page]) -> tuple[HistogramFigure, ...]:
    figures = []
    for page in pages:
        if page.verification != "text_match" or not (caption := CAPTION.search(page.prose)):
            continue
        figure, generator = caption.groups()
        command = ["pdftocairo", "-f", str(page.number), "-l", str(page.number), "-svg"]
        try:
            words, size = positioned_words(path, page.number)
            ordinary = run([*command, str(path), "-"], timeout=40)
            source = read_histogram_geometry(ordinary, words, size, figure)
            if not source:
                continue
            width, height = (str(round(dimension * SVG_EXPANSION)) for dimension in size)
            expanded = run([*command, "-expand", "-nocenter", "-paperw", width, "-paperh", height, str(path), "-"],
                           timeout=40)
            candidate = read_histogram_geometry(expanded, words, size, figure, expansion=SVG_EXPANSION)
        except (ExtractionError, ElementTree.ParseError, KeyError, ValueError) as error:
            print(f"page {page.number}, histogram: {error}", file=sys.stderr)
            continue
        if len(source) != len(candidate) or any(
            (old.dataset, old.number, old.count) != (new.dataset, new.number, new.count)
            or max(abs(old.lower - new.lower), abs(old.upper - new.upper), abs(old.value - new.value)) > TICK_TOLERANCE
            or any(abs(a - b) * dimension > 1 for a, b, dimension in zip(old.bbox, new.bbox, (*size, *size), strict=True))
            for old, new in zip(source, candidate, strict=True)
        ):
            continue
        # The expanded surface resolves the printed bin boundaries. Keep the
        # evidence coordinates from the source-sized viewport.
        bars = tuple(replace(new, bbox=old.bbox) for old, new in zip(source, candidate, strict=True))
        figures.append(HistogramFigure(page, figure, generator, bars))
    return tuple(figures)


def line_label(words: tuple[Word, ...], text: str) -> Word | None:
    tokens = tuple(text.split())
    hits = []
    for index in range(len(words) - len(tokens) + 1):
        span = words[index:index + len(tokens)]
        if tuple(word.text for word in span) != tokens:
            continue
        if all(abs(word.y - span[0].y) < 1 for word in span) and all(
            0 <= following.x0 - word.x1 < 5 for word, following in zip(span, span[1:])
        ):
            hits.append(span[0])
    return hits[0] if len(hits) == 1 else None


def read_histogram_geometry(svg: str | bytes, words: tuple[Word, ...], size: tuple[float, float], figure: str, *,
                            expansion: int = 1) -> tuple[HistogramBar, ...]:
    root = ElementTree.fromstring(svg)
    scale = surface_scale(root, size, expansion)
    region = figure_region(words, size[0], figure)
    if scale is None or region is None:
        return ()
    local = tuple(word for word in words if region.contains(word.x, word.y))
    title = line_label(local, CHART_TITLE)
    horizontal = line_label(local, "Answer F1 Score")
    vertical = line_label(local, "Fraction")
    if title is None or horizontal is None or vertical is None:
        return ()
    paths = list(visible_paths(root))
    boxes = [box for attrs in paths if (box := white_rectangle(attrs, scale))
             and region.contains(box.x0, box.y0) and region.contains(box.x1, box.y1)
             and title.y < box.y0 < box.y1 < horizontal.y
             and box.x0 <= horizontal.x <= box.x1 and 0 < box.x0 - vertical.x1 < 20
             and box.y0 < vertical.y < box.y1]
    if len(boxes) != 1:
        return ()
    box, = boxes
    ticks = [tick for attrs in paths if (tick := tick_segment(attrs, scale))]
    numeric = tuple(word for word in local if DECIMAL.fullmatch(word.text))
    x_axis = calibration(paired_ticks(box, ticks, numeric, vertical=False), vertical=False)
    y_axis = calibration(paired_ticks(box, ticks, numeric, vertical=True), vertical=True)
    if not x_axis or not y_axis or abs(y_axis.evaluate(box.y1)) > TICK_TOLERANCE:
        return ()
    shapes = [(attrs["fill"], shape) for attrs in paths
              if attrs.get("fill-opacity") == "0.6" and attrs.get("fill", "").startswith("rgb(")
              and (shape := rectangle_path(attrs, scale))
              and box.x0 < shape.x0 < shape.x1 < box.x1 and box.y0 < shape.y0 < shape.y1 <= box.y1]
    columns = [(color, shape) for color, shape in shapes if abs(box.y1 - shape.y1) < POSITION_TOLERANCE]
    legend = [(color, shape) for color, shape in shapes if abs(box.y1 - shape.y1) >= POSITION_TOLERANCE]
    bars = []
    used_colors = set()
    for text, dataset in LABELS:
        label = line_label(local, text)
        if label is None:
            return ()
        swatches = [color for color, shape in legend if
                    0 < label.x0 - shape.x1 < 12 and abs(label.y - (shape.y0 + shape.y1) / 2) < 1.5]
        if len(swatches) != 1 or swatches[0] in used_colors:
            return ()
        color, = swatches
        used_colors.add(color)
        series = histogram_bins(dataset, [shape for fill, shape in columns if fill == color], x_axis, y_axis, size)
        if not series or (bars and bars[0].count != series[0].count):
            return ()
        bars.extend(series)
    if {color for color, _ in columns} != used_colors or max(bar.upper for bar in bars) < 1 - TICK_TOLERANCE:
        return ()
    return tuple(bars)


def histogram_bins(dataset: str, shapes: list[Box], horizontal: Affine, vertical: Affine,
                   size: tuple[float, float]) -> tuple[HistogramBar, ...]:
    if len(shapes) < 2:
        return ()
    shapes.sort(key=lambda box: box.x0)
    origin = shapes[0].x0
    if abs(horizontal.evaluate(origin)) > TICK_TOLERANCE:
        return ()
    span = shapes[-1].x1 - origin
    count = round(span / median(box.x1 - box.x0 for box in shapes))
    if count < len(shapes):
        return ()
    step = span / count
    width, height = size
    bars = []
    for shape in shapes:
        left_index = round((shape.x0 - origin) / step)
        if any(abs(edge - origin - index * step) > POSITION_TOLERANCE for edge, index in
               ((shape.x0, left_index), (shape.x1, left_index + 1))):
            return ()
        lower, upper = horizontal.evaluate(shape.x0), horizontal.evaluate(shape.x1)
        value = vertical.scale * (shape.y0 - shape.y1)
        if (not -TICK_TOLERANCE <= lower < upper <= 1 + TICK_TOLERANCE or not 0 < value <= 1
                or (bars and bars[-1].number >= left_index + 1)):
            return ()
        bars.append(HistogramBar(dataset, left_index + 1, count, max(0, lower), min(1, upper), value,
                                 (shape.x0 / width, shape.y0 / height, shape.x1 / width, shape.y1 / height)))
    # The evidence is a set of filled rectangles. Keep gaps in the grid as
    # gaps, while confirming the displayed fractions cover the distribution.
    return tuple(bars) if abs(sum(bar.value for bar in bars) - 1) <= 0.001 else ()
