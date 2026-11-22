"""Calibrate matched scatter panels from vector markers and printed linear axes."""

from dataclasses import dataclass
import math
from pathlib import Path
import re
import sys
from xml.etree import ElementTree

from pdf_pages import ExtractionError, Page, Word, positioned_words, run


CAPTION = re.compile(
    r"\b(?P<label>Figure \d+): Scatter plot of answer F1 versus the knowledge "
    r"(?P<metric>precision|recall|F1) for (?P<top>[\w.-]+) \(top\), "
    r"(?P<middle>[\w. -]+?) \(middle\), and (?P<bottom>[\w. -]+?) \(bottom\); "
    r"the left column shows results on (?P<left>[\w-]+), and the right shows (?P<right>[\w-]+)\. "
    r"Each data point corresponds to a full experiment on the entire sampled dataset\.", re.I,
)
LENGTH_CAPTION = re.compile(
    r"\b(Figure \d+): Scatter plot of the number of selected knowledge sentences versus the knowledge precision on "
    r"(Wizard of Wikipedia) \(left\) and (HotpotQA) \(right\)\. Each data point corresponds to a full experiment "
    r"on the entire sampled dataset\.", re.I,
)
CAPPED_CAPTION = re.compile(
    r"\b(Figure \d+): Scatter plot of answer F1 versus the knowledge precision for ([\w.-]+) on "
    r"(Wizard of Wikipedia) \(left\) and (HotpotQA) \(right\) by limiting the number of knowledge sentences up to k\s*=\s*(\d+)\.",
    re.I,
)
SVG = "{http://www.w3.org/2000/svg}"
NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
COORD = re.compile(NUMBER)
RECTANGLE = re.compile(r"M (?:\S+ \S+ L ){3}\S+ \S+ Z(?: M \S+ \S+)?\s*")
LINE = re.compile(r"M (\S+) (\S+) L (\S+) (\S+)\s*")
MATRIX = re.compile(r"matrix\(([^()]*)\)")
DECIMAL = re.compile(r"\d+\.\d+")
BLACK = "rgb(0%, 0%, 0%)"
WHITE = "rgb(100%, 100%, 100%)"
METRICS = ("precision", "recall", "f1")
POSITION_TOLERANCE = 0.015
TICK_TOLERANCE = 0.0001
SVG_EXPANSION = 100


@dataclass(frozen=True)
class ScatterPoint:
    x: float
    y: float
    bbox: tuple[float, float, float, float]


PageGeometry = tuple[tuple[ScatterPoint, ...], ...]


@dataclass(frozen=True)
class ScatterPanel:
    dataset: str
    generator: str | None
    points: tuple[ScatterPoint, ...]


@dataclass(frozen=True)
class ScatterFigure:
    page: Page
    label: str
    metric: str
    panels: tuple[ScatterPanel, ...]
    vertical_metric: str = "Answer F1"
    condition: str = "unconstrained knowledge sampling"


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, x: float, y: float) -> bool:
        return self.x0 < x < self.x1 and self.y0 < y < self.y1


@dataclass(frozen=True)
class Affine:
    offset: float
    scale: float

    def evaluate(self, point: float) -> float:
        return self.offset + self.scale * point


def read_vector_scatter(path: Path, pages: list[Page]) -> tuple[ScatterFigure, ...]:
    selected = [(page, caption) for page in pages if page.verification == "text_match"
                if (caption := CAPTION.search(page.prose))]
    if tuple(caption["metric"].casefold() for _, caption in selected) != METRICS:
        return ()
    conditions = [tuple(caption[key] for key in ("top", "middle", "bottom", "left", "right"))
                  for _, caption in selected]
    if len(set(conditions)) != 1:
        return ()
    return (
        read_matched_figures(path, selected, conditions[0], refine=True)
        or read_matched_figures(path, selected, conditions[0], refine=False)
    )


def read_matched_figures(path: Path, selected: list[tuple[Page, re.Match]], conditions: tuple[str, ...],
                         *, refine: bool) -> tuple[ScatterFigure, ...]:
    top, middle, bottom, left, right = conditions
    figures = []
    for page, caption in selected:
        geometry = read_page_geometry(path, page, refine=refine)
        if len(geometry) != 6:
            return ()
        panels = tuple(ScatterPanel(dataset, generator, points)
                       for (generator, dataset), points in zip(
                           ((generator, dataset) for generator in (top, middle, bottom) for dataset in (left, right)),
                           geometry, strict=True,
                       ))
        figures.append(ScatterFigure(page, caption["label"], f"Knowledge {caption['metric']}", panels))
    first = figures[0]
    for figure in figures[1:]:
        for reference, other in zip(first.panels, figure.panels, strict=True):
            # Equal printed precision alone cannot distinguish two nearby markers.
            # Compare ordered values before rounding, within vector conversion error.
            if len(reference.points) != len(other.points) or any(
                abs(a.y - b.y) > 0.000005 or f"{a.y:.4f}" != f"{b.y:.4f}"
                for a, b in zip(reference.points, other.points, strict=True)
            ):
                return ()
    return tuple(figures)


def read_vector_length_scatter(path: Path, pages: list[Page]) -> tuple[ScatterFigure, ...]:
    """Read the precision and recall axes in the length and capped plots."""
    for page in pages:
        if page.verification != "text_match":
            continue
        length = LENGTH_CAPTION.search(page.prose)
        capped = CAPPED_CAPTION.search(page.prose)
        if not length or not capped or length.start() >= capped.start() or length.groups()[1:] != capped.groups()[2:4]:
            continue
        # On these figures the outcome described in the caption is displayed
        # by color. The vertical axis is instead explicitly labeled Recall.
        if any(len(re.findall(rf"\bKnowledge {axis}\b", page.text)) != 4 for axis in ("Precision", "Recall")):
            continue
        geometry = read_page_geometry(path, page, marker_opacity="0.8")
        if len(geometry) != 4 or tuple(map(len, geometry[:2])) != tuple(map(len, geometry[2:])):
            continue
        datasets = length.groups()[1:]
        pairs = (
            (length[1], None, "knowledge selection length analysis", geometry[:2]),
            (capped[1], capped[2], f"knowledge selection capped at {capped[5]} sentences", geometry[2:]),
        )
        return tuple(ScatterFigure(page, label, "Knowledge precision", tuple(
            ScatterPanel(dataset, generator, points) for dataset, points in zip(datasets, sections, strict=True)
        ), "Knowledge recall", condition) for label, generator, condition, sections in pairs)
    return ()


def read_page_geometry(path: Path, page: Page, *, marker_opacity: str = "0.5", refine: bool = True) -> PageGeometry:
    command = ["pdftocairo", "-f", str(page.number), "-l", str(page.number), "-svg"]
    try:
        words, size = positioned_words(path, page.number)
        svg = run([*command, str(path), "-"], timeout=40)
        ordinary = read_panel_geometry(svg, words, size, marker_opacity=marker_opacity)
    except (ExtractionError, ElementTree.ParseError, KeyError, ValueError) as error:
        print(f"page {page.number}, scatter plot: {error}", file=sys.stderr)
        return ()
    if not ordinary or not refine:
        return ordinary
    try:
        width, height = (str(round(dimension * SVG_EXPANSION)) for dimension in size)
        # Cairo rounds path coordinates on its output surface. A larger vector
        # surface lowers that error before the axes and markers are calibrated.
        enlarged = run([*command, "-expand", "-nocenter", "-paperw", width, "-paperh", height, str(path), "-"],
                       timeout=40)
        candidate = read_panel_geometry(enlarged, words, size, marker_opacity=marker_opacity, expansion=SVG_EXPANSION)
    except (ExtractionError, ElementTree.ParseError, KeyError, ValueError):
        return ordinary
    if tuple(map(len, candidate)) != tuple(map(len, ordinary)):
        return ordinary
    bounds = (*size, *size)
    refined = []
    for old_panel, new_panel in zip(ordinary, candidate, strict=True):
        panel = []
        for old, new in zip(old_panel, new_panel, strict=True):
            if (max(abs(old.x - new.x), abs(old.y - new.y)) > TICK_TOLERANCE
                    or any(abs(a - b) * dimension > 1
                           for a, b, dimension in zip(old.bbox, new.bbox, bounds, strict=True))):
                return ordinary
            # Keep the citation in the original PDF page's coordinate space.
            panel.append(ScatterPoint(new.x, new.y, old.bbox))
        refined.append(tuple(panel))
    return tuple(refined)


def visible_paths(svg: ElementTree.Element):
    for child in svg:
        if child.tag == SVG + "path":
            yield child.attrib
        elif child.tag == SVG + "g" and "transform" not in child.attrib:
            yield from visible_paths(child)


def matrix(attributes: dict[str, str]) -> tuple[float, ...] | None:
    if "transform" not in attributes:
        return 1, 0, 0, 1, 0, 0
    match = MATRIX.fullmatch(attributes["transform"])
    if not match:
        return None
    values = tuple(map(float, match[1].replace(",", " ").split()))
    return values if len(values) == 6 and all(map(math.isfinite, values)) else None


def white_rectangle(attributes: dict[str, str], scale: float) -> Box | None:
    path = attributes.get("d", "")
    if attributes.get("fill") != WHITE or attributes.get("fill-opacity") != "1" or "transform" in attributes:
        return None
    if not RECTANGLE.fullmatch(path):
        return None
    coordinates = tuple(map(float, COORD.findall(path)))[:8]
    xs, ys = set(coordinates[::2]), set(coordinates[1::2])
    if len(xs) != 2 or len(ys) != 2 or len(set(zip(coordinates[::2], coordinates[1::2]))) != 4:
        return None
    return Box(min(xs) / scale, min(ys) / scale, max(xs) / scale, max(ys) / scale)


def tick_segment(attributes: dict[str, str], scale: float) -> tuple[float, float, float, float] | None:
    if attributes.get("stroke") != BLACK or attributes.get("fill") != "none":
        return None
    marker = LINE.fullmatch(attributes.get("d", ""))
    transform = matrix(attributes)
    if not marker or not transform:
        return None
    x0, y0, x1, y1 = map(float, marker.groups())
    a, b, c, d, e, f = transform
    points = ((a*x0 + c*y0 + e) / scale, (b*x0 + d*y0 + f) / scale,
              (a*x1 + c*y1 + e) / scale, (b*x1 + d*y1 + f) / scale)
    if not 0.5 <= math.hypot(points[2] - points[0], points[3] - points[1]) <= 4:
        return None
    return points


def scatter_marker(attributes: dict[str, str], opacity: str, scale: float) -> tuple[float, float, float] | None:
    if attributes.get("stroke-opacity") != opacity:
        return None
    if attributes.get("fill") != "none":
        if attributes.get("fill") != attributes.get("stroke") or attributes.get("fill-opacity") != opacity:
            return None
    path = attributes.get("d", "")
    if re.findall(r"[A-Z]", path) != ["M", *["C"] * 8, "Z", "M"]:
        return None
    transform = matrix(attributes)
    if transform is None:
        return None
    a, b, c, d, x, y = transform
    if b or c or not math.isclose(abs(a), abs(d)):
        return None
    positions = tuple(map(float, COORD.findall(path)))
    radius = abs(a) * max(map(abs, positions)) / scale
    if not 0.3 <= radius <= 2:
        return None
    return x / scale, y / scale, radius


def paired_ticks(box: Box, lines: list[tuple[float, float, float, float]], words: tuple[Word, ...], *,
                 vertical: bool) -> list[tuple[float, float]]:
    result = []
    for x0, y0, x1, y1 in lines:
        if vertical:
            if not (abs(y0 - y1) < POSITION_TOLERANCE and abs(max(x0, x1) - box.x0) < POSITION_TOLERANCE
                    and box.y0 < y0 < box.y1):
                continue
            matched = [word for word in words if 0 < min(x0, x1) - word.x1 < 6 and abs(y0 - word.y) < 1.5]
        else:
            if not (abs(x0 - x1) < POSITION_TOLERANCE and abs(min(y0, y1) - box.y1) < POSITION_TOLERANCE
                    and box.x0 < x0 < box.x1):
                continue
            matched = [word for word in words if 1 < word.y - min(y0, y1) < 8 and abs(x0 - word.x) < 1]
        if len(matched) != 1:
            return []
        word = matched[0]
        result.append((y0 if vertical else x0, float(word.text)))
    return sorted(result)


def calibration(ticks: list[tuple[float, float]], *, vertical: bool) -> Affine | None:
    if len(ticks) < 3:
        return None
    start, lower = ticks[0]
    stop, upper = ticks[-1]
    if start >= stop or not 0 <= min(lower, upper) < max(lower, upper) <= 1:
        return None
    scale = (upper - lower) / (stop - start)
    mapping = Affine(lower - scale * start, scale)
    if (scale >= 0) == vertical or any(abs(mapping.evaluate(point) - value) > TICK_TOLERANCE
                                      for point, value in ticks):
        return None
    return mapping


def read_panel_geometry(svg: str | bytes, words: tuple[Word, ...], size: tuple[float, float], *,
                        marker_opacity: str = "0.5", expansion: int = 1) -> PageGeometry:
    root = ElementTree.fromstring(svg)
    view = tuple(map(float, root.attrib.get("viewBox", "").split()))
    if root.tag != SVG + "svg" or len(view) != 4 or view[:2] != (0, 0) or any(
        not abs(svg_size - pdf_size * expansion) <= 1 for svg_size, pdf_size in zip(view[2:], size, strict=True)
    ):
        return ()
    scale = min(view[2] / size[0], view[3] / size[1]) if expansion > 1 else 1
    paths = list(visible_paths(root))
    boxes = [box for attributes in paths if (box := white_rectangle(attributes, scale))]
    ticks = [tick for attributes in paths if (tick := tick_segment(attributes, scale))]
    markers = [marker for attributes in paths if (marker := scatter_marker(attributes, marker_opacity, scale))]
    numeric = tuple(word for word in words if DECIMAL.fullmatch(word.text))
    panels = []
    width, height = size
    for box in sorted(boxes, key=lambda box: (round(box.y0, 1), box.x0)):
        x_axis = calibration(paired_ticks(box, ticks, numeric, vertical=False), vertical=False)
        y_axis = calibration(paired_ticks(box, ticks, numeric, vertical=True), vertical=True)
        if x_axis is None or y_axis is None:
            continue
        points = []
        for x, y, radius in markers:
            if not box.contains(x, y):
                continue
            scores = x_axis.evaluate(x), y_axis.evaluate(y)
            if any(not -TICK_TOLERANCE <= score <= 1 + TICK_TOLERANCE for score in scores):
                return ()
            # Include the stroke and the sub-point displacement introduced by SVG conversion.
            radius += 0.7
            x0, y0, x1, y1 = (x - radius) / width, (y - radius) / height, (x + radius) / width, (y + radius) / height
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                return ()
            points.append(ScatterPoint(*(min(1, max(0, score)) for score in scores), (x0, y0, x1, y1)))
        panels.append(tuple(points))
    if not all(panels) or sum(map(len, panels)) != len(markers):
        return ()
    return tuple(panels)
