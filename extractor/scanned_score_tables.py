"""Read scanned model-score grids with distinct cells for mean and deviation."""

from collections import Counter
from dataclasses import dataclass, replace
import re
import sys

from pdf_pages import ExtractionError, Page, Word
from rag_tables import Column, Header, Table
from scan_ocr import ScanImage, contrast, deskew, recognize, scan_region


GENERAL = "General Cybersecurity Questions"
MULTI_HOP = "Multi-Hop Cybersecurity Questions"
CVE = "CVE Questions"
FRACTION = re.compile(r"[01]\.\d{2}")
FRACTION_3 = re.compile(r"[01]\.\d{3}")
PERCENT = re.compile(r"\d{1,3}%")
RUBRIC = re.compile(r"[1-5]\.\d{2,3}|-")
MODEL = re.compile(r"MoRSE|GPT-4 0125(?:-Preview)?|MIXTRAL(?: 7X8)?|HACKERGPT|GEMINI(?: 1\.0 Pro)?", re.I)
CAPTION_II = re.compile(
    r"\b(TABLE II): Comparative Assessment of RAG Models on \d+ General and \d+ Multi-Hop Cybersecurity Questions",
    re.I,
)
CAPTION_III = re.compile(r"\b(TABLE III): Performance comparison of models on \d+ CVE", re.I)
CAPTION_IV = re.compile(r"\b(TABLE IV): Fraction of Wins \(No Ties\) of Each Combination of Models", re.I)
CAPTION_V = re.compile(r"\b(TABLE V): Elo Ratings, MLE Ratings, and Bootstrap Ratings", re.I)
CAPTION_VI = re.compile(r"\b(TABLE VI): Scores Across All Cybersecurity Questions types", re.I)
CAPTION_VII = re.compile(r"\b(TABLE VII):", re.I)
ELO = re.compile(r"\d{3,4}")
DECIMAL_ELO = re.compile(r"\d{3,4}\.\d{2}")
BOOTSTRAP = re.compile(r"(\d{3,4}\.\d{2})\s*\((\d{3,4})-(\d{3,4})\)")
TIME = re.compile(r"\d{1,3}\.\d{1,3}")
COMPONENT = re.compile(r"[\w-]+ Retriever|Question Ret\. Sys\.|[\w-]+ BUFFERS|CONTEXT TRANSF\.", re.I)
COMPONENT_BOUNDS = (.085, .075, .895, .339)
COMPONENT_ROWS = (546, 602, 658, 716, 770, 827, 884, 1063, 1119, 1174, 1232, 1290)
COMPONENT_SIZE = re.compile(r"(\d+(?:\.\d+)?)(MB|GB)")
DOCUMENT_COUNT = re.compile(r"[~∼](\d+)")
WIN_HEADING = re.compile(r"(.+?) General Qs Mul[ .-]+Hop [QO]s", re.I)
WIN_PERCENT = re.compile(r"(?:100|[1-9]?\d)%")


@dataclass(frozen=True)
class Grid:
    bounds: tuple[float, float, float, float]
    body: tuple[float, float, float, float]
    # Reference geometry in the bounded 480 dpi raster. Header guards and
    # the observed row labels and cells are required in these regions.
    columns: tuple[int, ...]
    rows: tuple[int, ...]
    headings: tuple[str, ...]
    formats: tuple[re.Pattern[str], ...]


PAIRED = Grid(
    (.13, .072, .88, .224), (.07, .505, .925, .91),
    (902, 1064, 1228, 1392, 1553, 1720, 1886, 2050, 2214, 2378, 2542, 2708),
    (449, 507, 563, 619, 677),
    ("Model", "Relevance", "Similarity", "Correctness", "Relevance", "Similarity", "Correctness"),
    (FRACTION,) * 12,
)
CVE_GRID = Grid(
    (.10, .574, .487, .688), (.02, .70, .97, .92),
    (625, 813, 1002, 1189, 1425), (465, 523),
    ("Model", "Similarity", "Correctness", "Accuracy"), (FRACTION_3,) * 4 + (PERCENT,),
)
JUDGE_GRID = Grid(
    (.095, .34, .447, .452), (.12, .40, .977, .906),
    (679, 1010, 1308), (274, 331, 389, 446, 504),
    ("Model", "General", "Multi-Hop", "CVE"), (RUBRIC,) * 3,
)


def fragment(page: Page, pattern: str) -> tuple[Page, str] | None:
    for text in (page.reading_text, page.text):
        if match := re.search(pattern, text, re.I):
            return page, match[0]
    return None


def metric_columns(page: Page, definitions: Page, grid: Grid) -> list[Column]:
    spec = {
        "Relevance": ("answer relevance", ("RAGAS; N=3 generated questions; cosine to original question",
                                           (r"\bN\s+equal\s+to\s+3\b", r"cosine\s+similarity"))),
        "Similarity": ("answer similarity", ("RAGAS; cosine to reference answer",
                                             (r"\bAnswer\s+Similarity\b", r"prede[\s-]*fined correct answers"))),
        "Correctness": ("answer correctness", ("RAGAS; 0.75 factual F1 + 0.25 semantic similarity",
                                               (r"\bAnswer\s+Correctness\b", r"0\.75\s+and\s+0\.25"))),
        "Accuracy": ("Accuracy", ("correct identification of vulnerability",
                                  (r"(?:For\s+CVEs,\s+accuracy\s+is\s+straightforward:\s+we\s+check\s+if\s+the\s+model"
                                   r"\s+identified\s+the\s+correct\s+vulnerability|checking\s+whether\s+the\s+models"
                                   r"\s+correctly\s+identified\s+the\s+vulnerability)",))),
    }
    if grid is PAIRED:
        cells = [(dataset, name, agg) for dataset in (GENERAL, MULTI_HOP)
                 for name in ("Relevance", "Similarity", "Correctness") for agg in ("mean", "std")]
    else:
        cells = [(CVE, name, agg) for name in ("Similarity", "Correctness") for agg in ("mean", "std")]
        cells.append((CVE, "Accuracy", None))
    columns = []
    for dataset, label, agg in cells:
        name, (variant, patterns) = spec[label]
        source = page if label == "Accuracy" else definitions
        evidence = [fragment(source, pattern) for pattern in patterns]
        confirmed = all(evidence) and (label == "Accuracy" or bool(fragment(definitions, r"RAGAS\s+framework")))
        variant_evidence = evidence[0] if confirmed else None
        if confirmed and label == "Accuracy":
            variant_evidence = fragment(source, r"\bidentified\s+the(?:\s+correct)?\s+vulnerability\b")
        columns.append(Column(
            Header(label, name, 0, None, "%" if label == "Accuracy" else None),
            dataset, agg, direction="higher_is_better" if agg != "std" else None,
            name_evidence=fragment(source, rf"\b{re.escape(name)}\b"),
            metric_variant=variant if confirmed else None,
            metric_variant_evidence=variant_evidence,
            aggregation_evidence=(fragment(definitions, r"mean\s*\([^)]*\)\s+and\s+standard deviation")
                                  if agg else None),
        ))
    return columns


def cropped_words(image: ScanImage, engine: str, bounds: tuple[float, float, float, float]) -> tuple[Word, ...]:
    x0, y0, x1, y1 = (round(value * axis) for value, axis in
                      zip(bounds, (image.width, image.height, image.width, image.height), strict=True))
    return tuple(Word(w.text, w.x0 + x0, w.y0 + y0, w.x1 + x0, w.y1 + y0)
                 for w in recognize(image.crop(x0, y0, x1, y1), engine))


def reread_cell(image: ScanImage, engine: str, x: int, y: int,
                pattern: re.Pattern[str]) -> tuple[str, tuple[float, float, float, float]] | None:
    selections = []
    for width, height, psm, white in ((54, 25, 7, 193), (62, 24, 8, 220)):
        crop = image.crop(x - width, y - height, x + width, y + height)
        words = recognize(contrast(crop, white), engine, psm=psm, digits_only=True)
        if len(words) != 1 or not pattern.fullmatch(words[0].text):
            return None
        word = words[0]
        selections.append((word.text, crop.source_bbox(word.x0, word.y0, word.x1, word.y1)))
    return selections[0] if selections[0][0] == selections[1][0] else None


def read_cell_view(image: ScanImage, engine: str, bounds: tuple[int, int, int, int],
                   pattern: re.Pattern[str], threshold: int | None):
    x0, y0, x1, y1 = bounds
    crop = image.crop(x0, y0, x1, y1)
    pixels = crop.pixels
    if threshold is not None:
        pixels = pixels.translate(bytes(0 if value < threshold else 255 for value in range(256)))
    # A white margin prevents line OCR from treating an edge of the crop as a glyph.
    pad = 15
    margin = b"\xff" * pad
    width = crop.width + 2 * pad
    border = b"\xff" * (width * pad)
    padded = border + b"".join(margin + pixels[start:start + crop.width] + margin
                               for start in range(0, len(pixels), crop.width)) + border
    words = recognize(replace(crop, width=width, height=crop.height + 2 * pad, pixels=padded),
                      engine, psm=7, digits_only=True)
    raw = "".join(word.text for word in sorted(words, key=lambda word: word.x))
    if not pattern.fullmatch(raw):
        return None
    left, top = min(w.x0 for w in words) - pad, min(w.y0 for w in words) - pad
    right, bottom = max(w.x1 for w in words) - pad, max(w.y1 for w in words) - pad
    if not (0 <= left < right <= crop.width and 0 <= top < bottom <= crop.height):
        return None
    return raw, crop.source_bbox(left, top, right, bottom)


def confirm_cell(image: ScanImage, engine: str, bounds: tuple[int, int, int, int],
                 pattern: re.Pattern[str], regional: Word | None = None):
    x0, y0, x1, y1 = bounds
    readings = [read_cell_view(image, engine, bounds, pattern, 157),
                read_cell_view(image, engine, (x0 - 12, y0 - 1, x1 + 12, y1 + 1), pattern, 168),
                read_cell_view(image, engine, (x0 - 3, y0, x1 + 3, y1), pattern, None)]
    if regional and pattern.fullmatch(regional.text):
        readings.append((regional.text, image.source_bbox(regional.x0, regional.y0, regional.x1, regional.y1)))
    counts = Counter(cell[0] for cell in readings if cell).most_common(2)
    if not counts or counts[0][1] < 2 or (len(counts) == 2 and counts[0][1] == counts[1][1]):
        return None
    return next(cell for cell in readings if cell and cell[0] == counts[0][0])


def scan_grid(page: Page, grid: Grid):
    source = page.scan
    if source is None:
        return None
    image = deskew(scan_region(source, grid.bounds))
    headings = recognize(image, source.engine)
    header_words = [word.text.casefold() for word in sorted(headings, key=lambda word: word.x)
                    if image.height * .30 < word.y < image.height * .67 and word.y < grid.rows[0] - 45]
    at = iter(header_words)
    if not all(heading.casefold() in at for heading in grid.headings):
        return None
    words = cropped_words(image, source.engine, grid.body)
    # Read labels independently from the value columns. Unrecognized values
    # stay empty and one unreadable cell cannot shift a mean/SD pair.
    rows = []
    boxes = []
    for y in grid.rows:
        label = " ".join(word.text for word in sorted(words, key=lambda word: word.x)
                         if abs(word.y - y) < 25 and word.x1 < grid.columns[0] - 45)
        if not MODEL.fullmatch(label) or label in {name for name, _ in rows}:
            return None
        values = []
        row_boxes = []
        for x, pattern in zip(grid.columns, grid.formats, strict=True):
            candidates = [word for word in words if abs(word.x - x) < 38 and abs(word.y - y) < 24]
            good = [word for word in candidates if pattern.fullmatch(word.text)]
            if len(candidates) == 1 and len(good) == 1:
                word = good[0]
                cell = (word.text, image.source_bbox(word.x0, word.y0, word.x1, word.y1))
            else:
                cell = reread_cell(image, source.engine, x, y, pattern)
                if cell is None and grid is PAIRED and len(candidates) <= 1:
                    alternatives = [word for word in headings if abs(word.x - x) < 38 and abs(word.y - y) < 24]
                    regional = alternatives[0] if len(alternatives) == 1 else None
                    center = round(candidates[0].y if candidates else regional.y if regional else y)
                    cell = confirm_cell(image, source.engine, (x - 58, center - 24, x + 58, center + 24),
                                        pattern, regional)
            values.append(cell[0] if cell else None)
            row_boxes.append(cell[1] if cell else None)
        rows.append((label, values))
        boxes.append(tuple(row_boxes))
    return rows, tuple(boxes)


def bootstrap_cell(image: ScanImage, engine: str, words: tuple[Word, ...],
                   left: int, right: int, y: int):
    x0, y0, x1, y1 = left, y - 27, right, y + 27
    cell_words = sorted((word for word in words if left < word.x < right and abs(word.y - y) < 26),
                        key=lambda word: word.x)
    regional = BOOTSTRAP.fullmatch(" ".join(word.text for word in cell_words))
    crop = image.crop(x0, y0, x1, y1)
    primary = BOOTSTRAP.fullmatch("".join(word.text for word in recognize(crop, engine, 7, digits_only=True)))
    if primary is None:
        return None
    values = primary.groups()
    if regional is None or regional.groups() != values:
        secondary = BOOTSTRAP.fullmatch("".join(word.text for word in recognize(
            image.crop(x0 + 4, y0 + 2, x1 - 4, y1 - 2), engine, 8, digits_only=True,
        )))
        if secondary is None or secondary.groups() != values:
            return None
    return (values, image.source_bbox(x0, y0, x1, y1)) if ordered_interval(values) else None


def ordered_interval(values: tuple[str, str, str]) -> bool:
    center, upper, lower = values
    return int(lower) <= float(center) <= int(upper)


def win_cell(image: ScanImage, engine: str, x: int, y: int, regional: Word | None):
    readings = []
    for width, height, white in ((57, 23, 173), (73, 25, 220)):
        crop = image.crop(x - width, y - height, x + width, y + height)
        words = recognize(contrast(crop, white), engine, psm=7, digits_only=True)
        if len(words) == 1 and WIN_PERCENT.fullmatch(words[0].text):
            word = words[0]
            readings.append((word.text, crop.source_bbox(word.x0, word.y0, word.x1, word.y1)))
    if len(readings) == 2 and readings[0][0] == readings[1][0]:
        return readings[0]
    return confirm_cell(image, engine, (x - 58, y - 24, x + 58, y + 24), WIN_PERCENT, regional)


def win_table(page: Page, label: str, caption: str, judge: tuple[Page, str]) -> Table | None:
    source = page.scan
    image = deskew(scan_region(source, (.10, .075, .92, .208)))
    positions = ((872, 2), (1168, 3), (1464, -1), (1761, -2),
                 (2056, -3), (2353, 5), (2651, 12), (2946, 20))
    opponents = []
    for left, right in ((720, 1300), (1320, 1900), (1930, 2500), (2540, 3100)):
        heading = " ".join(w.text for w in recognize(image.crop(left, 238, right, 364), source.engine))
        match = WIN_HEADING.fullmatch(heading)
        if not match or not MODEL.fullmatch(match[1]) or match[1].casefold() in {n.casefold() for n in opponents}:
            return None
        opponents.append(match[1])
    words = cropped_words(image, source.engine, (.055, .535, .94, .885))
    row_centers = (407, 466, 522, 580)
    labels = [" ".join(w.text for w in sorted(words, key=lambda w: w.x)
                       if w.x1 < 730 and abs(w.y - y) < 24) for y in row_centers]
    opponent_keys = [opponent.casefold() for opponent in opponents]
    if (not MODEL.fullmatch(labels[0]) or labels[0].casefold() in opponent_keys
            or [label.casefold() for label in labels[1:]] != opponent_keys[:-1]):
        return None
    columns = []
    for opponent in opponents:
        for dataset, printed in ((GENERAL, "General Qs"), (MULTI_HOP, "Mul.Hop Qs")):
            columns.append(Column(
                Header(printed, "pairwise win fraction", 0, None, "%"), dataset, direction="higher_is_better",
                source_label=f"{opponent} / {printed}",
                metric_variant=f"reference-guided GPT-4 judging; opponent {opponent}; ties excluded",
                metric_variant_evidence=judge,
            ))
    rows, boxes = [], []
    for row, (model, y) in enumerate(zip(labels, row_centers, strict=True)):
        values, row_boxes = [None] * len(columns), [None] * len(columns)
        for index, (x, offset) in enumerate(positions):
            if index // 2 < row:
                continue
            center = y + offset
            candidates = [word for word in words if abs(word.x - x) < 38 and abs(word.y - center) < 21]
            cell = win_cell(image, source.engine, x, center, candidates[0] if len(candidates) == 1 else None)
            if cell:
                values[index], row_boxes[index] = cell
        rows.append((model, values))
        boxes.append(tuple(row_boxes))
    return Table(page, label, caption, 0, columns, rows, family="scanned_scores", value_bboxes=tuple(boxes))


def rank_table(page: Page, label: str, caption: str, interval_evidence: tuple[Page, str] | None) -> Table | None:
    source = page.scan
    image = deskew(scan_region(source, (.126, .204, .88, .337)))
    heading_words = recognize(image, source.engine)
    heading = [w.text for w in heading_words if 235 < w.y < 308]
    if heading.count("Elo") != 6 or heading.count("Bootstrap") != 3 or "Model" not in heading:
        return None
    labels = cropped_words(image, source.engine, (.021, .44, .127, .882))
    body = cropped_words(image, source.engine, (.112, .449, .985, .886))
    datasets = (GENERAL, MULTI_HOP, "Complete Cybersecurity Questions")
    columns = [Column(Header(heading, metric, 0, None, None), dataset, agg,
                      aggregation_evidence=interval_evidence if agg else None)
               for dataset in datasets for heading, metric, agg in (
                   ("Elo", "Elo", None), ("MLE Elo", "MLE Elo", None), ("Bootstrap", "bootstrap Elo", "median"),
                   ("Bootstrap", "bootstrap Elo", "q0.975"), ("Bootstrap", "bootstrap Elo", "q0.025"),
               )]
    rows, bounds = [], []
    for y in (355, 412, 469, 526, 583):
        model = " ".join(w.text for w in sorted(labels, key=lambda w: w.x) if abs(w.y - y) < 25)
        model = model.strip("()| ")
        if not MODEL.fullmatch(model):
            model = " ".join(w.text for w in sorted(heading_words, key=lambda w: w.x)
                             if w.x1 < 394 and abs(w.y - y) < 24).strip("()| ")
        if not MODEL.fullmatch(model):
            label_crop = image.crop(25, y - 35, 348, y + 17)
            model = " ".join(w.text for w in recognize(label_crop, source.engine, psm=7))
        if not MODEL.fullmatch(model):
            continue
        values, boxes = [], []
        for x1, x2, left, right, shift in ((449, 653, 788, 1255, 0), (1350, 1552, 1686, 2150, 0),
                                           (2246, 2448, 2580, 3050, 23)):
            for x, pattern in ((x1, ELO), (x2, DECIMAL_ELO)):
                matches = [word for word in body if abs(word.x - x) < 53 and abs(word.y - y) < 24
                           and pattern.fullmatch(word.text)]
                if len(matches) == 1:
                    word = matches[0]
                    result = word.text, image.source_bbox(word.x0, word.y0, word.x1, word.y1)
                else:
                    result = reread_cell(image, source.engine, x, y, pattern)
                    if result is None and pattern is DECIMAL_ELO:
                        center = y + (9 if shift else 0)
                        result = confirm_cell(image, source.engine, (x - 77, center - 24, x + 77, center + 24), pattern)
                values.append(result[0] if result else None)
                boxes.append(result[1] if result else None)
            result = None
            if interval_evidence:
                center = y + shift
                result = bootstrap_cell(image, source.engine, body, left, right, center)
                if result is None and shift:
                    cell = confirm_cell(image, source.engine, (left + 9, center - 24, right - 4, center + 24), BOOTSTRAP)
                    if cell and (match := BOOTSTRAP.fullmatch(cell[0])) and ordered_interval(match.groups()):
                        result = match.groups(), cell[1]
            values.extend(result[0] if result else (None,) * 3)
            boxes.extend((result[1] if result else None,) * 3)
        rows.append((model, values))
        bounds.append(tuple(boxes))
    if len(rows) < 2:
        return None
    return Table(page, label, caption, 0, columns, rows, family="scanned_scores", value_bboxes=tuple(bounds))


def component_label(words: tuple[Word, ...], y: int) -> str:
    return " ".join(w.text for w in sorted(words, key=lambda w: w.x)
                    if 170 < w.x and w.x1 < 790 and y - 2 < w.y < y + 31)


def component_inventory_tables(page: Page, label: str, caption: str,
                               views: tuple[tuple[ScanImage, tuple[Word, ...]], ...]) -> list[Table]:
    _, positioned = views[0]
    header = [w.text for w in sorted(positioned, key=lambda w: w.x)
              if 250 < w.y < 305 and (2200 < w.x < 2410 or 2820 < w.x < 3090)]
    if header != ["Size", "No.", "Doc."]:
        return []

    def confirmed_cell(y, left, right, offset, pattern):
        readings = {}
        for image, words in views:
            scale = image.dpi / 480
            cell = sorted((w for w in words if left < w.x / scale < right and abs(w.y / scale - y - offset) < 21),
                          key=lambda word: word.x)
            raw = "".join(w.text for w in cell)
            match = pattern.fullmatch(raw)
            if raw != "-" and not match:
                continue
            if match and pattern is DOCUMENT_COUNT:
                # OCR prints the mathematical approximation glyph as an ASCII tilde.
                raw = f"∼{match[1]}"
            box = image.source_bbox(min(w.x0 for w in cell), min(w.y0 for w in cell),
                                    max(w.x1 for w in cell), max(w.y1 for w in cell))
            readings.setdefault(raw, []).append(box)
        agreed = [(raw, boxes[0]) for raw, boxes in readings.items() if len(boxes) >= 2]
        return agreed[0] if len(agreed) == 1 else None

    tables = []
    for index, y in enumerate(COMPONENT_ROWS, start=1):
        component = component_label(positioned, y)
        if not COMPONENT.fullmatch(component):
            continue
        columns, values, boxes = [], [], []
        for title, name, left, right, offset, pattern in (
            ("Size", "dense retriever size", 2200, 2410, 3, COMPONENT_SIZE),
            ("No. Doc.", "knowledge-base document count", 2820, 3090, -9, DOCUMENT_COUNT),
        ):
            if cell := confirmed_cell(y, left, right, offset, pattern):
                raw, box = cell
                size = COMPONENT_SIZE.fullmatch(raw)
                columns.append(Column(Header(title, name, (left + right) / 2, None, size[2] if size else None), None))
                values.append(raw)
                boxes.append(box)
        if values:
            tables.append(Table(page, label, caption, index, columns, [(component, values)],
                                family="scanned_scores", value_bboxes=(tuple(boxes),)))
    return tables


def component_view(page: Page, dpi: int = 480, *, inventory: bool = False) -> tuple[ScanImage, tuple[Word, ...]]:
    image = deskew(scan_region(page.scan, COMPONENT_BOUNDS, dpi=dpi)) if dpi != 480 else deskew(
        scan_region(page.scan, COMPONENT_BOUNDS))
    if not inventory:
        return image, recognize(image, page.scan.engine)
    scale = dpi / 480
    x0, y0, x1, y1 = (round(value * scale) for value in (2200, 478, 3100, 1324))
    words = recognize(image.crop(x0, y0, x1, y1), page.scan.engine)
    return image, tuple(Word(w.text, w.x0 + x0, w.y0 + y0, w.x1 + x0, w.y1 + y0) for w in words)


def latency_table(page: Page, label: str, caption: str, *,
                  positioned: tuple[ScanImage, tuple[Word, ...]] | None = None) -> Table | None:
    source = page.scan
    if not fragment(page, r"Struc\s*tured\s+and\s+Unstructured\s+RAG\s+components"):
        return None
    if not fragment(page, r"Mean Time\s*\(s\)") or not fragment(page, r"Time Std\."):
        return None
    image, words = positioned or component_view(page)
    headings = [w.text for w in sorted(words, key=lambda w: w.x) if 340 < w.y < 389]
    if headings != ["GPU", "CPU", "GPU", "CPU"]:
        return None
    columns = [Column(Header(f"{printed} / {device}", "retrieval latency", x, None, "s"), None,
                      aggregation, direction="lower_is_better")
               for printed, aggregation, start in (("Mean Time (s)", "mean", 855), ("Time Std.", "std", 1235))
               for device, x in (("GPU", start), ("CPU", start + 190))]
    rows, bounds = [], []
    for y in COMPONENT_ROWS:
        component = component_label(words, y)
        if not COMPONENT.fullmatch(component):
            continue
        values, boxes = [], []
        for column in columns:
            x = int(column.header.center)
            matches = [w for w in words if abs(w.x - x) < 42 and abs(w.y - y) < 23 and TIME.fullmatch(w.text)]
            if len(matches) == 1:
                word = matches[0]
                result = word.text, image.source_bbox(word.x0, word.y0, word.x1, word.y1)
            else:
                result = reread_cell(image, source.engine, x, y, TIME)
            values.append(result[0] if result else None)
            boxes.append(result[1] if result else None)
        rows.append((component, values))
        bounds.append(tuple(boxes))
    if len(rows) < 4:
        return None
    return Table(page, label, caption, 0, columns, rows, family="scanned_scores", value_bboxes=tuple(bounds))


def read_scanned_score_tables(pages: list[Page], tables: list[Table]) -> list[Table]:
    recovered = []
    replaced = set()
    definitions = next((page for page in pages if page.scan and CAPTION_II.search(page.prose)), None)
    if definitions is None:
        return tables
    rubric_evidence = next((match for page in pages if page.scan
                            and (match := fragment(page, r"Five[- ]Level\s+Judgment\s+Criteria"))
                            and fragment(page, r"GPT-4\s+0[1l/]25-Preview")), None)
    interval_evidence = next((match for page in pages if page.scan and (match := fragment(
        page, r"median,\s*2\.5\s*%\s*,?\s*and\s*97\.5\s*%\s+percentiles",
    ))), None)
    judge_evidence = next((match for page in pages if page.scan and fragment(
        page, r"reference\s+response,\s+we\s+ask\s+GPT-4\s+0125-Preview\s+to\s+choose\s+the\s+best\s+response"
              r"\s+between\s+Model\s+A\s+and\s+Model\s+B",
    ) and (match := fragment(page, r"we\s+ask\s+GPT-4\s+0125-Preview"))), None)
    for page in pages:
        if page.scan and judge_evidence and (wins := CAPTION_IV.search(page.prose)):
            try:
                if table := win_table(page, wins[1], wins[0], judge_evidence):
                    recovered.append(table)
            except (ExtractionError, ValueError) as error:
                print(f"page {page.number}, {wins[1]} crop OCR: {error}", file=sys.stderr)
        if page.scan and (timing := CAPTION_VII.search(page.prose) or CAPTION_VII.search(page.text)):
            try:
                caption = f"{timing[0]} Performance summary of RAG components"
                primary = component_view(page)
                if table := latency_table(page, timing[1], caption, positioned=primary):
                    recovered.append(table)
                    views = (primary, *(component_view(page, dpi, inventory=True) for dpi in (360, 405)))
                    recovered.extend(component_inventory_tables(page, timing[1], caption, views))
            except (ExtractionError, ValueError) as error:
                print(f"page {page.number}, {timing[1]} crop OCR: {error}", file=sys.stderr)
        if page.scan and (rank := CAPTION_V.search(page.prose)):
            try:
                if table := rank_table(page, rank[1], rank[0], interval_evidence):
                    recovered.append(table)
            except (ExtractionError, ValueError) as error:
                print(f"page {page.number}, {rank[1]} crop OCR: {error}", file=sys.stderr)
        for pattern, grid in ((CAPTION_II, PAIRED), (CAPTION_III, CVE_GRID), (CAPTION_VI, JUDGE_GRID)):
            match = pattern.search(page.prose)
            if not match or page.scan is None:
                continue
            try:
                data = scan_grid(page, grid)
            except (ExtractionError, ValueError) as error:
                print(f"page {page.number}, {match[1]} crop OCR: {error}", file=sys.stderr)
                continue
            if not data:
                continue
            rows, boxes = data
            if grid is JUDGE_GRID:
                if rubric_evidence is None:
                    continue
                columns = [Column(Header(label, "reference-based judge score", 0, None, None), dataset,
                                  metric_variant="GPT-4 five-level rubric", metric_variant_evidence=rubric_evidence)
                           for label, dataset in (("General Qs", GENERAL), ("Multi-Hop Qs", MULTI_HOP), ("CVE Qs", CVE))]
            else:
                columns = metric_columns(page, definitions, grid)
            prior = [table for table in tables if table.page.number == page.number
                     and table.label.casefold() == match[1].casefold()]
            if grid is PAIRED:
                cells = dict(rows)
                shape = [(column.dataset, column.header.label.casefold(), column.aggregation) for column in columns]
                if any(
                    [(column.dataset, column.header.label.casefold(), column.aggregation)
                     for column in table.columns] != shape
                    or any(cells.get(row) != values for row, values in table.rows)
                    for table in prior
                ):
                    continue
            elif prior:
                continue
            recovered.append(Table(page, match[1], match[0].strip(), 0, columns, rows,
                                   family="scanned_scores", value_bboxes=boxes))
            replaced.add((page.number, match[1].casefold()))
    return [table for table in tables if (table.page.number, table.label.casefold()) not in replaced] + recovered
