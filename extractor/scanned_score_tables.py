"""Read scanned model-score grids with distinct cells for mean and deviation."""

from dataclasses import dataclass
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
CAPTION_V = re.compile(r"\b(TABLE V): Elo Ratings, MLE Ratings, and Bootstrap Ratings", re.I)
CAPTION_VI = re.compile(r"\b(TABLE VI): Scores Across All Cybersecurity Questions types", re.I)
CAPTION_VII = re.compile(r"\b(TABLE VII):", re.I)
ELO = re.compile(r"\d{3,4}")
DECIMAL_ELO = re.compile(r"\d{3,4}\.\d{2}")
BOOTSTRAP = re.compile(r"(\d{3,4}\.\d{2})\s*\((\d{3,4})-(\d{3,4})\)")
TIME = re.compile(r"\d{1,3}\.\d{1,3}")
COMPONENT = re.compile(r"[\w-]+ Retriever|Question Ret\. Sys\.|[\w-]+ BUFFERS|CONTEXT TRANSF\.", re.I)


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
                                  (r"identified\s+the\s+vulnerability",))),
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
        columns.append(Column(
            Header(label, name, 0, None, "%" if label == "Accuracy" else None),
            dataset, agg, direction="higher_is_better" if agg != "std" else None,
            name_evidence=fragment(source, rf"\b{re.escape(name)}\b"),
            metric_variant=variant if confirmed else None,
            metric_variant_evidence=evidence[0] if confirmed else None,
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
    center, upper, lower = values
    if not int(lower) <= float(center) <= int(upper):
        return None
    return values, image.source_bbox(x0, y0, x1, y1)


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
            continue
        values, boxes = [], []
        for x1, x2, left, right in ((449, 653, 788, 1255), (1350, 1552, 1686, 2150), (2246, 2448, 2580, 3050)):
            for x, pattern in ((x1, ELO), (x2, DECIMAL_ELO)):
                matches = [word for word in body if abs(word.x - x) < 53 and abs(word.y - y) < 24
                           and pattern.fullmatch(word.text)]
                if len(matches) == 1:
                    word = matches[0]
                    result = word.text, image.source_bbox(word.x0, word.y0, word.x1, word.y1)
                else:
                    result = reread_cell(image, source.engine, x, y, pattern)
                values.append(result[0] if result else None)
                boxes.append(result[1] if result else None)
            result = bootstrap_cell(image, source.engine, body, left, right, y) if interval_evidence else None
            values.extend(result[0] if result else (None,) * 3)
            boxes.extend((result[1] if result else None,) * 3)
        rows.append((model, values))
        bounds.append(tuple(boxes))
    if len(rows) < 2:
        return None
    return Table(page, label, caption, 0, columns, rows, family="scanned_scores", value_bboxes=tuple(bounds))


def latency_table(page: Page, label: str, caption: str) -> Table | None:
    source = page.scan
    if not fragment(page, r"Struc\s*tured\s+and\s+Unstructured\s+RAG\s+components"):
        return None
    if not fragment(page, r"Mean Time\s*\(s\)") or not fragment(page, r"Time Std\."):
        return None
    image = deskew(scan_region(source, (.085, .075, .895, .339)))
    words = recognize(image, source.engine)
    headings = [w.text for w in sorted(words, key=lambda w: w.x) if 340 < w.y < 389]
    if headings != ["GPU", "CPU", "GPU", "CPU"]:
        return None
    columns = [Column(Header(f"{printed} / {device}", "retrieval latency", x, None, "s"), None,
                      aggregation, direction="lower_is_better")
               for printed, aggregation, start in (("Mean Time (s)", "mean", 855), ("Time Std.", "std", 1235))
               for device, x in (("GPU", start), ("CPU", start + 190))]
    rows, bounds = [], []
    for y in (546, 602, 658, 716, 770, 827, 884, 1063, 1119, 1174, 1232, 1290):
        component = " ".join(w.text for w in sorted(words, key=lambda w: w.x)
                             if 170 < w.x and w.x1 < 790 and y - 2 < w.y < y + 31)
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
    for page in pages:
        if page.scan and (timing := CAPTION_VII.search(page.prose) or CAPTION_VII.search(page.text)):
            try:
                if table := latency_table(page, timing[1], f"{timing[0]} Performance summary of RAG components"):
                    recovered.append(table)
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
