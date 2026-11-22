"""Recover values from locally recognized training-table scans."""

from collections.abc import Callable
import re
from statistics import median

from pdf_pages import ExtractionError, Page, Word
from rag_tables import Column, Header, Table
from scan_ocr import ScanImage, contrast, deskew, recognize, scan_region


OVERHEAD = re.compile(r"^\s*(Table\s+\d+)[:.]\s*C\s*omputational overhead\b", re.I | re.M)
OVERHEAD_CAPTION = re.compile(
    r"(Table\s+\d+)[:.]\s+Computational overhead on (\w+) with (Qwen\d+(?:\.\d+)?-)\s*"
    r"(\d+(?:\.\d+)?B-Instruct) and (Top-(\d+) retrieval)\b", re.I,
)
OVERHEAD_COLUMNS = (
    ("Step time (s)", "Average training step time", "s"),
    ("Total time (h)", "Total training time", "h"),
    ("Sampled peak GPU (GiB)", "Sampled peak GPU memory", "GiB"),
    ("Non-ckpt. storage (MiB)", "Non-checkpoint storage", "MiB"),
)
DECIMAL = re.compile(r"\d+\.\d+")
UNIT_TOKEN = re.compile(r"\((s|h|gib|mib)\)$", re.I)
TIME_ALIGNED = re.compile(r"^\s*(Table\s+\d+)[:.]\s*Time-aligned\b", re.I | re.M)
ALIGNED_CAPTION = re.compile(
    r"(Table\s+\d+):\s*Time-aligned task, training-diagnostic, and retrieval-behavior statistics on the fixed "
    r"(Qwen\d+(?:\.\d+)?-\d+(?:\.\d+)?B)\s+(\w+)\s+(Top-(\d+)) evaluation set\. "
    r"The miss column reports an answer-string retrieval-miss proxy\.", re.I,
)
ALIGNED_ROW = re.compile(r"^(PPO|OAS)[,.]\s+step\s+(\d+)\)?(?=\s|$)", re.I)
ALIGNED_SCORE = re.compile(r"(?<![\d.,+-])[+−-]?\d+[.,]\d{3}(?![\d.,])")
ALIGNED_HEADERS = (
    re.compile(r"^Run\s+EM\s*/\s*nVDD\b.*\bATE\s*/\s*OAGC\b", re.I),
    re.compile(r"^Run\s+Searches\s*/\s*saturation\b.*\bMiss\b.*\bNo search\b", re.I),
)
ALIGNED_METRICS = (
    (("EM", "EM"), ("nVDD", "nVDD"), ("ATE", "ATE"), ("OAGC", "OAGC")),
    (("Searches", "Executed searches"), ("saturation", "Search budget saturation"),
     ("Miss", "Answer-string retrieval-miss proxy"), ("No search", "No-search rate")),
)
Bounds = tuple[float, float, float, float]
Cell = tuple[str, Bounds]
Checkpoint = tuple[int, str, str]
CheckpointCells = dict[Checkpoint, dict[int, Cell]]
CHECKPOINT_ROWS = (("PPO", "150"), ("PPO", "250"), ("OAS", "150"), ("OAS", "250"))
CHECKPOINT_CENTERS = ((566, 673, 825, 935), (515, 620, 821, 996))
# Positions in the 405 dpi right-column region. The row names and both
# header bands are checked against the OCR before reading individual cells.
CHECKPOINT_Y = (
    ((154, 156, 158, 158), (194, 196, 198, 198), (236, 238, 240, 242), (276, 278, 280, 282)),
    ((442, 443, 445, 446), (483, 484, 486, 486), (522, 524, 526, 528), (566, 567, 569, 570)),
)
CHECKPOINT_DECIMAL = re.compile(r"\d{1,2}[.,]\d{3}")


def ordered_lines(words: tuple[Word, ...]) -> list[list[Word]]:
    if not words:
        return []
    tolerance = max(2, median(word.y1 - word.y0 for word in words) * 0.65)
    lines: list[list[Word]] = []
    for word in sorted(words, key=lambda word: word.y):
        if not lines or word.y - lines[-1][-1].y > tolerance:
            lines.append([])
        lines[-1].append(word)
    return [sorted(line, key=lambda word: word.x) for line in lines]


def text(line: list[Word]) -> str:
    return " ".join(word.text for word in line)


def overhead_table(page: Page, words: tuple[Word, ...], read_cell: Callable[[Word], tuple[str, Bounds] | None]) -> list[Table]:
    lines = ordered_lines(words)
    caption = OVERHEAD_CAPTION.search(" ".join(text(line) for line in lines))
    if not caption:
        return []
    label, dataset, model_family, model_size, budget_fragment, budget = caption.groups()
    model = model_family + model_size
    header_index = next((i for i, line in enumerate(lines) if line[0].text == "Method"), None)
    if header_index is None:
        return []
    heading = lines[header_index]
    headings = [(unit[1].casefold(), word.x) for word in heading if (unit := UNIT_TOKEN.search(word.text))]
    if [unit for unit, _ in headings] != ["s", "h", "gib", "mib"]:
        return []
    if not re.search(r"peak GPU\b.*[sM]t?orage", text(heading), re.I):
        return []

    centers = [center for _, center in headings]
    method_limit = centers[0] * 0.65
    column_tolerance = min(right - left for left, right in zip(centers, centers[1:])) / 2
    tables = []
    for offset, line in enumerate(lines[header_index + 1:header_index + 5], start=header_index + 1):
        left = [word for word in line if word.x < method_limit]
        method = re.search(r"(?:Search-R[1I!l]\s+PPO|\bOAS\b)(?=\s|$)", text(left))
        if method is None:
            continue
        prefix = [word for word in left if word.text in method[0].split()]
        row_y = median(word.y for word in prefix)
        height = prefix[0].y1 - prefix[0].y0
        cells = [word for word in line if word.x >= method_limit and abs(word.y - row_y) <= height * 0.5
                 and word.y1 - word.y0 <= height * 1.6]
        if len(cells) != len(centers):
            continue
        if any(abs(center - cell.x) > column_tolerance for center, cell in zip(centers, cells, strict=True)):
            continue
        values, columns, boxes = [], [], []
        for cell, (printed, metric, unit) in zip(cells, OVERHEAD_COLUMNS, strict=True):
            recognized = read_cell(cell)
            if recognized is None:
                continue
            value, box = recognized
            values.append(value)
            boxes.append(box)
            columns.append(Column(Header(printed, metric, cell.x, None, unit), dataset, generator=model,
                                  source_label=printed, direction="lower_is_better"))
        if values:
            name = "OAS" if method[0] == "OAS" else "Search-R1 PPO"
            tables.append(Table(page, label, caption[0], offset, columns, [(name, values)], top_k=int(budget),
                                top_k_evidence=(page, budget_fragment), value_bboxes=(tuple(boxes),)))
    return tables


def overhead_cell(image: ScanImage, alternate: ScanImage, alternate_words: tuple[Word, ...],
                  check_high_resolution: Callable[[Bounds, str], bool], engine: str, word: Word) -> tuple[str, Bounds] | None:
    height = max(11, round(image.dpi / 31))
    x0, x1 = max(0, round(word.x0) - height), min(image.width, round(word.x1) + height)
    y0, y1 = max(0, round(word.y) - height), min(image.height, round(word.y) + height)
    box = image.source_bbox(x0, y0, x1, y1)
    matches = [other.text for other in alternate_words
               if inside_box(alternate, other, box) and DECIMAL.fullmatch(other.text.replace(",", "."))]
    if len(matches) != 1:
        return None
    value = matches[0]
    if "," in value:
        dotted = value.replace(",", ".")
        # The dot can be transcribed only when an independent reading shows it.
        # A fully formed conflicting number from the first pass remains disputed.
        if not DECIMAL.fullmatch(word.text) and check_high_resolution(box, dotted):
            return dotted, box
        return None
    if word.text in (value, value.replace(".", "")):
        return value, box
    clip = image.crop(x0, y0, x1, y1)
    clip_text = "".join(w.text for w in recognize(contrast(clip, 205), engine, 7, digits_only=True))
    if clip_text == value:
        return value, box
    return None


def inside_box(image: ScanImage, word: Word, box: Bounds) -> bool:
    x0, y0, x1, y1 = image.source_bbox(word.x0, word.y0, word.x1, word.y1)
    return box[0] < (x0 + x1) / 2 < box[2] and box[1] < (y0 + y1) / 2 < box[3]


def time_aligned_rows(lines: list[list[Word]]) -> dict[tuple[int, str, str], list[tuple[str, Word]]]:
    rows = {}
    block = None
    for line in lines:
        printed = text(line)
        if TIME_ALIGNED.match(printed):
            break
        heading = next((i for i, pattern in enumerate(ALIGNED_HEADERS) if pattern.search(printed)), None)
        if heading is not None:
            block = heading
            continue
        row = ALIGNED_ROW.match(printed)
        if block is None or not row:
            continue
        cells = [(match[0], word) for word in line[3:] for match in ALIGNED_SCORE.finditer(word.text)]
        if len(cells) == len(ALIGNED_METRICS[block]):
            key = block, row[1].upper(), row[2]
            if key in rows:
                return {}
            rows[key] = cells
    return rows


def agreed_checkpoint_cells(rows, alternate, extra, bbox) -> CheckpointCells:
    confirmed = {}
    for key, cells in rows.items():
        if key not in alternate:
            continue
        matches = {}
        for col, ((first, word), (second, _)) in enumerate(zip(cells, alternate[key], strict=True)):
            dotted = first.replace(",", ".")
            if dotted != second.replace(",", ".") or "." not in first + second:
                continue
            if key in extra and extra[key][col][0].replace(",", ".") != dotted:
                continue
            matches[col] = (first if "." in first else second, bbox(word.x0, word.y0, word.x1, word.y1))
        confirmed[key] = matches
    return confirmed


def bounded_checkpoint_cells(first: ScanImage, second: ScanImage, first_words: tuple[Word, ...],
                             second_words: tuple[Word, ...], confirmed: CheckpointCells,
                             engine: str, higher_resolution: Callable[[], ScanImage] | None = None) -> CheckpointCells:
    def local_text(words, image, left, right, y, radius):
        scale = image.dpi / 405
        return text(sorted((w for w in words if left * scale < w.x < right * scale
                            and abs(w.y / scale - y) < radius), key=lambda w: w.x))

    regions = ((first, first_words), (second, second_words))
    headings = tuple(any(pattern.search(local_text(words, image, 90, 1100, y, 18))
                         for image, words in regions)
                     for pattern, y in zip(ALIGNED_HEADERS, (84, 372), strict=True))
    if not all(headings):
        return {}

    def read_cell(image, x, y, width, height, white, psm):
        scale = image.dpi / 405
        box = tuple(round(value * scale) for value in (x - width, y - height, x + width, y + height))
        crop = contrast(image.crop(*box), white)
        words = recognize(crop, engine, psm, digits_only=True)
        if len(words) != 1 or not CHECKPOINT_DECIMAL.fullmatch(words[0].text):
            return None
        word = words[0]
        return word.text, crop.source_bbox(word.x0, word.y0, word.x1, word.y1)

    found = {}
    for block, (left, right) in enumerate(((230, 480), (128, 392))):
        for i, (method, step) in enumerate(CHECKPOINT_ROWS):
            key = block, method, step
            ys = CHECKPOINT_Y[block][i]
            labels = [ALIGNED_ROW.match(local_text(words, image, left, right, ys[0], 18))
                      for image, words in regions]
            if not all(label and (label[1].upper(), label[2]) == (method, step) for label in labels):
                continue
            for col, (x, y) in enumerate(zip(CHECKPOINT_CENTERS[block], ys, strict=True)):
                if col in confirmed.get(key, {}):
                    continue
                radius = (45, 48, 53, 49)[col]
                readings = [value for image, white, psm in ((first, 184, 7), (second, 240, 7), (second, 208, 8))
                            if (value := read_cell(image, x, y, radius, 18, white, psm)) is not None]
                dotted = None
                if len(readings) >= 2 and len({raw.replace(",", ".") for raw, _ in readings}) == 1:
                    dotted = next(((raw, box) for raw, box in readings if "." in raw), None)
                elif higher_resolution is not None:
                    candidate = read_cell(first, x, y, radius + 1, 17, 240, 7)
                    if candidate is not None and "." in candidate[0]:
                        rescanned = read_cell(higher_resolution(), x, y, radius, 18, 184, 7)
                        if rescanned is not None and rescanned[0] == candidate[0]:
                            dotted = rescanned
                if dotted is not None:
                    found.setdefault(key, {})[col] = dotted
    return found


def time_aligned_tables(page: Page, pages: list[Page], words: tuple[Word, ...], alternate_words: tuple[Word, ...],
                        bbox: Callable[[float, float, float, float], Bounds],
                        extra_words: tuple[Word, ...] = (), *,
                        read_missing: Callable[[CheckpointCells], CheckpointCells] | None = None) -> list[Table]:
    lines = ordered_lines(words)
    caption = ALIGNED_CAPTION.search(" ".join(text(line) for line in lines))
    if not caption:
        return []
    label, model, dataset, budget_fragment, budget = caption.groups()
    # The discussion on the earlier page links the controlled methods and
    # evaluation set to the fuller results in this table.
    trajectory = re.compile(r"one controlled trajectory per method", re.I)
    evaluation = re.compile(rf"{re.escape(model)}\s+{re.escape(dataset)}\s+{re.escape(budget_fragment)}\s+evaluation set", re.I)
    number = label.split()[-1]
    reference = re.compile(rf"\bTable\s+{number}\b|\bTab\b.{{0,50}}\ble\s+{number}\b|\ble\s+{number}\b.{{0,50}}\bTab\b", re.I)
    protocol = next((p for p in pages if p.number < page.number and reference.search(p.prose)
                     and trajectory.search(p.prose) and evaluation.search(p.prose)), None)
    if protocol is None:
        return []
    full_model = re.compile(re.escape(model).replace(r"\-", r"-\s*") + r"-Instruct\b", re.I)
    if any(full_model.search(p.prose) for p in pages):
        model += "-Instruct"

    confirmed = agreed_checkpoint_cells(time_aligned_rows(lines), time_aligned_rows(ordered_lines(alternate_words)),
                                        time_aligned_rows(ordered_lines(extra_words)), bbox)
    supplemental = read_missing(confirmed) if read_missing is not None else {}
    tables = []
    for index, key in enumerate(dict.fromkeys((*confirmed, *supplemental))):
        block, method, step = key
        cells = {**supplemental.get(key, {}), **confirmed.get(key, {})}
        values, columns, boxes = [], [], []
        for col, (raw, box) in sorted(cells.items()):
            printed, metric = ALIGNED_METRICS[block][col]
            values.append(raw)
            boxes.append(box)
            columns.append(Column(Header(printed, metric, 0, None, None), dataset,
                                  aggregation="single controlled trajectory", generator=model,
                                  aggregation_evidence=(protocol, trajectory.search(protocol.prose)[0]),
                                  metric_variant=f"step {step}", source_label=printed))
        if values:
            name = "Search-R1 PPO" if method == "PPO" else method
            tables.append(Table(page, label, caption[0], index, columns, [(f"{method}, step {step}", values)],
                                config_name=name, top_k=int(budget), top_k_evidence=(page, budget_fragment),
                                value_bboxes=(tuple(boxes),)))
    return tables


def read_time_aligned(page: Page, pages: list[Page]) -> list[Table]:
    bounds = (575 / 1117, 175 / 1579, 973 / 1117, 478 / 1579)
    engine = page.scan.engine
    first = deskew(scan_region(page.scan, bounds, dpi=495))
    second = deskew(scan_region(page.scan, bounds, dpi=405))
    primary = recognize(contrast(first, 203), engine)
    alternate = recognize(second, engine)
    extra = recognize(contrast(second, 203), engine)
    high_resolution = None

    def read_high_resolution():
        nonlocal high_resolution
        if high_resolution is None:
            high_resolution = deskew(scan_region(page.scan, bounds, dpi=660))
        return high_resolution

    def read_missing(confirmed):
        return bounded_checkpoint_cells(second, first, alternate, primary, confirmed, engine, read_high_resolution)

    return time_aligned_tables(page, pages, primary, alternate, first.source_bbox, extra,
                               read_missing=read_missing)


def read_training_scan_tables(pages: list[Page], tables: list[Table]) -> list[Table]:
    occupied = {(table.page.number, table.label.casefold()) for table in tables}
    found = []
    for page in pages:
        if not page.scan:
            continue
        if (aligned := TIME_ALIGNED.search(page.text)) and (page.number, aligned[1].casefold()) not in occupied:
            try:
                found.extend(read_time_aligned(page, pages))
            except (ExtractionError, ValueError):
                pass
        if not (caption := OVERHEAD.search(page.text)):
            continue
        if (page.number, caption[1].casefold()) in occupied:
            continue
        try:
            # This layout puts the narrow overhead table in the right column; the
            # secondary recognition verifies the caption, method and unit columns.
            bounds = (566 / 1117, 476 / 1579, 960 / 1117, 655 / 1579)
            crop = deskew(scan_region(page.scan, bounds, dpi=405))
            words = recognize(crop, page.scan.engine)
            alternate = deskew(scan_region(page.scan, bounds, dpi=570))
            alternate_words = recognize(contrast(alternate, 220), page.scan.engine)
            precise = None

            def high_resolution_matches(box, raw):
                nonlocal precise
                if precise is None:
                    image = deskew(scan_region(page.scan, bounds, dpi=660))
                    precise = image, recognize(contrast(image, 220), page.scan.engine)
                image, precise_words = precise
                return any(word.text == raw and inside_box(image, word, box) for word in precise_words)

            found.extend(overhead_table(page, words, lambda word: overhead_cell(
                crop, alternate, alternate_words, high_resolution_matches, page.scan.engine, word)))
        except (ExtractionError, ValueError):
            continue
    return found
