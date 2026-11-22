"""Read mean and sample deviation from ruled, grouped OAS comparison scans."""

from collections import defaultdict
from dataclasses import dataclass
import re

from pdf_pages import ExtractionError, Page, Word
from rag_tables import Column, Header, Table
from scan_ocr import ScanImage, contrast, deskew, recognize, scan_region
from training_scan import Bounds, ordered_lines, text


CAPTION_START = re.compile(r"^\s*(Table\s+\d+)[:.]\s+(Main|Additional) OAS comparison\b", re.I | re.M)
CAPTION = re.compile(
    r"(Table\s+\d+):\s*(?:Main|Additional) OAS comparison with\s+"
    r"(Qwen\d+(?:\.\d+)?-\d+(?:\.\d+)?B-Instruct)\.\s*Exact-match percentages are reported as mean\s*"
    r"\(sample SD\) over two runs\s*\((?:[nm]m?|7)\s*=\s*(2)\)", re.I,
)
CAPTION_MODEL = re.compile(r"with (Qwen\d+(?:\.\d+)?-\d+(?:\.\d+)?B-)Inst(?:ruct|r)?\b", re.I)
CAPTION_SCALE = re.compile(r"Exact-match percentages are reported as mean\b", re.I)
CAPTION_RUNS = re.compile(r"\bSD\) over two runs \(n\s*=\s*(2)\)", re.I)
BUDGET = re.compile(r"Top-k\s*=\s*(\d+)\b", re.I)
DATASET = re.compile(r"\b(NQ|TriviaQA|Hotpo[tr]QA)\b", re.I)
PAIR = re.compile(r"(\d{2}\.\d)\(([0-9]?\.\d{2})\)")
ROW_NAMES = (
    (r"Search-R[1I]\s+PPO", "Search-R1 PPO"),
    (r"Search-R[1I]\s*GRPO", "Search-R1 GRPO"),
    (r"DAPO", "DAPO"),
    (r"KL-Co[vwy]", "KL-Cov"),
    (r"DAR", "DAR"),
    (r"[I1]-PPO", "I-PPO"),
    (r"Ours\s+\(OVR only\)", "OVR only"),
    (r"Ours\s+\(OAS\)", "OAS"),
)


@dataclass(frozen=True)
class Layout:
    # Locations describe a 135 dpi render of the two printed layouts. The
    # actual rule positions are discovered independently in each column.
    region: tuple[int, int, int, int]
    values_left: int
    values_right: int
    header_band: tuple[int, int]
    footer_band: tuple[int, int]
    header_endpoints: tuple[float, float]
    footer_endpoints: tuple[float, float]


LAYOUTS = {
    "main": Layout((158, 609, 975, 897), 280, 956, (654, 690), (782, 815), (664, 680), (791, 805.4)),
    "additional": Layout((165, 161, 981, 421), 286, 972, (207, 242), (331, 359), (209, 225), (334, 351.6)),
}


CellReadings = dict[tuple[int, int], tuple[str, str, Bounds]]


def paired_tables(page: Page, caption_text: str, heading: str, row_labels: list[str],
                  cells: CellReadings) -> list[Table]:
    caption = CAPTION.search(caption_text)
    if not caption:
        return []
    budget_labels = list(BUDGET.finditer(heading))
    datasets = ["HotpotQA" if match[1].casefold().startswith("hotpo") else match[1]
                for match in DATASET.finditer(heading)]
    if [match[1] for match in budget_labels] != ["2", "3", "5"] or [name.casefold() for name in datasets] != (
        ["nq", "triviaqa", "hotpotqa"] * 3
    ):
        return []
    if len(row_labels) != len(ROW_NAMES):
        return []
    tables = []
    for row, (pattern, configuration) in enumerate(ROW_NAMES):
        label = row_labels[row]
        if not re.fullmatch(pattern, label, re.I):
            continue
        for i, budget in enumerate(budget_labels):
            columns, values, boxes = [], [], []
            for col in range(i * 3, i * 3 + 3):
                first, second, box = cells.get((row, col), ("", "", None))
                if first != second or not (pair := PAIR.fullmatch(first)):
                    continue
                for raw, metric, aggregation in zip(
                    pair.groups(), ("EM", "EM sample SD"),
                    (f"mean over {caption[3]} runs", f"sample standard deviation over {caption[3]} runs"), strict=True,
                ):
                    dataset = datasets[col]
                    columns.append(Column(Header(metric, metric, col, None, None), dataset, aggregation=aggregation,
                                          generator=caption[2], reported_scale="percent",
                                          source_label=f"{budget[0]} / {dataset} / {metric}", sample_size=int(caption[3]),
                                          sample_size_label="n"))
                    values.append(raw)
                    boxes.append(box)
            if values:
                tables.append(Table(page, caption[1], caption[0], row * len(budget_labels) + i, columns,
                                    [(label, values)], config_name=configuration, top_k=int(budget[1]),
                                    top_k_evidence=(page, budget[0]), value_bboxes=(tuple(boxes),)))
    return tables


def projection(image: ScanImage, x0: int, y0: int, x1: int, y1: int, cutoff: int) -> list[int]:
    return [sum(cutoff - value for value in image.pixels[y * image.width + x0:y * image.width + x1]
                if value < cutoff) for y in range(y0, y1)]


def crop_cells(image: ScanImage, layout: Layout) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    scale = image.dpi / 135

    def raster_x(x):
        return round(x * scale) - image.left

    def raster_y(y):
        return round(y * scale) - image.top

    def rule(xa, xb, column, band, endpoints):
        y0, y1 = map(raster_y, band)
        expected = raster_y(endpoints[0] + (endpoints[1] - endpoints[0]) * (column / 8))
        light = projection(image, xa, y0, xb, y1, 110)
        return y0 + max(range(3, len(light) - 3),
                        key=lambda row: sum(light[row - 3:row + 4]) - 100 * abs(row + y0 - expected))

    cells = {}
    for col in range(9):
        width = (layout.values_right - layout.values_left) / 9
        xa = raster_x(layout.values_left + width * col)
        xb = raster_x(layout.values_left + width * (col + 1))
        top = rule(xa + 15, xb - 15, col, layout.header_band, layout.header_endpoints)
        bottom = rule(xa + 15, xb - 15, col, layout.footer_band, layout.footer_endpoints)
        if not 112 * scale < bottom - top < 141 * scale:
            continue
        ink = projection(image, xa + 15, top, xb - 15, bottom, 170)
        pitch = (bottom - top) / 8.8
        radius = round(pitch * 0.38)
        for row in range(len(ROW_NAMES)):
            target = round(pitch * (0.9 + row))
            candidates = range(max(radius + 1, target - round(pitch * 0.25)),
                               min(len(ink) - radius - 1, target + round(pitch * 0.30)))
            center = max(candidates, key=lambda y: (
                sum(ink[z] * (radius - abs(z - y)) for z in range(y - radius, y + radius + 1)), -abs(target - y),
            ))
            cells[row, col] = xa + 9, round(top + center - pitch * 0.48), xb - 9, round(top + center + pitch * 0.48)
    return cells


def recognize_cells(image: ScanImage, cells: dict[tuple[int, int], tuple[int, int, int, int]], engine: str,
                    white: int) -> dict[tuple[int, int], str]:
    # Give every cell its own whitespace, then OCR the tiles as one image so
    # tightly spaced source rows and columns cannot flow into each other.
    x_pitch, y_pitch = 300, 100
    width, height = x_pitch * 9, y_pitch * len(ROW_NAMES)
    pixels = bytearray(b"\xff") * (width * height)
    for (row, col), (x0, y0, x1, y1) in cells.items():
        clip = contrast(image.crop(x0, y0, x1, y1), white)
        for y in range(clip.height):
            target = (row * y_pitch + 20 + y) * width + col * x_pitch + 35
            pixels[target:target + clip.width] = clip.pixels[y * clip.width:(y + 1) * clip.width]
    tile_sheet = ScanImage(width, height, bytes(pixels), image.dpi, width, height)
    readings = defaultdict(list)
    for word in recognize(tile_sheet, engine, digits_only=True):
        readings[int(word.y) // y_pitch, int(word.x) // x_pitch].append(word)
    return {cell: "".join(word.text for word in sorted(words, key=lambda word: word.x))
            for cell, words in readings.items()}


def aligned_row_labels(image: ScanImage, words: tuple[Word, ...], cells, page_image: ScanImage) -> list[str]:
    lines = ordered_lines(words)
    labels = [""] * len(ROW_NAMES)
    for row, (pattern, _) in enumerate(ROW_NAMES):
        first_cell = cells.get((row, 0))
        if first_cell is None:
            continue
        box = page_image.source_bbox(*first_cell)
        for line in lines:
            match = re.search(pattern, text(line), re.I)
            if not match:
                continue
            _, y0, _, y1 = image.source_bbox(line[0].x0, line[0].y0, line[0].x1, line[0].y1)
            if abs((y0 + y1) / 2 - (box[1] + box[3]) / 2) <= (box[3] - box[1]) * 0.45:
                labels[row] = match[0]
                break
    return labels


def read_training_scan_pairs(pages: list[Page], tables: list[Table]) -> list[Table]:
    occupied = {(table.page.number, table.label.casefold()) for table in tables}
    result = []
    for page in pages:
        start = CAPTION_START.search(page.text)
        if not page.scan or not start or (page.number, start[1].casefold()) in occupied:
            continue
        fragments = page.text[start.start():start.end() + 750]
        model = CAPTION_MODEL.search(fragments)
        runs = CAPTION_RUNS.search(fragments)
        if (not model or not runs or not CAPTION_SCALE.search(fragments) or "sample" not in fragments
                or not (model[0].endswith("Instruct") or re.search(r"\br?uct\.", fragments[model.end():]))):
            continue
        caption = (f"{start[1]}: {start[2]} OAS comparison with {model[1]}Instruct. Exact-match percentages "
                   f"are reported as mean (sample SD) over two runs (n = {runs[1]})")
        layout = LAYOUTS[start[2].casefold()]
        x0, y0, x1, y1 = layout.region
        try:
            scan = scan_region(page.scan, (x0 / 1117, y0 / 1579, x1 / 1117, y1 / 1579), dpi=405)
            aligned = deskew(scan)
            engine = page.scan.engine
            # The layout page OCR preserves budget numbers better than OCR of
            # the tiny caption-aligned header; the header must precede the caption.
            section = page.text[max(0, start.start() - 4500):start.start()]
            header_start = re.search(r"Top-k\s*=\s*2", section, re.I)
            if not header_start:
                continue
            methods_at = re.search(r"\bMethod\b", section[header_start.start():], re.I)
            if not methods_at:
                continue
            header = section[header_start.start():header_start.start() + methods_at.start() + 400]
            header = re.sub(r"Top-k\s+\w{1,6}\s*=", "Top-k =", header, flags=re.I)
            header = header.replace("_", " ")
            header_y = round((sum(layout.header_endpoints) / 2 - 1) * 3) - aligned.top
            heading_region = aligned.crop(0, 0, aligned.width, header_y + 10)
            heading = " ".join(word.text for word in sorted(recognize(heading_region, engine, 11), key=lambda word: word.x))
            if len(DATASET.findall(heading.replace("_", " "))) != 9:
                heading = header
            header = " ".join([*(budget[0] for budget in BUDGET.finditer(header)),
                               *(dataset[0] for dataset in DATASET.finditer(heading.replace("_", " ")))])
            label_left = max(0, (layout.values_left - 1) * 3 - aligned.left)
            label_bottom = round((sum(layout.footer_endpoints) / 2 + 4) * 3) - aligned.top
            body_top = header_y - 20
            body = aligned.crop(0, body_top, label_left, label_bottom)
            cells = crop_cells(scan, layout)
            rows = aligned_row_labels(body, recognize(body, engine), cells, scan)
            first, second = (recognize_cells(scan, cells, engine, white) for white in (184, 200))
            result.extend(paired_tables(page, caption, header, rows, {
                key: (first.get(key, ""), second.get(key, ""), scan.source_bbox(*box)) for key, box in cells.items()
            }))
        except (ExtractionError, ValueError):
            continue
    return result
