"""Read evaluator scores from scanned factual-grounding and accuracy grids."""

from dataclasses import dataclass
import re

from pdf_pages import ExtractionError, Page, Word
from rag_tables import Column, Header, Table
from scan_ocr import ScanImage, deskew, recognize, scan_region


SCORE = re.compile(r"(?:0\.\d{2}|1\.00)")
JUDGE = re.compile(r"\(?GPT-4[Oo0]\)?")
METHODS = ("Deepchecks", "RAGAS", "Langsmith")


@dataclass(frozen=True)
class EvaluationLayout:
    label: str
    caption_anchor: re.Pattern[str]
    metric_anchor: re.Pattern[str]
    metric: str
    task: str
    bounds: tuple[float, float, float, float]
    profiles: tuple[tuple[int, int], ...]
    heading_bottom: float
    method_x: tuple[float, float, float]
    score_x: tuple[float, float, float]
    rows: tuple[tuple[str, float], ...]
    headings: tuple[str, str, str]


GROUNDING = EvaluationLayout(
    "Table 1", re.compile(r"\bTable\s+1\s*[.:]", re.I), re.compile(r"\bfactual\s+grounding\b", re.I),
    "ROC-AUC", "groundedness evaluation", (.321, .592, .691, .718), ((300, 11), (420, 6)), .650,
    (.455, .545, .635), (.460, .550, .641), (("TRUE", .663), ("SQuAD", .676), ("PubmedQA", .689)),
    ("Deepchecks Grounded in Context", "RAGAS Faithfulness", "Langsmith Answer Faithfulness"),
)
ACCURACY = EvaluationLayout(
    "Table 2", re.compile(r"\b[TF]able\s+2[.:]", re.I), re.compile(r"\bsured\s+by\s+accuracy\b|measured\s+by\s+accuracy\b", re.I),
    "Evaluator accuracy", "interaction quality evaluation", (.230, .651, .800, .781),
    ((300, 11), (240, 6), (420, 6)), .685, (.456, .530, .673), (.460, .570, .713),
    (("RAG-dataset-12000", .688), ("SQUAD", .701), ("HAGRID", .714),
     ("Client 1 (Tech Support)", .726), ("Client 2 (Job Profiles)", .739)), METHODS,
)


def evaluation_tables(page: Page, layout: EvaluationLayout,
                      views: tuple[tuple[ScanImage, tuple[Word, ...]], ...]) -> list[Table]:
    def placed_words(image, words):
        return tuple((w, (image.left + w.x) / image.page_width, (image.top + w.y) / image.page_height) for w in words)

    positioned = tuple((image, placed_words(image, words)) for image, words in views)
    headings = [(word, x) for _, words in positioned for word, x, y in words if y < layout.heading_bottom]
    if not any(w.text.casefold() in {"dataset", "datase"} and x < .395 for w, x in headings):
        return []
    if not all(any(w.text == name and abs(x - center) < .020 for w, x in headings)
               for name, center in zip(METHODS, layout.method_x, strict=True)):
        return []
    if layout is GROUNDING and not {"Grounded", "Context", "Faithfulness"}.issubset(w.text for w, _ in headings):
        return []

    def observed_dataset(words, label, y):
        # The second grid's left edge slopes down relative to its score columns.
        slope, tolerance = (0, .006) if layout is GROUNDING else (.045, .005)
        row = (w.text for w, x, py in sorted(words, key=lambda entry: entry[1])
               if x < .408 and abs(py - y - (.46 - x) * slope) < tolerance)
        return "".join(row).strip("[]{}|_").casefold() == "".join(label.split()).casefold()

    tables = []
    for method_index, (method, method_title, center) in enumerate(zip(METHODS, layout.headings, layout.score_x, strict=True)):
        judge = None
        if method != "Deepchecks":
            # Tesseract frequently confuses the final letter o with a zero.
            judge_x = center if layout is GROUNDING else (.598 if method == "RAGAS" else .752)
            if any(JUDGE.fullmatch(w.text) and abs(x - judge_x) < .024 for w, x in headings):
                judge = "GPT-4o"
                method_title += " (GPT-4o)"
        columns, values, boxes = [], [], []
        for dataset, row_y in layout.rows:
            if not any(observed_dataset(words, dataset, row_y) for _, words in positioned):
                continue
            readings = {}
            for image, words in positioned:
                matches = [w for w, x, y in words if abs(x - center) < .014 and abs(y - row_y) < .005
                           and SCORE.fullmatch(w.text)]
                if len(matches) == 1:
                    word = matches[0]
                    box = image.source_bbox(word.x0, word.y0, word.x1, word.y1)
                    readings.setdefault(word.text, []).append(box)
            agreed = [(raw, locations[0]) for raw, locations in readings.items() if len(locations) >= 2]
            if len(agreed) != 1:
                continue
            raw, box = agreed[0]
            columns.append(Column(Header(method_title, layout.metric, center, None, None), dataset, generator=judge,
                                  original_row=dataset, source_label=method_title, reported_scale="fraction"))
            values.append(raw)
            boxes.append(box)
        if values:
            configuration = layout.headings[0] if method_index == 0 else method
            tables.append(Table(page, layout.label, f"{layout.label}. {layout.metric}", method_index, columns,
                                [(configuration, values)], config_name=configuration, task_type=layout.task,
                                value_bboxes=(tuple(boxes),)))
    return tables


def read_evaluation_scan_tables(pages: list[Page], tables: list[Table]) -> list[Table]:
    occupied = {(table.page.number, table.label.casefold()) for table in tables}
    found = []
    for page in pages:
        if page.scan is None:
            continue
        for layout in (GROUNDING, ACCURACY):
            if ((page.number, layout.label.casefold()) in occupied or not layout.caption_anchor.search(page.text)
                    or not layout.metric_anchor.search(page.text)):
                continue
            try:
                if layout is GROUNDING and not re.search(r"\bROC-AUC\b", page.text):
                    protocol = scan_region(page.scan, (.206, .529, .822, .596), dpi=300)
                    if not any(w.text == "ROC-AUC" for w in recognize(protocol, page.scan.engine)):
                        continue
                views = []
                for dpi, psm in layout.profiles:
                    image = deskew(scan_region(page.scan, layout.bounds, dpi=dpi))
                    views.append((image, recognize(image, page.scan.engine, psm=psm)))
                found.extend(evaluation_tables(page, layout, tuple(views)))
            except (ExtractionError, ValueError):
                continue
    return found
