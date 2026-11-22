"""Read the paired accuracy grid for synthetic training with and without RAG."""

import re

from failure_rates import visual_rows
from native_benchmark_tables import normalized_boxes, positioned_caption, words_text
from pdf_pages import Page, Word
from rag_tables import Column, Header, NATIVE_CAPTION, Table
from retrieval_tables import source_statement


CAPTION = re.compile(r"Our training complements retrieval augmentation\.")
DELTA_DEFINITION = re.compile(r"[∆Δ] denotes the absolute improvement of “Ours \+ RAG” over “Vanilla RAG”\.")
EVALUATION = re.compile(r"\bWe generate eight responses per question \(n\s*=\s*8\) and report the average\s+accuracy\b")
RAG_DEPTH = re.compile(r"\bre-?\s*rank these chunks and select\s+the top-(\d+) as context\b", re.I)
MODELS = {"Llama": re.compile(r"\bLlama 3\.1 8B\s+Instruct\b"), "Qwen": re.compile(r"\bQwen3 8B\b")}
HEADINGS = ("Ours", "Ours + RAG", "Vanilla RAG", "∆")
SCORE = re.compile(r"(?:100(?:\.0+)?|\d{1,2}(?:\.\d+)?)%")
DELTA = re.compile(r"[+−-]\d+(?:\.\d+)?")


def read_synthetic_accuracy(pages: list[Page], tables: list[Table]) -> list[Table]:
    native = [page for page in pages if page.verification == "text_match"]
    evaluation = source_statement(native, EVALUATION)
    if evaluation is None:
        return tables
    model_sources = {name: source_statement(native, pattern) for name, pattern in MODELS.items()}
    depth = source_statement(native, RAG_DEPTH)
    replacements = {}
    for page in native:
        if not page.words or not CAPTION.search(page.text):
            continue
        rows = visual_rows(page.words)
        lines = page.text.splitlines()
        for index, line in enumerate(lines):
            locator = NATIVE_CAPTION.match(line)
            if not locator or not CAPTION.match(line[locator.end():].lstrip()):
                continue
            label = locator[1]
            anchor = positioned_caption(rows, label, CAPTION)
            if not anchor:
                continue
            caption = " ".join(" ".join(lines[index:index + 4]).split())
            if not DELTA_DEFINITION.search(caption):
                continue
            parsed = accuracy_rows(rows, anchor)
            if parsed:
                replacements[page.number, label.casefold()] = [
                    make_cell(page, label, caption, index, dataset, model, i, value, model_sources[model], evaluation, depth)
                    for dataset, model, values in parsed for i, value in enumerate(values)
                ]
    return ([table for table in tables if (table.page.number, table.label.casefold()) not in replacements]
            + [table for group in replacements.values() for table in group])


def accuracy_rows(rows: list[list[Word]], anchor: Word) -> list[tuple[str, str, list[Word]]]:
    for index, row in enumerate(rows):
        if 0 < anchor.y - row[0].y < 140 and words_text(row) == "Benchmark Model " + " ".join(HEADINGS):
            break
    else:
        return []
    edges = [row[i].x1 for i in (2, 5, 7, 8)]
    grid = [[word for word in line if row[0].x0 - 4 < word.x < edges[-1] + 10]
            for line in rows[index + 1:] if line[0].y < anchor.y - 3]
    grid = [line for line in grid if line]
    if not grid or len(grid) % 3:
        return []
    results = []
    for offset in range(0, len(grid), 3):
        first, group, second = grid[offset:offset + 3]
        if (len(group) != 1 or abs(group[0].x0 - row[0].x0) > 4
                or not re.fullmatch(r"[A-Za-z][\w.-]+", group[0].text)):
            return []
        for expected, line in zip(MODELS, (first, second), strict=True):
            if len(line) != 5 or line[0].text != expected or abs(line[0].x0 - row[1].x0) > 4:
                return []
            cells = line[1:]
            if not (all(SCORE.fullmatch(word.text) for word in cells[:3]) and DELTA.fullmatch(cells[3].text)
                    and all(abs(word.x1 - edge) <= 8 for word, edge in zip(cells, edges, strict=True))):
                return []
            results.append((group[0].text, expected, cells))
    return results


def make_cell(page, label, caption, index, dataset, model, i, value, model_source, evaluation, depth):
    heading = HEADINGS[i]
    delta = heading == "∆"
    generator = model_source[1][0] if model_source else None
    column = Column(
        Header(heading, "Accuracy gain over Vanilla RAG" if delta else "Accuracy", value.x, None, None),
        dataset, None if delta else "mean", " ".join(generator.split()) if generator else None,
        aggregation_evidence=(evaluation[0], evaluation[1][0]) if not delta else None,
        generator_evidence=(model_source[0], generator) if model_source else None,
        source_label=heading,
    )
    retrieval = depth if i in (1, 2) else None
    return Table(page, label, caption, index, [column], [(model, [value.text])], row_section=dataset,
                 config_name="Ours + RAG vs Vanilla RAG" if delta else heading,
                 top_k=int(retrieval[1][1]) if retrieval else None,
                 top_k_evidence=(retrieval[0], retrieval[1][0]) if retrieval else None,
                 value_bboxes=normalized_boxes(page, [value]))
