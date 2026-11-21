"""Read checkpoint score pairs when the caption identifies both metrics."""

import re

from pdf_pages import Page
from rag_tables import CAPTION, Column, Header, Table, column_boundaries


FEVER_CAPTION = re.compile(r"^\s*Table\s+\d+[:.]\s*FEVER fact-verification results\b", re.I)
FEVER_ORDER = re.compile(r"\blabel accuracy\s*/\s*nVDD\b", re.I)
STEP = re.compile(r"\bStep (\d+)\b", re.I)
SCORE_PAIR = re.compile(r"(?<!\S)(0[.,]\d+)\s*/\s*(0[.,]\d+)(?=\s|$)")
METHOD = re.compile(r"^\s*(PPO|OAS)\s{2,}")
MODEL = re.compile(r"\bQwen2\.5-\s*1\.5B-Instruct\b")
FIXED_BUDGET = re.compile(r"\bTop-(\d+) retrieval\b", re.I)
FINAL_NVDD = re.compile(r"\bfalls to (0\.\d+) at step (\d+),\s*yielding a final nVDD of (0\.\d+)\b", re.I)


def read_training_tables(pages: list[Page], tables: list[Table]) -> list[Table]:
    occupied = {(table.page.number, table.label.casefold()) for table in tables}
    ppo_name = "Search-R1 PPO" if any("Search-R1 PPO" in page.prose for page in pages) else "PPO"
    found = []
    for page in pages:
        lines = page.text.splitlines()
        for index, line in enumerate(lines):
            if not FEVER_CAPTION.match(line):
                continue
            label = CAPTION.match(line)[1]
            if (page.number, label.casefold()) in occupied:
                continue
            fragment_lines = []
            margin = len(line) - len(line.lstrip())
            for part in lines[index:index + 9]:
                if part.strip() and len(part) - len(part.lstrip()) <= margin + 4:
                    fragment_lines.append(re.split(r"\s{5,}", part.strip(), maxsplit=1)[0])
                if FEVER_ORDER.search(" ".join(fragment_lines)):
                    break
            fragment = "\n".join(fragment_lines)
            caption = " ".join(fragment.split()).replace("re- ports", "reports")
            order = FEVER_ORDER.search(fragment)
            budget = FIXED_BUDGET.search(fragment)
            if not order or not budget:
                continue
            model = MODEL.search(fragment)
            generator = "".join(model[0].split()) if model else None
            prose = " ".join(re.split(r"\s{5,}", line.lstrip(), maxsplit=1)[0]
                             for line in lines[index + len(fragment_lines):index + 17]
                             if line.strip() and len(line) - len(line.lstrip()) <= margin + 4)
            found.extend(checkpoint_rows(page, lines, index, label, caption, generator, budget, ppo_name,
                                         FINAL_NVDD.search(prose)))
    return found


def checkpoint_rows(page: Page, lines: list[str], index: int, label: str, caption: str,
                    generator: str | None, budget: re.Match[str], ppo_name: str,
                    final_nvdd: re.Match[str] | None) -> list[Table]:
    for header_at in range(index - 1, max(-1, index - 8), -1):
        if re.match(r"^\s*Method\s{2,}Step\b", lines[header_at], re.I):
            break
    else:
        return []
    headings = list(STEP.finditer(lines[header_at]))
    if len(headings) < 2 or len({step[1] for step in headings}) != len(headings):
        return []
    centers = [(step.start() + step.end()) / 2 for step in headings]
    boundaries = column_boundaries(centers)
    found = []
    for offset, line in enumerate(lines[header_at + 1:index], start=header_at + 1):
        method = METHOD.match(line)
        if not method:
            continue
        pairs = list(SCORE_PAIR.finditer(line, method.start(1)))
        if len(pairs) != len(headings):
            continue
        if any(not left <= (pair.start() + pair.end()) / 2 < right
               for pair, left, right in zip(pairs, boundaries[:-1], boundaries[1:], strict=True)):
            continue
        columns = []
        values = []
        for step, center, pair in zip(headings, centers, pairs, strict=True):
            for name, printed, raw in zip(("Label accuracy", "nVDD"), ("label accuracy", "nVDD"), pair.groups(), strict=True):
                # The adjacent prose can disambiguate a scanned decimal in the PPO final checkpoint.
                if "," in raw:
                    if (page.verification != "ocr_match" or name != "nVDD" or method[1] != "PPO"
                            or not final_nvdd or final_nvdd.groups() != (pair[1], step[1], raw.replace(",", "."))):
                        continue
                    raw = final_nvdd[3]
                columns.append(Column(Header(printed, name, center, None, None), "FEVER", generator=generator,
                                      metric_variant=f"step {step[1]}", metric_variant_evidence=(page, step[0]),
                                      source_label=f"{step[0]} / {printed}",
                                      direction="higher_is_better" if name == "Label accuracy" else "lower_is_better"))
                values.append(raw)
        if values:
            name = ppo_name if method[1] == "PPO" else method[1]
            found.append(Table(page, label, caption, offset, columns, [(method[1], values)], config_name=name,
                               top_k=int(budget[1]), top_k_evidence=(page, budget[0])))
    return found
