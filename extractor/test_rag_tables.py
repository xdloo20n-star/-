import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from pdf_pages import Page, read_pdf
from rag_tables import read_tables


def layout_line(*cells: tuple[int, str]) -> str:
    line = ""
    for start, value in cells:
        if start < len(line):
            raise ValueError(f"Overlapping cell at {start}: {value}")
        line = line.ljust(start) + value
    return line


BENCHMARK_COMPARISON = [
    layout_line((2, "Benchmark"), (20, "Model"), (35, "Hybrid"), (51, "Baseline"), (68, "Δ")),
    layout_line((20, "Small"), (35, "87.2%"), (51, "80.5%"), (68, "+6.7")),
    layout_line((2, "News QA")),
    layout_line((20, "Large"), (35, "89.1%"), (51, "81.2%"), (68, "+7.9")),
    layout_line((20, "Small"), (35, "76.5%"), (51, "69.0%"), (68, "+7.5")),
    layout_line((2, "Medical QA")),
    layout_line((20, "Large"), (35, "82.3%"), (51, "79.9%"), (68, "+2.4")),
    "",
    "Table 2: Accuracy scores by benchmark.",
]


METRIC_COMPARISON = [
    layout_line((0, "Unrelated prose"), (28, "Table 4: Evaluation scores.")),
    layout_line((0, "More prose"), (30, "Metric"), (56, "Raw KB Enr. KB"), (79, "Δ")),
    layout_line((30, "Coverage (%)"), (59, "14.2"), (68, "16.1"), (78, "+13%*")),
    layout_line((0, "Other column"), (30, "Sentiment σ"), (59, "5.66"), (68, "7.78"), (78, "+37.5%")),
    layout_line((30, "Response time (s)"), (59, "1.04"), (68, "1.00"), (78, "−3.8%*")),
    layout_line((30, "Tokens"), (57, "14,320"), (67, "12,203"), (78, "−15%")),
]


class NativeComparisonTest(unittest.TestCase):
    def test_centered_benchmark_rows_preserve_score_and_coordinates(self):
        page = Page(4, "\n".join(BENCHMARK_COMPARISON), "text_match")
        table, = read_tables(page)
        self.assertEqual((table.page, table.label, table.index), (page, "Table 2", 0))
        self.assertEqual(
            [(column.dataset, column.generator, column.original_row) for column in table.columns],
            [("News QA", "Small", "News QA / Small"), ("News QA", "Large", "News QA / Large"),
             ("Medical QA", "Small", "Medical QA / Small"), ("Medical QA", "Large", "Medical QA / Large")],
        )
        self.assertEqual([(col.header.name, col.header.unit) for col in table.columns], [("Accuracy", "%")] * 4)
        self.assertEqual(table.rows, [
            ("Hybrid", ["87.2%", "89.1%", "76.5%", "82.3%"]),
            ("Baseline", ["80.5%", "81.2%", "69.0%", "79.9%"]),
        ])

    def test_benchmark_without_centered_pair_is_rejected(self):
        lines = BENCHMARK_COMPARISON.copy()
        lines[4], lines[6] = lines[6], lines[4]
        self.assertEqual(read_tables(Page(4, "\n".join(lines), "text_match")), [])

    def test_metric_rows_in_right_column_recover_literal_headings_and_values(self):
        page = Page(6, "\n".join(METRIC_COMPARISON), "text_match")
        table, = read_tables(page)
        self.assertEqual((table.page, table.label, table.index), (page, "Table 4", 1))
        self.assertEqual(table.caption, "Table 4: Evaluation scores.")
        self.assertEqual(
            [(col.original_row, col.header.label, col.header.name, col.header.unit) for col in table.columns],
            [("Coverage (%)", "Coverage (%)", "Coverage", "%"), ("Sentiment σ", "Sentiment σ", "Sentiment σ", None),
             ("Response time (s)", "Response time (s)", "Response time", "s")],
        )
        self.assertEqual(table.rows, [("Raw KB", ["14.2", "5.66", "1.04"]), ("Enr. KB", ["16.1", "7.78", "1.00"])])

    def test_metric_rows_with_ambiguous_heading_split_are_rejected(self):
        lines = [
            "Table 1: Evaluation scores.",
            layout_line((2, "Metric"), (21, "A B C D E"), (46, "Δ")),
            layout_line((2, "Score"), (20, "0.0123"), (29, "0.021"), (44, "+72%")),
            layout_line((2, "Coverage"), (20, "0.0345"), (29, "0.055"), (44, "+59%")),
        ]
        self.assertEqual(read_tables(Page(5, "\n".join(lines), "text_match")), [])

    def test_successive_footer_captions_belong_to_their_own_grids(self):
        heading = layout_line((2, "Metric"), (20, "Raw"), (30, "Enriched"), (44, "Δ"))
        lines = [
            heading,
            layout_line((2, "Coverage"), (20, "45"), (31, "50"), (44, "+5")),
            layout_line((2, "Relevance"), (20, "70"), (31, "71"), (44, "+1")),
            "  Table 1: Group A.",
            "",
            heading,
            layout_line((2, "Coverage"), (20, "52"), (31, "56"), (44, "+4")),
            layout_line((2, "Relevance"), (20, "62"), (31, "68"), (44, "+6")),
            "  Table 2: Group B.",
        ]
        tables = read_tables(Page(2, "\n".join(lines), "text_match"))
        self.assertEqual([(t.label, t.rows[0]) for t in tables],
                         [("Table 1", ("Raw", ["45", "70"])), ("Table 2", ("Raw", ["52", "62"]))])

    def test_native_fallback_preserves_ocr_path(self):
        self.assertEqual(read_tables(Page(6, "\n".join(METRIC_COMPARISON), "ocr_match")), [])


class NativeDevPDFTest(unittest.TestCase):
    def test_unsupported_and_baseline_dev_pdfs(self):
        archive = Path(__file__).resolve().parents[2] / "drive-download-20261120T125610Z-1-001.zip"
        if not archive.is_file() or not all(shutil.which(tool) for tool in ("pdfinfo", "pdftotext")):
            self.skipTest("The dev PDF archive and Poppler are needed for corpus verification")
        expected = {
            "D003": {(8, "Table 2"): 18},
            "D004": {(6, "Table 2"): 14, (17, "Table 12"): 14},
            "D020": {(5, "Table 5"): 6},
            "D001": {(6, "Table 1"): 252},
            "D019": {(5, "Table 1"): 154, (7, "Table 4"): 20},
        }
        with TemporaryDirectory() as work, ZipFile(archive) as corpus:
            with patch.dict(os.environ, {"TESSERACT_BIN": str(Path(work, "unavailable"))}):
                for document, counts in expected.items():
                    with self.subTest(document=document):
                        pdf = read_pdf(Path(corpus.extract(f"dev/{document}.pdf", path=work)))
                        tables = [table for page in pdf.pages for table in read_tables(page)]
                        actual = {(t.page.number, t.label): sum(len(values) for _, values in t.rows) for t in tables}
                        self.assertEqual(actual, counts)
                        for table in tables:
                            self.assertIn(table.label, table.page.text)
                            for _, values in table.rows:
                                self.assertEqual(len(values), len(table.columns))
                                self.assertTrue(all(value in table.page.text for value in values))


if __name__ == "__main__":
    unittest.main()
