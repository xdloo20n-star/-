import unittest

from pdf_pages import Word
from vector_scatter import read_panel_geometry


# D014 PDF page 16, Figure 10, GPT-4o-mini / Wizard of Wikipedia panel.
# SVG coordinates are from the marker and axis paths exposed by Poppler.
MARK = "M 0 -1.58 C 0.42 -1.58 0.82 -1.41 1.12 -1.12 C 1.41 -0.82 1.58 -0.42 1.58 0 " \
       "C 1.58 0.42 1.41 0.82 1.12 1.12 C 0.82 1.41 0.42 1.58 0 1.58 " \
       "C -0.42 1.58 -0.82 1.41 -1.12 1.12 C -1.41 0.82 -1.58 0.42 -1.58 0 " \
       "C -1.58 -0.42 -1.41 -0.82 -1.12 -1.12 C -0.82 -1.41 -0.42 -1.58 0 -1.58 Z M 0 -1.58"
SVG = f"""<svg viewBox="0 0 595.276 841.89" xmlns="http://www.w3.org/2000/svg">
<path fill="rgb(100%, 100%, 100%)" fill-opacity="1" d="M 100.203125 298.601562 L 268.726562 298.601562 L 268.726562 173.023438 L 100.203125 173.023438 Z M 100.203125 298.601562"/>
<path fill="rgb(12%, 47%, 71%)" fill-opacity="0.5" stroke="rgb(12%, 47%, 71%)" stroke-opacity="0.5" d="{MARK}" transform="matrix(0.47189, 0, 0, -0.47189, 107.863745, 292.892375)"/>
<path fill="none" stroke="rgb(0%, 0%, 0%)" d="M 107.863745 298.601562 L 107.863745 300.25"/>
<path fill="none" stroke="rgb(0%, 0%, 0%)" d="M 138.503971 298.601562 L 138.503971 300.25"/>
<path fill="none" stroke="rgb(0%, 0%, 0%)" d="M 169.144199 298.601562 L 169.144199 300.25"/>
<path fill="none" stroke="rgb(0%, 0%, 0%)" d="M 100.203125 291.6825 L 98.55 291.6825"/>
<path fill="none" stroke="rgb(0%, 0%, 0%)" d="M 100.203125 277.4235 L 98.55 277.4235"/>
<path fill="none" stroke="rgb(0%, 0%, 0%)" d="M 100.203125 263.1645 L 98.55 263.1645"/>
</svg>"""

WORDS = (
    Word("0.0", 104.2374, 299.5761, 111.7495, 307.5891),
    Word("0.2", 134.9149, 299.5761, 142.427, 307.5891),
    Word("0.4", 165.5923, 299.5761, 173.1045, 307.5891),
    Word("0.20", 86.4986, 287.5487, 97.0157, 295.5617),
    Word("0.21", 86.4986, 273.2727, 97.0157, 281.2857),
    Word("0.22", 86.4986, 258.9966, 97.0157, 267.0096),
)


class VectorScatterTest(unittest.TestCase):
    def test_reads_marker_using_both_pdf_tick_axes_and_preserves_bbox(self):
        panels = read_panel_geometry(SVG, WORDS, (595.276, 841.89))
        self.assertEqual([len(panel) for panel in panels], [1])
        point = panels[0][0]
        self.assertEqual((f"{point.x:.4f}", f"{point.y:.4f}"), ("0.0000", "0.1992"))
        self.assertLess(point.bbox[0], 0.1814)
        self.assertGreater(point.bbox[2], 0.1814)
        self.assertLess(point.bbox[1], 0.348)
        self.assertGreater(point.bbox[3], 0.348)

    def test_rejects_non_linear_or_incomplete_tick_labels(self):
        altered = (*WORDS[:-1], Word("0.29", 86.4986, 258.9966, 97.0157, 267.0096))
        for words in (altered, WORDS[:-1]):
            with self.subTest(words=words):
                self.assertEqual(read_panel_geometry(SVG, words, (595.276, 841.89)), ())

    def test_keeps_base_paths_in_length_panels_and_ignores_highlight_overlays(self):
        # On page 14 Poppler draws some colored points as a fill and a separate stroke.
        # Large overlays are threshold annotations over the same observations.
        colored = SVG.replace('fill="rgb(12%, 47%, 71%)" fill-opacity="0.5"', 'fill="none"').replace(
            'stroke-opacity="0.5"', 'stroke-opacity="0.8"')
        highlight = (f'<path fill="none" stroke="rgb(100%, 65%, 0%)" stroke-opacity="0.8" '
                     f'd="{MARK}" transform="matrix(5, 0, 0, -5, 107.863745, 292.892375)"/>')
        panels = read_panel_geometry(colored.replace('</svg>', highlight + '</svg>'), WORDS, (595.276, 841.89),
                                     marker_opacity="0.8")
        self.assertEqual([len(panel) for panel in panels], [1])
        self.assertEqual(f"{panels[0][0].y:.4f}", "0.1992")
