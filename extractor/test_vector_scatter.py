from pathlib import Path
import unittest
from unittest.mock import patch

from pdf_pages import Page, Word
from vector_scatter import read_page_geometry, read_panel_geometry


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


# Page 16, Figure 10, GPT-4o-mini / Wizard of Wikipedia, point 2. Its PDF
# marker center is y=53.7326630364 in the plot's local coordinates. The
# printed ticks at 52.675889 and 294.40539 put Answer F1 at 0.20034974.
def second_point_svg(enlarged):
    if enlarged:
        view = "59528 84189"
        box = "M 10020.4375 29814.59375 L 26872.675781 29814.59375 L 26872.675781 17256.957031 L 10020.4375 17256.957031 Z"
        marker = "47.189286, 0, 0, -47.189286, 10786.447012, 29072.935922"
        tick_matrix = "47.189286, 0, 0, -47.189286, 7302.333324, 31608.541929"
        ticks = (
            "M 73.832691 38.016006 L 73.832691 34.515973",
            "M 138.763677 38.016006 L 138.763677 34.515973",
            "M 203.694581 38.016006 L 203.694581 34.515973",
            "M 57.600028 52.675882 L 54.099994 52.675882",
            "M 57.600028 173.540678 L 54.099994 173.540678",
            "M 57.600028 294.405391 L 54.099994 294.405391",
        )
    else:
        view = "595.276 841.89"
        box = "M 100.203125 298.601562 L 268.726562 298.601562 L 268.726562 173.023438 L 100.203125 173.023438 Z"
        marker = "0.47189, 0, 0, -0.47189, 107.863745, 291.183822"
        tick_matrix = "0.47189, 0, 0, -0.47189, 73.022843, 316.539711"
        ticks = (
            "M 73.831744 38.013437 L 73.831744 34.51189",
            "M 138.763497 38.013437 L 138.763497 34.51189",
            "M 203.695251 38.013437 L 203.695251 34.51189",
            "M 57.598805 52.673577 L 54.097259 52.673577",
            "M 57.598805 173.539019 L 54.097259 173.539019",
            "M 57.598805 294.404462 L 54.097259 294.404462",
        )
    tick_paths = "".join(
        f'<path fill="none" stroke="rgb(0%, 0%, 0%)" d="{tick}" transform="matrix({tick_matrix})"/>'
        for tick in ticks
    )
    return f"""<svg viewBox="0 0 {view}" xmlns="http://www.w3.org/2000/svg">
<path fill="rgb(100%, 100%, 100%)" fill-opacity="1" d="{box}"/>
<path fill="rgb(12%, 47%, 71%)" fill-opacity="0.5" stroke="rgb(12%, 47%, 71%)" stroke-opacity="0.5" d="{MARK}" transform="matrix({marker})"/>
{tick_paths}
</svg>"""


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

    def test_refines_real_marker_score_and_keeps_original_page_bbox(self):
        words = (*WORDS[:4],
                 Word("0.24", 86.4986, 230.5396, 97.0157, 238.5526),
                 Word("0.28", 86.4986, 173.4358, 97.0157, 181.4488))
        size = (595.276, 841.89)
        ordinary = second_point_svg(False)
        expanded = second_point_svg(True)
        for candidate, expected in (
            (expanded, "0.2003"),
            (expanded.replace('stroke-opacity="0.5"', 'stroke-opacity="0.9"'), "0.2004"),
        ):
            with self.subTest(expected=expected):
                with patch("vector_scatter.positioned_words", return_value=(words, size)), \
                        patch("vector_scatter.run", side_effect=[ordinary, candidate]):
                    ((point,),) = read_page_geometry(Path("figure.pdf"), Page(16, "", "text_match"))
                self.assertEqual((f"{point.x:.4f}", f"{point.y:.4f}"), ("0.0000", expected))
                self.assertAlmostEqual((point.bbox[0] + point.bbox[2]) / 2, 0.18119955, places=7)
                self.assertAlmostEqual((point.bbox[1] + point.bbox[3]) / 2, 0.34586920, places=7)
