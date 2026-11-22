import unittest

from pdf_pages import Word
from vector_histogram import read_histogram_geometry


SIZE = (595.276, 841.89)
BLUE = "rgb(12.156677%, 46.665955%, 70.587158%)"
ORANGE = "rgb(100%, 49.803162%, 5.490112%)"


def rectangle(x, y, right, bottom, fill, opacity):
    return (f'<path fill="{fill}" fill-opacity="{opacity}" '
            f'd="M {x * 100} {bottom * 100} L {right * 100} {bottom * 100} '
            f'L {right * 100} {y * 100} L {x * 100} {y * 100} Z"/>')


def tick(x, y, end_x, end_y):
    return (f'<path fill="none" stroke="rgb(0%, 0%, 0%)" '
            f'd="M {x * 100} {y * 100} L {end_x * 100} {end_y * 100}"/>')


# D014 page 13, Figure 7. The source chart has a different grid width for
# each dataset, with gaps where the orange bars have no visible height.
BLUE_TOPS = (
    695.4692578, 691.7326953, 688.5848438, 679.6579297, 675.0563281, 675.6810937,
    677.1228516, 679.4897266, 682.2771484, 685.7013281, 690.6273438, 691.6966406,
    694.0995703, 696.2982422, 697.5838281, 697.9682813, 698.0764453, 698.4729297,
    698.6771484, 698.7011719, 698.8213281, 698.7973047, 698.9414844, 698.9535156, 698.9414844,
)
ORANGE_TOPS = {
    1: 566.8823828, 5: 698.9471875, 6: 698.9288672, 7: 698.8647656, 8: 698.5716797,
    9: 697.8480859, 11: 697.3077344, 13: 695.7873047, 15: 698.7457031, 16: 698.9471875,
    17: 692.8655469, 19: 698.8555859, 21: 696.3734766, 22: 698.7273828, 25: 660.1401563,
    12: 698.9563672, 23: 698.9563672,
}
LEFT, TOP, RIGHT, BASE = (330.3946484, 560.2782031, 516.5117969, 698.9655078)
ORIGIN = 338.8544922
PARTS = [
    rectangle(LEFT, TOP, RIGHT, BASE, "rgb(100%, 100%, 100%)", "1"),
    rectangle(440.9694531, 565.6114453, 452.0875391, 569.5027734, BLUE, "0.6"),
    rectangle(440.9694531, 573.7676172, 452.0875391, 577.6589453, ORANGE, "0.6"),
    *(tick(x, BASE, x, BASE + 1.946) for x in (ORIGIN, 372.6939843, 508.0519141)),
    *(tick(LEFT, y, LEFT - 1.946, y) for y in (680.2935156, 624.2794142, 568.2699613)),
]
for color, tops, width in ((BLUE, dict(enumerate(BLUE_TOPS, 1)), 5.5145453), (ORANGE, ORANGE_TOPS, 6.767897)):
    PARTS.extend(rectangle(ORIGIN + width * (i - 1), y, ORIGIN + width * i, BASE, color, "0.6")
                 for i, y in tops.items())
SVG = '<svg viewBox="0 0 59528 84189" xmlns="http://www.w3.org/2000/svg">' + "".join(PARTS) + "</svg>"

WORDS = (
    Word("Answer", 327.794822, 547.544074, 352.393336, 560.175162),
    Word("F1", 354.517237, 547.544074, 362.605426, 560.175162),
    Word("Distribution", 364.729327, 547.544074, 403.700906, 560.175162),
    Word("Given", 405.824807, 547.544074, 425.153642, 560.175162),
    Word("Single", 427.277543, 547.544074, 447.815265, 560.175162),
    Word("Knowledge", 449.939166, 547.544074, 486.659945, 560.175162),
    Word("Sentence", 488.783846, 547.544074, 520.108046, 560.175162),
    Word("Wizard", 457.086982, 561.797978, 476.077070, 572.323885),
    Word("of", 477.846987, 561.797978, 483.212397, 572.323885),
    Word("Wikipedia", 484.982315, 561.797978, 512.132877, 572.323885),
    Word("HotpotQA", 457.086982, 569.964018, 484.170059, 580.489925),
    Word("Fraction", 306.268574, 619.264022, 316.794481, 641.488870),
    Word("0.1", 318.045144, 674.837184, 326.894732, 685.363091),
    Word("0.4", 318.045144, 618.757065, 326.894732, 629.282972),
    Word("0.7", 318.045144, 562.676945, 326.894732, 573.202852),
    Word("0.0", 334.837825, 699.539859, 343.687412, 710.065766),
    Word("0.2", 368.718235, 699.539859, 377.567822, 710.065766),
    Word("1.0", 504.239876, 699.539859, 513.089463, 710.065766),
    Word("Answer", 400.776090, 707.149321, 421.274851, 717.675228),
    Word("F1", 423.044769, 707.149321, 429.784926, 717.675228),
    Word("Score", 431.554844, 707.149321, 447.145284, 717.675228),
    Word("Figure", 306.144307, 729.832954, 332.053902, 741.838487),
    Word("7:", 334.534713, 729.832954, 342.254998, 741.838487),
)


class HistogramGeometryTest(unittest.TestCase):
    def test_maps_overlapping_bins_to_the_positioned_legend_and_skips_absent_bars(self):
        bars = read_histogram_geometry(SVG, WORDS, SIZE, "Figure 7", expansion=100)
        self.assertEqual(len(bars), 42)
        indexed = {(bar.dataset, bar.number): bar for bar in bars}
        blue, orange = indexed["WoW", 1], indexed["HotpotQA", 1]
        self.assertEqual((blue.metric_variant, f"{blue.value:.4f}"), ("bin1 [0.00000, 0.03259)", "0.0187"))
        self.assertEqual((orange.metric_variant, f"{orange.value:.4f}"), ("bin1 [0.00000, 0.04000)", "0.7074"))
        self.assertEqual(indexed["HotpotQA", 25].metric_variant, "bin25 [0.96000, 1.00000]")
        self.assertNotIn(("HotpotQA", 2), indexed)
        self.assertLess(orange.bbox[0], blue.bbox[2])
        self.assertLess(orange.bbox[1], blue.bbox[1])
        for caption, words in (("Figure 6", WORDS), ("Figure 7", WORDS[:10] + WORDS[11:])):
            with self.subTest(caption=caption, word_count=len(words)):
                self.assertEqual(read_histogram_geometry(SVG, words, SIZE, caption, expansion=100), ())

    def test_rejects_unequal_grid_counts_even_if_remaining_heights_sum_to_one(self):
        last_blue_bar = PARTS[-len(ORANGE_TOPS) - 1]
        self.assertEqual(read_histogram_geometry(SVG.replace(last_blue_bar, ""), WORDS, SIZE, "Figure 7",
                                                 expansion=100), ())
