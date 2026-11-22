from types import SimpleNamespace
import unittest

from pdf_pages import Page
from publication import Payload, add_scatter_figures


class ScatterPublicationTest(unittest.TestCase):
    def figure(self, number, metric, first, *, vertical="Answer F1",
               condition="unconstrained knowledge sampling"):
        label = f"Figure {number}"
        page = Page(number + 11, f"{label}: Knowledge versus Answer F1.", "text_match")
        positions = [
            SimpleNamespace(x=.1342, y=first, bbox=(.18, .34, .19, .35)),
            SimpleNamespace(x=.1342, y=.1992, bbox=(.28, .34, .29, .35)),
        ]
        panel = SimpleNamespace(dataset="WoW", generator="GPT-4o-mini", points=positions)
        return SimpleNamespace(page=page, label=label, metric=metric, panels=(panel,),
                               vertical_metric=vertical, condition=condition)

    def test_equal_values_in_different_points_remain_separate(self):
        data = Payload(document={}, publication={})
        add_scatter_figures(data, [
            self.figure(10, "Knowledge precision", .1992),
            self.figure(11, "Knowledge recall", .1992),
        ])
        x, y, other_x, other_y, _, repeated, _, other_repeated = data.results
        self.assertEqual((x["metric_name"], x["raw_value"], x["reported_scale"], x["raw_unit"]),
                         ("Knowledge precision", "~0.1342", "fraction", None))
        self.assertEqual(y["metric_name"], "Answer F1")
        self.assertNotEqual(y["result_id"], other_y["result_id"])
        self.assertNotIn("duplicate_of", other_x)
        self.assertEqual(repeated["duplicate_of"], y["result_id"])
        self.assertEqual(other_repeated["duplicate_of"], other_y["result_id"])
        source, = [e for e in data.evidence if e["entity_id"] == repeated["result_id"]]
        self.assertEqual((source["page"], source["figure"], source["bbox"]),
                         (22, "Figure 11", (.18, .34, .19, .35)))
        self.assertIn("point 1", repeated["note"])

    def test_a_different_answer_coordinate_remains_canonical(self):
        data = Payload(document={}, publication={})
        add_scatter_figures(data, [
            self.figure(10, "Knowledge precision", .1992),
            self.figure(11, "Knowledge recall", .2107),
        ])
        self.assertNotIn("duplicate_of", data.results[5])

    def test_length_conditions_are_separate_and_recall_is_the_vertical_axis(self):
        data = Payload(document={}, publication={})
        add_scatter_figures(data, [
            self.figure(8, "Knowledge precision", .1992, vertical="Knowledge recall",
                        condition="knowledge selection length analysis"),
            self.figure(9, "Knowledge precision", .1992, vertical="Knowledge recall",
                        condition="knowledge selection capped at 3 sentences"),
        ])
        self.assertEqual((data.results[5]["metric_name"], data.results[5]["raw_value"]),
                         ("Knowledge recall", "~0.1992"))
        self.assertNotIn("duplicate_of", data.results[5])
        self.assertNotEqual(data.results[1]["experiment_id"], data.results[5]["experiment_id"])


if __name__ == "__main__":
    unittest.main()
