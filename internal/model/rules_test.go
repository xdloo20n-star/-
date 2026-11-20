package model_test

import (
	"maps"
	"math"
	"testing"

	"sverka"
	"sverka/internal/extract"
	"sverka/internal/model"
)

func TestNormalization(t *testing.T) {
	for _, test := range []struct {
		name     string
		result   model.Record
		expected *model.MeasuredValue
	}{
		{"explicit percent", model.Record{"reported_value": 70.0, "reported_scale": "percent"}, &model.MeasuredValue{Value: 0.7, Unit: "fraction"}},
		{"unit with zero", model.Record{"reported_value": 0.0, "raw_unit": "%"}, &model.MeasuredValue{Value: 0, Unit: "fraction"}},
		{"explicit seconds", model.Record{"reported_value": 1.25, "raw_unit": "s"}, &model.MeasuredValue{Value: 1250, Unit: "ms"}},
		{"unknown scale", model.Record{"reported_value": 0.7, "metric_name": "Recall@5"}, nil},
		{"conflicting scales", model.Record{"reported_value": 0.7, "raw_unit": "%", "reported_scale": "fraction"}, nil},
	} {
		t.Run(test.name, func(t *testing.T) {
			actual := model.Normalized(test.result)
			if actual == nil || test.expected == nil {
				if actual != test.expected {
					t.Fatalf("normalization expected: %+v, got: %+v", test.expected, actual)
				}
				return
			}
			if *actual != *test.expected {
				t.Fatalf("normalization expected: %+v, got: %+v", test.expected, actual)
			}
		})
	}
}

func TestComparison(t *testing.T) {
	importer, err := extract.NewImporter(sverka.ExtractorSchema)
	if err != nil {
		t.Fatal(err)
	}
	data, err := importer.Prepare(sverka.MockExtraction, model.Document{ID: "demo", ExtractionMode: "mock"})
	if err != nil {
		t.Fatal(err)
	}
	left, right := data.ResultDetail(data.Results[0]), data.ResultDetail(data.Results[1])
	match := model.Compare(left, right)
	if match.Status != model.Comparable || match.Numeric == nil ||
		math.Abs(match.Numeric.Delta-0.08) > 1e-9 || match.Numeric.Preferred != "right" {
		t.Fatalf("comparison expected: comparable with delta 0.08 favoring right, got: %+v", match)
	}

	left.Dataset, right.Dataset = maps.Clone(left.Dataset), maps.Clone(right.Dataset)
	delete(left.Dataset, "dataset_version")
	delete(right.Dataset, "dataset_version")
	missing := model.Compare(left, right)
	if missing.Status != model.InsufficientInformation || missing.Numeric != nil {
		t.Fatalf("comparison expected: insufficient_information with no delta, got: %+v", missing)
	}
	right.Result = maps.Clone(right.Result)
	right.Result["metric_k"] = 10.0
	right.Result["metric_name"] = "Recall@10"
	mismatch := model.Compare(left, right)
	if mismatch.Status != model.NotComparable || mismatch.Numeric != nil {
		t.Fatalf("comparison expected: not_comparable with no delta, got: %+v", mismatch)
	}
}

func TestCrossPublicationComparisonRequiresMeasurementContext(t *testing.T) {
	importer, err := extract.NewImporter(sverka.ExtractorSchema)
	if err != nil {
		t.Fatal(err)
	}
	data, err := importer.Prepare(sverka.MockExtraction, model.Document{ID: "demo", ExtractionMode: "mock"})
	if err != nil {
		t.Fatal(err)
	}
	left, right := data.ResultDetail(data.Results[0]), data.ResultDetail(data.Results[1])
	left.DocumentID, right.DocumentID = "paper-a", "paper-b"
	left.Publication = model.Record{"publication_id": "pub-a", "title": "Paper A"}
	right.Publication = model.Record{"publication_id": "pub-b", "title": "Paper B"}

	match := model.Compare(left, right)
	if match.Status != model.Comparable || match.Numeric == nil {
		t.Fatalf("cross-publication comparison with matching measurement context expected comparable, got: %+v", match)
	}

	right.Experiment = maps.Clone(right.Experiment)
	right.Experiment["measurement_conditions"] = "different laboratory temperature"
	mismatch := model.Compare(left, right)
	if mismatch.Status != model.NotComparable || mismatch.Numeric != nil {
		t.Fatalf("different measurement conditions expected not_comparable without delta, got: %+v", mismatch)
	}

	right.Experiment = maps.Clone(right.Experiment)
	delete(right.Experiment, "instrument")
	right.Experiment["measurement_conditions"] = left.Experiment.Text("measurement_conditions")
	missing := model.Compare(left, right)
	if missing.Status != model.InsufficientInformation || missing.Numeric != nil {
		t.Fatalf("missing instrument expected insufficient_information without delta, got: %+v", missing)
	}
}

func TestConfirmedInstrumentEquivalenceAllowsCrossPublicationComparison(t *testing.T) {
	importer, err := extract.NewImporter(sverka.ExtractorSchema)
	if err != nil {
		t.Fatal(err)
	}
	data, err := importer.Prepare(sverka.MockExtraction, model.Document{ID: "demo", ExtractionMode: "mock"})
	if err != nil {
		t.Fatal(err)
	}
	left, right := data.ResultDetail(data.Results[0]), data.ResultDetail(data.Results[1])
	left.DocumentID, right.DocumentID = "paper-a", "paper-b"
	left.Publication = model.Record{"publication_id": "pub-a", "title": "Paper A"}
	right.Publication = model.Record{"publication_id": "pub-b", "title": "Paper B"}
	left.Experiment, right.Experiment = maps.Clone(left.Experiment), maps.Clone(right.Experiment)
	left.Experiment["instrument_equivalence_id"] = "validated-harness-v1"
	right.Experiment["instrument_equivalence_id"] = "validated-harness-v1"
	right.Experiment["instrument"] = "Synthetic evaluation harness v2"

	match := model.Compare(left, right)
	if match.Status != model.Comparable || match.Numeric == nil {
		t.Fatalf("confirmed equivalent instruments expected comparable, got: %+v", match)
	}

	right.Experiment["instrument_equivalence_id"] = "different-family"
	right.Experiment["instrument"] = left.Experiment.Text("instrument")
	mismatch := model.Compare(left, right)
	if mismatch.Status != model.NotComparable || mismatch.Numeric != nil {
		t.Fatalf("different instrument equivalence IDs expected not_comparable, got: %+v", mismatch)
	}
}

func TestSamePublicationDoesNotRequireInterPublicationMeasurementContext(t *testing.T) {
	importer, err := extract.NewImporter(sverka.ExtractorSchema)
	if err != nil {
		t.Fatal(err)
	}
	data, err := importer.Prepare(sverka.MockExtraction, model.Document{ID: "demo", ExtractionMode: "mock"})
	if err != nil {
		t.Fatal(err)
	}
	left, right := data.ResultDetail(data.Results[0]), data.ResultDetail(data.Results[1])
	left.Experiment, right.Experiment = maps.Clone(left.Experiment), maps.Clone(right.Experiment)
	delete(left.Experiment, "instrument")
	delete(left.Experiment, "instrument_characteristics")
	delete(left.Experiment, "measurement_conditions")
	delete(right.Experiment, "instrument")
	delete(right.Experiment, "instrument_characteristics")
	delete(right.Experiment, "measurement_conditions")

	comparison := model.Compare(left, right)
	if comparison.Status != model.Comparable || comparison.Numeric == nil {
		t.Fatalf("same-publication comparison should not require inter-publication context, got: %+v", comparison)
	}
}

func TestComparableGroupsRequireDistinctPublicationsAndMatchingMeasurementContext(t *testing.T) {
	importer, err := extract.NewImporter(sverka.ExtractorSchema)
	if err != nil {
		t.Fatal(err)
	}
	data, err := importer.Prepare(sverka.MockExtraction, model.Document{ID: "demo", ExtractionMode: "mock"})
	if err != nil {
		t.Fatal(err)
	}
	left, right := data.ResultDetail(data.Results[0]), data.ResultDetail(data.Results[1])
	if groups := model.ComparableGroups([]model.ResultDetail{left, right}); len(groups) != 0 {
		t.Fatalf("same-publication rows expected no cross-publication groups, got: %+v", groups)
	}
	left.DocumentID, right.DocumentID = "paper-a", "paper-b"
	left.Publication = model.Record{"publication_id": "pub-a", "title": "Paper A"}
	right.Publication = model.Record{"publication_id": "pub-b", "title": "Paper B"}
	groups := model.ComparableGroups([]model.ResultDetail{left, right})
	if len(groups) != 1 || len(groups[0].Results) != 2 {
		t.Fatalf("matching cross-publication rows expected one two-result group, got: %+v", groups)
	}
	left.Experiment, right.Experiment = maps.Clone(left.Experiment), maps.Clone(right.Experiment)
	left.Experiment["instrument_equivalence_id"] = "validated-harness-v1"
	right.Experiment["instrument_equivalence_id"] = "validated-harness-v1"
	right.Experiment["instrument"] = "Synthetic evaluation harness v2"
	if groups := model.ComparableGroups([]model.ResultDetail{left, right}); len(groups) != 1 {
		t.Fatalf("confirmed equivalent instruments expected one group, got: %+v", groups)
	}
	right.Experiment = maps.Clone(right.Experiment)
	right.Experiment["instrument_characteristics"] = "different apparatus"
	if groups := model.ComparableGroups([]model.ResultDetail{left, right}); len(groups) != 0 {
		t.Fatalf("mismatched apparatus expected no cross-publication group, got: %+v", groups)
	}
}
