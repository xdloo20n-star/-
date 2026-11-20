package model_test

import (
	"maps"
	"math"
	"testing"

	"papertrace"
	"papertrace/internal/extract"
	"papertrace/internal/model"
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
	importer, err := extract.NewImporter(papertrace.ExtractorSchema)
	if err != nil {
		t.Fatal(err)
	}
	data, err := importer.Prepare(papertrace.MockExtraction, model.Document{ID: "demo", ExtractionMode: "mock"})
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
