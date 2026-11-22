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

func TestComparisonRequiresNumericValuesAndMatchingModes(t *testing.T) {
	importer, err := extract.NewImporter(sverka.ExtractorSchema)
	if err != nil {
		t.Fatal(err)
	}
	data, err := importer.Prepare(sverka.MockExtraction, model.Document{ID: "comparison", ExtractionMode: "mock"})
	if err != nil {
		t.Fatal(err)
	}
	left, right := data.ResultDetail(data.Results[0]), data.ResultDetail(data.Results[1])
	left.Normalized = nil
	if comparison := model.Compare(left, right); comparison.Status != model.InsufficientInformation || comparison.Numeric != nil {
		t.Fatalf("missing numeric value: %+v", comparison)
	}
	left = data.ResultDetail(data.Results[0])
	left.Result = maps.Clone(left.Result)
	left.Result["extraction_status"] = "absent"
	if comparison := model.Compare(left, right); comparison.Status != model.InsufficientInformation || comparison.Numeric != nil {
		t.Fatalf("absent value: %+v", comparison)
	}
	left = data.ResultDetail(data.Results[0])
	right.ExtractionMode = "python"
	if comparison := model.Compare(left, right); comparison.Status != model.NotComparable || comparison.Numeric != nil {
		t.Fatalf("mixed extraction modes: %+v", comparison)
	}
}

func TestCrossPublicationComparisonUsesRAGCriteria(t *testing.T) {
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
		t.Fatalf("RAG results matching task, dataset, metric, scale, and protocol expected comparable, got: %+v", match)
	}
	right.Experiment = maps.Clone(right.Experiment)
	right.Experiment["evaluation_protocol"] = "a different benchmark protocol"
	mismatch := model.Compare(left, right)
	if mismatch.Status != model.NotComparable || mismatch.Numeric != nil {
		t.Fatalf("different RAG evaluation protocols expected not_comparable without delta, got: %+v", mismatch)
	}
}

func groupsForResults(details []model.ResultDetail) []model.ComparableGroup {
	var candidates model.GroupCandidates
	for _, detail := range details {
		candidates.AddDocument(detail.ExtractionMode, model.DocumentData{
			Publication: detail.Publication,
			Results:     []model.ResultSummary{detail.ResultSummary},
			Datasets:    []model.Record{detail.Dataset},
			Tasks:       []model.Record{detail.Task},
			Experiments: []model.Record{detail.Experiment},
		})
	}
	return candidates.Groups()
}

func TestComparableGroupsRequireConfirmedPublicationPair(t *testing.T) {
	importer, err := extract.NewImporter(sverka.ExtractorSchema)
	if err != nil {
		t.Fatal(err)
	}
	data, err := importer.Prepare(sverka.MockExtraction, model.Document{ID: "demo", ExtractionMode: "mock"})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, leftDOI, rightDOI, rightTitle, rightOrigin, rightMode string
		want                                                        int
	}{
		{name: "same title with incomplete DOI", leftDOI: "10.1234/a", rightTitle: " paper  A "},
		{name: "same DOI with different title", leftDOI: "doi:10.1234/A", rightDOI: "https://doi.org/10.1234/a", rightTitle: "Other title"},
		{name: "different DOI with same title", leftDOI: "10.1234/a", rightDOI: "10.1234/b", rightTitle: "Paper A", want: 1},
		{name: "different title with incomplete DOI", leftDOI: "10.1234/a", rightTitle: "Paper B", want: 1},
		{name: "missing publication metadata", leftDOI: "10.1234/a"},
		{name: "cited result", rightTitle: "Paper B", rightOrigin: "cited"},
		{name: "different extraction mode", rightTitle: "Paper B", rightMode: "python"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			left, right := data.ResultDetail(data.Results[0]), data.ResultDetail(data.Results[1])
			left.Publication = model.Record{"publication_id": "pub-a", "title": "Paper A", "doi": tc.leftDOI}
			right.Publication = model.Record{"publication_id": "pub-b", "title": tc.rightTitle, "doi": tc.rightDOI}
			if tc.rightOrigin != "" {
				right.Result = maps.Clone(right.Result)
				right.Result["result_origin"] = tc.rightOrigin
			}
			if tc.rightMode != "" {
				right.ExtractionMode = tc.rightMode
			}
			groups := groupsForResults([]model.ResultDetail{left, right})
			if len(groups) != tc.want {
				t.Fatalf("group count expected: %d, got: %d", tc.want, len(groups))
			}
		})
	}
}

func TestComparableGroupsResolvePublicationTitleAliases(t *testing.T) {
	importer, err := extract.NewImporter(sverka.ExtractorSchema)
	if err != nil {
		t.Fatal(err)
	}
	data, err := importer.Prepare(sverka.MockExtraction, model.Document{ID: "demo", ExtractionMode: "mock"})
	if err != nil {
		t.Fatal(err)
	}
	base := data.ResultDetail(data.Results[0])
	type paper struct {
		id, title, doi string
		otherProtocol  bool
		python         bool
	}
	for _, tc := range []struct {
		name        string
		papers      []paper
		wantPair    [2]string
		wantMembers int
	}{
		{
			name: "three records for the same DOI with different titles",
			papers: []paper{
				{id: "a", title: "Paper A", doi: "10.1234/x"},
				{id: "b", title: "Paper B", doi: "https://doi.org/10.1234/X"},
				{id: "c", title: " paper  A "},
			},
		},
		{
			name: "same DOI resolved by a candidate with another protocol",
			papers: []paper{
				{id: "a", title: "Paper A", doi: "10.1234/x", otherProtocol: true},
				{id: "b", title: "Paper B", doi: "10.1234/x"},
				{id: "c", title: "Paper A"},
			},
		},
		{
			name: "unique alias establishes a distinct DOI across protocols",
			papers: []paper{
				{id: "a", title: "Paper A", doi: "10.1234/x", otherProtocol: true},
				{id: "b", doi: "10.1234/y"},
				{id: "c", title: "Paper A"},
			},
			wantPair: [2]string{"b", "c"}, wantMembers: 2,
		},
		{
			name: "ambiguous title across protocols cannot establish a pair",
			papers: []paper{
				{id: "a", title: "Paper A", doi: "10.1234/x", otherProtocol: true},
				{id: "b", title: "Paper A", doi: "10.1234/y", otherProtocol: true},
				{id: "c", title: "Paper A"},
				{id: "d", title: "Paper B", doi: "10.1234/z"},
			},
		},
		{
			name: "distinct explicit DOIs confirm a group with an ambiguous member",
			papers: []paper{
				{id: "a", title: "Paper A"},
				{id: "b", title: "Paper A", doi: "10.1234/x"},
				{id: "c", title: "Paper A", doi: "10.1234/y"},
			},
			wantPair: [2]string{"b", "c"}, wantMembers: 3,
		},
		{
			name: "mock DOI does not resolve a title in Python results",
			papers: []paper{
				{id: "a", title: "Paper A", doi: "10.1234/x"},
				{id: "b", doi: "10.1234/y", python: true},
				{id: "c", title: "Paper A", python: true},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			details := make([]model.ResultDetail, 0, len(tc.papers))
			for _, paper := range tc.papers {
				detail := base
				detail.DocumentID = paper.id
				detail.Publication = model.Record{"publication_id": paper.id, "title": paper.title, "doi": paper.doi}
				detail.Result = maps.Clone(base.Result)
				detail.Result["result_id"] = paper.id
				if paper.python {
					detail.ExtractionMode = "python"
				}
				if paper.otherProtocol {
					detail.Experiment = maps.Clone(base.Experiment)
					detail.Experiment["evaluation_protocol"] = "another protocol for " + paper.id
				}
				details = append(details, detail)
			}
			groups := groupsForResults(details)
			if tc.wantMembers == 0 {
				if len(groups) != 0 {
					t.Fatalf("expected no cross-publication groups, got: %+v", groups)
				}
				return
			}
			if len(groups) != 1 || len(groups[0].Results) != tc.wantMembers {
				t.Fatalf("expected one group with %d results, got: %+v", tc.wantMembers, groups)
			}
			results := groups[0].Results
			pair := [2]string{results[0].ResultID, results[1].ResultID}
			if pair != tc.wantPair {
				t.Fatalf("expected first pair %v, got %v", tc.wantPair, pair)
			}
		})
	}
}

func TestComparableGroupsUseRAGCriteriaAndDistinctPublications(t *testing.T) {
	importer, err := extract.NewImporter(sverka.ExtractorSchema)
	if err != nil {
		t.Fatal(err)
	}
	data, err := importer.Prepare(sverka.MockExtraction, model.Document{ID: "demo", ExtractionMode: "mock"})
	if err != nil {
		t.Fatal(err)
	}
	left, right := data.ResultDetail(data.Results[0]), data.ResultDetail(data.Results[1])
	if groups := groupsForResults([]model.ResultDetail{left, right}); len(groups) != 0 {
		t.Fatalf("same-publication rows expected no cross-publication groups, got: %+v", groups)
	}
	left.DocumentID, right.DocumentID = "paper-a", "paper-b"
	left.Publication = model.Record{"publication_id": "pub-a", "title": "Paper A"}
	right.Publication = model.Record{"publication_id": "pub-b", "title": "Paper B"}
	left.ExtractionMode, right.ExtractionMode = "python", "python"
	groups := groupsForResults([]model.ResultDetail{left, right})
	if len(groups) != 1 || len(groups[0].Results) != 2 {
		t.Fatalf("RAG results matching task, dataset, metric, and protocol expected one two-result group, got: %+v", groups)
	}
	if groups[0].ExtractionMode != "python" {
		t.Fatalf("expected group extraction mode python, got %q", groups[0].ExtractionMode)
	}
	right.Experiment = maps.Clone(right.Experiment)
	right.Experiment["evaluation_protocol"] = "a different protocol"
	if groups := groupsForResults([]model.ResultDetail{left, right}); len(groups) != 0 {
		t.Fatalf("different RAG protocols expected no group, got: %+v", groups)
	}
}
