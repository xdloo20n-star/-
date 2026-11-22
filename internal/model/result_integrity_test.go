package model_test

import (
	"testing"

	"sverka/internal/model"
)

func TestImportKeepsInvalidDuplicatesAsIndependentResults(t *testing.T) {
	data := importBaselineFixture(t, func(fixture model.Record) {
		rows := fixture["results"].([]any)
		left, right := rows[0].(map[string]any), rows[1].(map[string]any)
		right["duplicate_of"] = left["result_id"]
	})
	left, right := data.Results[0], data.Results[1]
	if right.Result.Text("duplicate_of") != "" || len(data.Issues) != 1 || data.Issues[0].CheckName != "duplicate_invalid" {
		t.Fatalf("expected two independent results with a warning, got %+v", data)
	}
	if left.IssueCount != 1 || right.IssueCount != 1 || len(data.Issues[0].EvidenceIDs) != 2 {
		t.Fatalf("expected one warning shared by both rows and both sources, got %+v", data.Issues)
	}
}

func TestImportPreservesAValidRoundedDuplicate(t *testing.T) {
	data := importBaselineFixture(t, func(fixture model.Record) {
		rows := fixture["results"].([]any)
		left, right := rows[0].(map[string]any), rows[1].(map[string]any)
		right["config_id"], right["duplicate_of"] = left["config_id"], left["result_id"]
		right["raw_value"], right["reported_value"] = "52.1", 52.1
	})
	left, right := data.Results[0], data.Results[1]
	if right.Result.Text("duplicate_of") != left.Result.Text("result_id") || len(data.Issues) != 0 {
		t.Fatalf("expected a duplicate consistent at the reported precision, got %+v", data)
	}
}

func TestImportFlagsContradictoryCutoffAndNumericValue(t *testing.T) {
	data := importBaselineFixture(t, func(fixture model.Record) {
		rows := fixture["results"].([]any)
		left, right := rows[0].(map[string]any), rows[1].(map[string]any)
		left["metric_name"], left["metric_k"] = "Recall @ 5", 10
		right["metric_name"], right["metric_k"] = "Recall@10", 10
		right["reported_value"] = 0.6
	})
	left, right := data.Results[0], data.Results[1]
	if right.Normalized != nil || right.Baseline != nil || right.Result.Text("raw_value") != "60" {
		t.Fatalf("expected original raw value without derived normalization or baseline, got %+v", right)
	}
	comparison := model.Compare(data.ResultDetail(left), data.ResultDetail(right))
	if comparison.Status != model.NotComparable || comparison.Numeric != nil {
		t.Fatalf("expected noncomparable results with no delta, got %+v", comparison)
	}
	warnings := map[string]bool{}
	for _, issue := range data.Issues {
		warnings[issue.CheckName] = true
	}
	for _, rule := range []string{"raw_value_conflict", "metric_k_conflict", "baseline_unavailable"} {
		if !warnings[rule] {
			t.Errorf("expected warning %q, got %+v", rule, data.Issues)
		}
	}
}
