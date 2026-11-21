package model_test

import (
	"encoding/json"
	"slices"
	"testing"

	"sverka"
	"sverka/internal/extract"
	"sverka/internal/model"
)

func importBaselineFixture(t *testing.T, edit func(model.Record)) *model.DocumentData {
	t.Helper()
	var fixture model.Record
	if err := json.Unmarshal(sverka.MockExtraction, &fixture); err != nil {
		t.Fatal(err)
	}
	edit(fixture)
	payload, err := json.Marshal(fixture)
	if err != nil {
		t.Fatal(err)
	}
	importer, err := extract.NewImporter(sverka.ExtractorSchema)
	if err != nil {
		t.Fatal(err)
	}
	data, err := importer.Prepare(payload, model.Document{ID: "baseline-test", ExtractionMode: "mock"})
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestBaselineIssueUsesAvailableEvidenceFromBothResults(t *testing.T) {
	for _, missingPrimary := range []bool{false, true} {
		name := "both results have evidence"
		if missingPrimary {
			name = "only baseline has evidence"
		}
		t.Run(name, func(t *testing.T) {
			data := importBaselineFixture(t, func(fixture model.Record) {
				rows := fixture["results"].([]any)
				rows[0].(map[string]any)["aggregation"] = "median"
				if missingPrimary {
					primary := rows[1].(map[string]any)
					primary["extraction_status"] = "present_not_extractable"
					primary["reported_value"] = nil
					fixture["evidence"] = slices.DeleteFunc(fixture["evidence"].([]any), func(item any) bool {
						return item.(map[string]any)["entity_id"] == primary["result_id"]
					})
				}
			})
			baseline, primary := data.Results[0], data.Results[1]
			if primary.Baseline != nil || len(data.Issues) != 1 {
				t.Fatalf("baseline and issues expected: nil and one warning, got: %v and %+v", primary.Baseline, data.Issues)
			}
			issue := data.Issues[0]
			wantIDs := []string{primary.Result.Text("result_id"), baseline.Result.Text("result_id")}
			if issue.CheckName != "baseline_unavailable" || !slices.Equal(issue.ResultIDs, wantIDs) {
				t.Fatalf("baseline warning expected with results %v, got: %+v", wantIDs, issue)
			}
			var expectedEvidence []string
			for _, id := range wantIDs {
				for _, item := range data.Evidence {
					if item.Text("entity_type") == "result" && item.Text("entity_id") == id {
						expectedEvidence = append(expectedEvidence, item.Text("evidence_id"))
					}
				}
			}
			if len(expectedEvidence) == 0 || !slices.Equal(issue.EvidenceIDs, expectedEvidence) {
				t.Fatalf("evidence IDs expected: %v, got: %v", expectedEvidence, issue.EvidenceIDs)
			}
			data.RefreshBaselines()
			if len(data.Issues) != 1 || data.Issues[0].ID != issue.ID || data.Results[0].IssueCount != 1 || data.Results[1].IssueCount != 1 {
				t.Fatalf("recalculation expected: unchanged warning with one issue per result, got: %+v", data)
			}
		})
	}
}

func TestBaselineValueRemainsWhenOnlyDeltaOverflows(t *testing.T) {
	data := importBaselineFixture(t, func(fixture model.Record) {
		for i, item := range fixture["results"].([]any) {
			row := item.(map[string]any)
			row["raw_unit"], row["reported_scale"] = "count", nil
			row["raw_value"], row["reported_value"] = "1e308", 1e308
			if i == 0 {
				row["raw_value"], row["reported_value"] = "-1e308", -1e308
			}
		}
	})
	baseline, primary := data.Results[0], data.Results[1]
	if primary.Baseline == nil || primary.Baseline.Value != -1e308 || len(data.Issues) != 0 {
		t.Fatalf("baseline and issues expected: -1e308 and no warning, got: %+v and %+v", primary.Baseline, data.Issues)
	}
	comparison := model.Compare(data.ResultDetail(primary), data.ResultDetail(baseline))
	if comparison.Status != model.Comparable || comparison.Numeric != nil {
		t.Fatalf("comparison expected: comparable with no numeric delta, got: %+v", comparison)
	}
}
