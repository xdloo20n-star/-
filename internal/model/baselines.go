package model

import (
	"strings"
)

const baselineUnavailable = "baseline_unavailable"

// RefreshBaselines recalculates baseline values and warnings for a document with scoped IDs.
func (data *DocumentData) RefreshBaselines() {
	var linked []*ResultSummary
	for i := range data.Results {
		if data.Results[i].Result.Text("baseline_result_id") != "" {
			linked = append(linked, &data.Results[i])
		}
	}
	if len(linked) == 0 {
		return
	}

	results := make(map[string]*ResultSummary, len(data.Results))
	for i := range data.Results {
		row := &data.Results[i]
		results[row.Result.Text("result_id")] = row
	}
	sources := resultEvidenceIDs(data.Evidence)
	datasets := recordsByID(data.Datasets, "dataset_id")
	tasks := recordsByID(data.Tasks, "task_id")
	experiments := recordsByID(data.Experiments, "experiment_id")
	detail := func(row *ResultSummary) ResultDetail {
		return ResultDetail{
			ResultSummary: *row,
			Dataset:       datasets[row.Result.Text("dataset_id")],
			Task:          tasks[row.Result.Text("task_id")],
			Experiment:    experiments[row.Result.Text("experiment_id")],
		}
	}

	// Preserve issue IDs from earlier imports while replacing their contents.
	previousIDs := make(map[string]string)
	issues := data.Issues[:0]
	for _, issue := range data.Issues {
		if issue.CheckName == baselineUnavailable && len(issue.ResultIDs) > 0 {
			previousIDs[issue.ResultIDs[0]] = issue.ID
			continue
		}
		issues = append(issues, issue)
	}
	for _, row := range linked {
		id, baselineID := row.Result.Text("result_id"), row.Result.Text("baseline_result_id")
		row.Baseline = nil
		baseline := results[baselineID]
		if baseline == nil || baseline == row {
			continue // The importer validates result references before this step.
		}
		criteria, status := compareConditions(detail(row), detail(baseline))
		if status == Comparable && row.Normalized != nil && baseline.Normalized != nil {
			row.Baseline = baseline.Normalized
			continue
		}
		evidenceIDs := make([]string, 0, len(sources[id])+len(sources[baselineID]))
		evidenceIDs = append(evidenceIDs, sources[id]...)
		evidenceIDs = append(evidenceIDs, sources[baselineID]...)
		issueID := previousIDs[id]
		if issueID == "" {
			issueID = resultIssueID(id, baselineUnavailable)
		}
		issues = append(issues, Issue{
			ID: issueID, CheckName: baselineUnavailable, Severity: "warning",
			Description: baselineUnavailableMessage(row.Result, baseline.Result, criteria),
			ResultIDs:   []string{id, baselineID}, EvidenceIDs: evidenceIDs,
		})
	}
	data.Issues = issues
	data.countResultIssues()
}

func baselineUnavailableMessage(result, baseline Record, criteria []Criterion) string {
	const prefix = "Baseline не включён в baseline_value: "
	var reasons []string
	for _, criterion := range criteria {
		if criterion.Status == "mismatch" || criterion.Status == "missing" {
			reasons = append(reasons, criterion.Message)
		}
	}
	if len(reasons) != 0 {
		return prefix + strings.Join(reasons, " ")
	}
	if result.Text("extraction_status") != "extracted" || baseline.Text("extraction_status") != "extracted" {
		return prefix + "один из результатов не имеет статуса extracted."
	}
	return prefix + "для одного или обоих результатов не подтверждена числовая нормализация."
}
