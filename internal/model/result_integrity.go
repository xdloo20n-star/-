package model

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"math/big"
	"regexp"
	"strconv"
	"strings"
)

const (
	checkRawValueConflict = "raw_value_conflict"
	checkCutoffConflict   = "metric_k_conflict"
	checkDuplicate        = "duplicate_invalid"
)

var decimalValue = regexp.MustCompile(`^[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$`)

// literalNumber recognizes unambiguous scalar values. Ranges, intervals and
// locale-dependent number formats are left for the extractor to interpret.
func literalNumber(raw string) (float64, bool) {
	text := strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(raw), "%"))
	text = strings.ReplaceAll(text, "−", "-")
	if !decimalValue.MatchString(text) {
		return 0, false
	}
	value, err := strconv.ParseFloat(text, 64)
	return value, err == nil && !math.IsInf(value, 0)
}

func RawValueConflict(result Record) bool {
	literal, known := literalNumber(result.Text("raw_value"))
	reported, parsed := result.Number("reported_value")
	return known && parsed && literal != reported
}

func CutoffConflict(result Record) bool {
	name := result.Text("metric_name")
	match := integerCutoffSuffix.FindStringSubmatch(name)
	if match == nil {
		return false
	}
	if result.Text("metric_k_applicability") == "not_applicable" {
		return true
	}
	cutoff, known := result.Number("metric_k")
	if !known {
		return false
	}
	inline, err := strconv.ParseFloat(match[1], 64)
	return err != nil || inline != cutoff
}

// DuplicateConflict returns the first field that prevents treating two rows as
// repeated records of the same result.
func DuplicateConflict(result, target Record) string {
	if result.Text("extraction_status") != "extracted" || target.Text("extraction_status") != "extracted" {
		return "extraction_status"
	}
	for _, field := range []string{"experiment_id", "config_id", "dataset_id", "task_id", "metric_name",
		"metric_variant", "metric_k_applicability", "aggregation", "result_origin", "cited_publication_id"} {
		if result.Text(field) != target.Text(field) {
			return field
		}
	}
	for _, field := range []string{"metric_k", "sample_size"} {
		left, leftKnown := result.Number(field)
		right, rightKnown := target.Number(field)
		if leftKnown != rightKnown || left != right {
			return field
		}
	}
	if RawValueConflict(result) || RawValueConflict(target) || CutoffConflict(result) || CutoffConflict(target) {
		return "reported_value or metric_k"
	}
	left, right := Normalized(result), Normalized(target)
	if left != nil && right != nil {
		if left.Unit != right.Unit {
			return "metric_unit"
		}
		if !sameAtReportedPrecision(left.Value, right.Value) {
			return "metric_value"
		}
		return ""
	}
	if (left == nil) != (right == nil) || result.Text("raw_unit") != target.Text("raw_unit") ||
		result.Text("reported_scale") != target.Text("reported_scale") {
		return "metric_unit"
	}
	if strings.TrimSpace(result.Text("raw_value")) == strings.TrimSpace(target.Text("raw_value")) {
		return ""
	}
	literal, known := literalNumber(result.Text("raw_value"))
	other, otherKnown := literalNumber(target.Text("raw_value"))
	if known && otherKnown && strings.HasSuffix(strings.TrimSpace(result.Text("raw_value")), "%") ==
		strings.HasSuffix(strings.TrimSpace(target.Text("raw_value")), "%") && sameAtReportedPrecision(literal, other) {
		return ""
	}
	return "raw_value"
}

// Use decimal half-up rounding at the coarser of the two printed precisions.
func sameAtReportedPrecision(a, b float64) bool {
	if a == b {
		return true
	}
	left, right := strconv.FormatFloat(a, 'f', -1, 64), strconv.FormatFloat(b, 'f', -1, 64)
	_, fracLeft, _ := strings.Cut(left, ".")
	_, fracRight, _ := strings.Cut(right, ".")
	if len(fracLeft) == len(fracRight) {
		return false
	}
	if len(fracLeft) > len(fracRight) {
		left, right = right, left
		fracLeft = fracRight
	}
	coarse, _ := new(big.Rat).SetString(left)
	fine, _ := new(big.Rat).SetString(right)
	scale := new(big.Rat).SetInt(new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(len(fracLeft))), nil))
	fine.Mul(fine, scale)
	coarse.Mul(coarse, scale)
	rounded, remainder := new(big.Int), new(big.Int)
	rounded.QuoRem(fine.Num(), fine.Denom(), remainder)
	if remainder.Abs(remainder).Lsh(remainder, 1).Cmp(fine.Denom()) >= 0 {
		rounded.Add(rounded, big.NewInt(int64(fine.Sign())))
	}
	return rounded.Cmp(coarse.Num()) == 0
}

func resultIssueID(id, rule string) string {
	sum := sha256.Sum256([]byte(id + "\x00" + rule))
	return "issue_" + hex.EncodeToString(sum[:16])
}

func resultEvidenceIDs(evidence []Record) map[string][]string {
	sources := make(map[string][]string)
	for _, item := range evidence {
		if item.Text("entity_type") == "result" {
			id := item.Text("entity_id")
			sources[id] = append(sources[id], item.Text("evidence_id"))
		}
	}
	return sources
}

// RefreshResultIntegrity records new integrity checks and repairs stale derived
// values and invalid duplicate links in snapshots created by earlier versions.
func (data *DocumentData) RefreshResultIntegrity() bool {
	results := make(map[string]Record, len(data.Results))
	for _, row := range data.Results {
		results[row.Result.Text("result_id")] = row.Result
	}
	issues := make(map[string]bool, len(data.Issues))
	for _, issue := range data.Issues {
		issues[issue.ID] = true
	}
	changed := false
	var sources map[string][]string
	for i := range data.Results {
		row := &data.Results[i]
		id := row.Result.Text("result_id")
		add := func(rule, description string, relatedIDs ...string) {
			issueID := resultIssueID(id, rule)
			if issues[issueID] {
				return
			}
			resultIDs := append([]string{id}, relatedIDs...)
			if sources == nil {
				sources = resultEvidenceIDs(data.Evidence)
			}
			evidenceIDs := append([]string{}, sources[id]...)
			for _, related := range relatedIDs {
				evidenceIDs = append(evidenceIDs, sources[related]...)
			}
			data.Issues = append(data.Issues, Issue{
				ID: issueID, CheckName: rule, Severity: "warning", Description: description,
				ResultIDs: resultIDs, EvidenceIDs: evidenceIDs,
			})
			changed = true
		}
		if RawValueConflict(row.Result) {
			if row.Normalized != nil {
				row.Normalized = nil
				changed = true
			}
			add(checkRawValueConflict, "Числовой разбор отличается от записи в PDF. Нормализация недоступна.")
		}
		if CutoffConflict(row.Result) {
			add(checkCutoffConflict, "Значение k в названии метрики противоречит отдельному полю.")
		}
		if related := row.Result.Text("duplicate_of"); related != "" && results[related] != nil {
			if field := DuplicateConflict(row.Result, results[related]); field != "" {
				delete(row.Result, "duplicate_of")
				changed = true
				add(checkDuplicate, "Запись сохранена отдельно: ссылка duplicate_of не согласуется с полем "+field+".", related)
			}
		}
	}
	if changed {
		data.countResultIssues()
	}
	return changed
}

func (data *DocumentData) countResultIssues() {
	counts := make(map[string]int, len(data.Results))
	for _, issue := range data.Issues {
		for _, id := range issue.ResultIDs {
			counts[id]++
		}
	}
	for i := range data.Results {
		row := &data.Results[i]
		row.IssueCount = counts[row.Result.Text("result_id")]
	}
}
