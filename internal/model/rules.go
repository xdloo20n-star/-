package model

import (
	"math"
	"regexp"
	"strings"
)

var integerCutoffSuffix = regexp.MustCompile(`@\d+$`)

// Normalized derives a value only from explicit units or scale information.
func Normalized(result Record) *MeasuredValue {
	value, ok := result.Number("reported_value")
	unit, factor := normalizationUnit(result)
	if !ok || unit == "" {
		return nil
	}
	if factor < 1 {
		value /= 1 / factor
	} else {
		value *= factor
	}
	if math.IsInf(value, 0) {
		return nil
	}
	return &MeasuredValue{Value: value, Unit: unit}
}

func normalizationUnit(result Record) (unit string, factor float64) {
	rawUnit := key(result.Text("raw_unit"))
	scale := result.Text("reported_scale")
	switch {
	case rawUnit == "%" || rawUnit == "percent":
		if scale != "fraction" {
			return "fraction", 0.01
		}
	case rawUnit == "fraction":
		if scale != "percent" {
			return "fraction", 1
		}
	case rawUnit == "":
		switch scale {
		case "percent":
			return "fraction", 0.01
		case "fraction":
			return "fraction", 1
		}
	case scale == "":
		switch rawUnit {
		case "ms", "millisecond", "milliseconds":
			return "ms", 1
		case "s", "sec", "second", "seconds":
			return "ms", 1_000
		case "min", "minute", "minutes":
			return "ms", 60_000
		case "h", "hour", "hours":
			return "ms", 3_600_000
		case "µs", "μs", "us", "microsecond", "microseconds":
			return "ms", 0.001
		case "ns", "nanosecond", "nanoseconds":
			return "ms", 0.000001
		case "tokens/query", "count":
			return rawUnit, 1
		}
	}
	return "", 0
}

// BoundedFraction identifies definitions whose range is known from the reported variant.
func BoundedFraction(result Record) bool {
	switch baseMetric(result) {
	case "recall", "precision":
		switch key(result.Text("metric_variant")) {
		case "document-level", "token-level", "word-level":
			return true
		}
	case "f1":
		switch key(result.Text("metric_variant")) {
		case "token-level", "word-level", "macro", "micro":
			return true
		}
	}
	return false
}

const (
	Comparable              = "comparable"
	NotComparable           = "not_comparable"
	InsufficientInformation = "insufficient_information"
)

type Comparison struct {
	LeftResultID        string             `json:"left_result_id"`
	RightResultID       string             `json:"right_result_id"`
	LeftExtractionMode  string             `json:"left_extraction_mode"`
	RightExtractionMode string             `json:"right_extraction_mode"`
	Status              string             `json:"status"`
	Criteria            []Criterion        `json:"criteria"`
	Numeric             *NumericComparison `json:"numeric"`
}

type Criterion struct {
	Name    string `json:"criterion"`
	Status  string `json:"status"`
	Message string `json:"message"`
}

type NumericComparison struct {
	LeftValue  float64 `json:"left_value"`
	RightValue float64 `json:"right_value"`
	Delta      float64 `json:"delta"`
	Unit       string  `json:"unit"`
	Preferred  string  `json:"preferred"`
}

func Compare(left, right ResultDetail) Comparison {
	leftUnit, _ := normalizationUnit(left.Result)
	rightUnit, _ := normalizationUnit(right.Result)
	comparison := Comparison{
		LeftResultID:        left.Result.Text("result_id"),
		RightResultID:       right.Result.Text("result_id"),
		LeftExtractionMode:  left.ExtractionMode,
		RightExtractionMode: right.ExtractionMode,
		Status:              Comparable,
		Criteria: []Criterion{
			compareText("task", "Тип задачи", left.Task.Text("task_type"), right.Task.Text("task_type")),
			compareText("dataset", "Датасет", left.Dataset.Text("dataset_name"), right.Dataset.Text("dataset_name")),
			compareText("dataset_version", "Версия датасета", left.Dataset.Text("dataset_version"), right.Dataset.Text("dataset_version")),
			compareText("data_split", "Сплит", left.Dataset.Text("data_split"), right.Dataset.Text("data_split")),
			compareDefinition(left.Result, right.Result),
			compareCutoff(left.Result, right.Result),
			compareText("scale", "Шкала после нормализации", leftUnit, rightUnit),
			compareText("aggregation", "Агрегация", left.Result.Text("aggregation"), right.Result.Text("aggregation")),
			compareText("evaluation_protocol", "Протокол оценки", left.Experiment.Text("evaluation_protocol"), right.Experiment.Text("evaluation_protocol")),
		},
	}
	for _, criterion := range comparison.Criteria {
		switch {
		case criterion.Status == "mismatch":
			comparison.Status = NotComparable
		case criterion.Status == "missing" && comparison.Status == Comparable:
			comparison.Status = InsufficientInformation
		}
	}
	if comparison.Status != Comparable || left.Normalized == nil || right.Normalized == nil {
		return comparison
	}

	delta := right.Normalized.Value - left.Normalized.Value
	if math.IsInf(delta, 0) {
		return comparison
	}
	comparison.Numeric = &NumericComparison{
		LeftValue: left.Normalized.Value, RightValue: right.Normalized.Value, Delta: delta,
		Unit: left.Normalized.Unit, Preferred: preferredSide(delta, left.Result, right.Result),
	}
	return comparison
}

func differentPublications(left, right ResultDetail) bool {
	leftDOI, rightDOI := key(left.Publication.Text("doi")), key(right.Publication.Text("doi"))
	if leftDOI != "" && rightDOI != "" {
		return normalizeDOI(leftDOI) != normalizeDOI(rightDOI)
	}
	leftTitle, rightTitle := key(left.Publication.Text("title")), key(right.Publication.Text("title"))
	if leftTitle != "" && rightTitle != "" {
		return leftTitle != rightTitle
	}
	leftID, rightID := left.Publication.Text("publication_id"), right.Publication.Text("publication_id")
	if leftID != "" && rightID != "" {
		return leftID != rightID
	}
	return left.DocumentID != right.DocumentID
}

func normalizeDOI(value string) string {
	for _, prefix := range []string{"https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi.org/", "doi:"} {
		value = strings.TrimPrefix(value, prefix)
	}
	return strings.TrimRight(strings.TrimSpace(value), ".,; ")
}

func compareText(name, label, left, right string) Criterion {
	switch {
	case key(left) == "" || key(right) == "":
		return Criterion{name, "missing", label + " не указан для одной или обеих записей."}
	case key(left) != key(right):
		return Criterion{name, "mismatch", label + " различается: «" + left + "» и «" + right + "»."}
	default:
		return Criterion{name, "match", label + " совпадает: «" + left + "»."}
	}
}

func compareDefinition(left, right Record) Criterion {
	definition := compareText("metric_definition", "Название метрики", baseMetric(left), baseMetric(right))
	if definition.Status != "match" {
		return definition
	}
	return compareText("metric_definition", "Вариант метрики", left.Text("metric_variant"), right.Text("metric_variant"))
}

func compareCutoff(left, right Record) Criterion {
	leftApplicability := left.Text("metric_k_applicability")
	rightApplicability := right.Text("metric_k_applicability")
	leftK, leftKnown := left.Number("metric_k")
	rightK, rightKnown := right.Number("metric_k")
	switch {
	case leftApplicability == "not_applicable" && rightApplicability == "not_applicable":
		return Criterion{"metric_k", "not_applicable", "У обеих метрик нет параметра k."}
	case leftApplicability == "unknown" || rightApplicability == "unknown":
		return Criterion{"metric_k", "missing", "Неизвестно, применим ли параметр k."}
	case leftApplicability != rightApplicability:
		return Criterion{"metric_k", "mismatch", "Параметр k применим только к одной метрике."}
	case !leftKnown || !rightKnown:
		return Criterion{"metric_k", "missing", "Не указан k для одной или обеих метрик."}
	case leftK != rightK:
		return Criterion{"metric_k", "mismatch", "Значения k различаются."}
	default:
		return Criterion{"metric_k", "match", "Значения k совпадают."}
	}
}

func preferredSide(delta float64, left, right Record) string {
	if delta == 0 {
		return "tie"
	}
	direction := left.Text("metric_direction")
	if direction != right.Text("metric_direction") {
		return "unknown"
	}
	switch direction {
	case "higher_is_better", "lower_is_better":
		if (delta > 0) == (direction == "higher_is_better") {
			return "right"
		}
		return "left"
	default:
		return "unknown"
	}
}

func baseMetric(result Record) string {
	name := key(result.Text("metric_name"))
	if result.Text("metric_k_applicability") == "applicable" {
		return integerCutoffSuffix.ReplaceAllString(name, "")
	}
	return name
}

func key(value string) string {
	return strings.ToLower(strings.Join(strings.Fields(value), " "))
}
