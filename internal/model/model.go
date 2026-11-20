package model

import (
	"encoding/json"
	"math"
)

const SchemaVersion = "1.0"

const (
	Queued     = "queued"
	Processing = "processing"
	Completed  = "completed"
	Failed     = "failed"
)

type Problem struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type Document struct {
	ID               string   `json:"document_id"`
	OriginalFilename string   `json:"original_filename"`
	CorpusID         *string  `json:"corpus_id"`
	Status           string   `json:"status"`
	ExtractionMode   string   `json:"extraction_mode"`
	Failure          *Problem `json:"failure"`
	FileURL          string   `json:"file_url"`
	UploadedAt       string   `json:"uploaded_at"`
}

// Record retains the fields defined by the extractor schema, including original text and nulls.
type Record map[string]any

func (r Record) Text(field string) string {
	value, _ := r[field].(string)
	return value
}

func (r Record) TextPtr(field string) *string {
	if value, ok := r[field].(string); ok {
		return &value
	}
	return nil
}

func (r Record) Number(field string) (float64, bool) {
	return Number(r[field])
}

func Number(value any) (float64, bool) {
	switch value := value.(type) {
	case json.Number:
		number, err := value.Float64()
		return number, err == nil && !math.IsInf(number, 0) && !math.IsNaN(number)
	case float64:
		return value, !math.IsInf(value, 0) && !math.IsNaN(value)
	default:
		return 0, false
	}
}

type MeasuredValue struct {
	Value float64 `json:"value"`
	Unit  string  `json:"unit"`
}

type Issue struct {
	ID          string   `json:"issue_id"`
	CheckName   string   `json:"check_name"`
	Description string   `json:"description"`
	Severity    string   `json:"severity"`
	ResultIDs   []string `json:"result_ids"`
	EvidenceIDs []string `json:"evidence_ids"`
}

type ResultSummary struct {
	DocumentID        string         `json:"document_id"`
	ExtractionMode    string         `json:"extraction_mode"`
	Result            Record         `json:"result"`
	Normalized        *MeasuredValue `json:"normalized"`
	Baseline          *MeasuredValue `json:"baseline"`
	DatasetName       *string        `json:"dataset_name"`
	TaskType          *string        `json:"task_type"`
	ConfigurationName *string        `json:"configuration_name"`
	IssueCount        int            `json:"issue_count"`
}

type DocumentData struct {
	SourceDocument Record          `json:"source_document"`
	Publication    Record          `json:"publication"`
	Configurations []Record        `json:"configurations"`
	Datasets       []Record        `json:"datasets"`
	Tasks          []Record        `json:"tasks"`
	Experiments    []Record        `json:"experiments"`
	Results        []ResultSummary `json:"results"`
	Evidence       []Record        `json:"evidence"`
	Issues         []Issue         `json:"issues"`
}

type DocumentDetail struct {
	Document Document      `json:"document"`
	Data     *DocumentData `json:"data"`
}

type ResultDetail struct {
	ResultSummary
	Publication   Record   `json:"publication"`
	Configuration Record   `json:"configuration"`
	Dataset       Record   `json:"dataset"`
	Task          Record   `json:"task"`
	Experiment    Record   `json:"experiment"`
	Evidence      []Record `json:"evidence"`
	Issues        []Issue  `json:"issues"`
}

func (data DocumentData) ResultDetail(summary ResultSummary) ResultDetail {
	result := summary.Result
	detail := ResultDetail{
		ResultSummary: summary,
		Publication:   data.Publication,
		Configuration: find(data.Configurations, "config_id", result.Text("config_id")),
		Dataset:       find(data.Datasets, "dataset_id", result.Text("dataset_id")),
		Task:          find(data.Tasks, "task_id", result.Text("task_id")),
		Experiment:    find(data.Experiments, "experiment_id", result.Text("experiment_id")),
		Evidence:      []Record{},
		Issues:        []Issue{},
	}
	related := map[string]string{
		"result":        result.Text("result_id"),
		"publication":   data.Publication.Text("publication_id"),
		"configuration": result.Text("config_id"),
		"dataset":       result.Text("dataset_id"),
		"task":          result.Text("task_id"),
		"experiment":    result.Text("experiment_id"),
	}
	sources := map[string]bool{}
	for _, issue := range data.Issues {
		for _, resultID := range issue.ResultIDs {
			if resultID == related["result"] {
				detail.Issues = append(detail.Issues, issue)
				for _, evidenceID := range issue.EvidenceIDs {
					sources[evidenceID] = true
				}
				break
			}
		}
	}
	for _, source := range data.Evidence {
		if related[source.Text("entity_type")] == source.Text("entity_id") || sources[source.Text("evidence_id")] {
			detail.Evidence = append(detail.Evidence, source)
		}
	}
	return detail
}

func find(records []Record, field, id string) Record {
	for _, record := range records {
		if record.Text(field) == id {
			return record
		}
	}
	return nil
}
