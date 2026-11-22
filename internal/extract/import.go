package extract

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/santhosh-tekuri/jsonschema/v6"

	"sverka/internal/model"
)

type Importer struct {
	schema *jsonschema.Schema
}

func NewImporter(rawSchema []byte) (*Importer, error) {
	var document any
	if err := json.Unmarshal(rawSchema, &document); err != nil {
		return nil, fmt.Errorf("decode embedded extractor schema: %w", err)
	}
	compiler := jsonschema.NewCompiler()
	const resource = "extractor.schema.json"
	if err := compiler.AddResource(resource, document); err != nil {
		return nil, err
	}
	schema, err := compiler.Compile(resource)
	if err != nil {
		return nil, err
	}
	return &Importer{schema: schema}, nil
}

type entitySet struct {
	kind    string
	idField string
	rows    []model.Record
}

type entityIndex map[string]map[string]model.Record

type entityReference struct {
	field string
	kind  string
}

var resultReferences = []entityReference{
	{"experiment_id", "experiment"},
	{"config_id", "configuration"},
	{"dataset_id", "dataset"},
	{"task_id", "task"},
	{"baseline_result_id", "result"},
	{"duplicate_of", "result"},
}

func (i *Importer) Prepare(raw []byte, doc model.Document) (*model.DocumentData, error) {
	reader := json.NewDecoder(bytes.NewReader(raw))
	reader.UseNumber()
	var object map[string]any
	if err := reader.Decode(&object); err != nil {
		return nil, fmt.Errorf("decode extractor response: %w", err)
	}
	var extra any
	if err := reader.Decode(&extra); err != io.EOF {
		return nil, errors.New("extractor must return exactly one JSON object")
	}
	if err := i.schema.Validate(object); err != nil {
		return nil, fmt.Errorf("extractor schema: %w", err)
	}
	if object["extraction_mode"] != doc.ExtractionMode {
		return nil, fmt.Errorf("unexpected extraction_mode %q", object["extraction_mode"])
	}

	sourceDocument := model.Record(object["document"].(map[string]any))
	pub := model.Record(object["publication"].(map[string]any))
	configs := records(object, "configurations")
	datasets := records(object, "datasets")
	tasks := records(object, "tasks")
	experiments := records(object, "experiments")
	results := records(object, "results")
	evidence := records(object, "evidence")
	sets := []entitySet{
		{"publication", "publication_id", []model.Record{pub}},
		{"configuration", "config_id", configs},
		{"dataset", "dataset_id", datasets},
		{"task", "task_id", tasks},
		{"experiment", "experiment_id", experiments},
		{"result", "result_id", results},
		{"evidence", "evidence_id", evidence},
	}
	if pub["in_scope"] == false && len(configs)+len(datasets)+len(tasks)+len(experiments)+len(results) != 0 {
		return nil, errors.New("out-of-scope publication has dependent records")
	}
	index, err := indexEntities(sets)
	if err != nil {
		return nil, err
	}
	if err := validateReferences(results, index); err != nil {
		return nil, err
	}
	evidenceByResult, err := validateEvidence(sourceDocument, evidence, index)
	if err != nil {
		return nil, err
	}
	for _, result := range results {
		id := result.Text("result_id")
		if result.Text("extraction_status") == "extracted" && len(evidenceByResult[id]) == 0 {
			return nil, fmt.Errorf("extracted result %q lacks a value source", id)
		}
	}

	summaries := makeResultSummaries(doc, results, index)
	issues := makeIssues(doc, pub, sourceDocument, summaries, evidenceByResult)
	for _, result := range results {
		for _, ref := range resultReferences {
			if id := result.Text(ref.field); id != "" {
				result[ref.field] = scopedID(doc.ID, ref.kind, id)
			}
		}
	}
	for _, item := range evidence {
		item["entity_id"] = scopedID(doc.ID, item.Text("entity_type"), item.Text("entity_id"))
	}
	for _, set := range sets {
		for _, record := range set.rows {
			record[set.idField] = scopedID(doc.ID, set.kind, record.Text(set.idField))
		}
	}
	data := &model.DocumentData{
		SourceDocument: sourceDocument, Publication: pub, Configurations: configs,
		Datasets: datasets, Tasks: tasks, Experiments: experiments, Results: summaries,
		Evidence: evidence, Issues: issues,
	}
	data.RefreshBaselines()
	return data, nil
}

func records(object map[string]any, key string) []model.Record {
	values := object[key].([]any)
	records := make([]model.Record, 0, len(values))
	for _, value := range values {
		records = append(records, model.Record(value.(map[string]any)))
	}
	return records
}

func indexEntities(sets []entitySet) (entityIndex, error) {
	index := make(entityIndex, len(sets))
	for _, set := range sets {
		rows := make(map[string]model.Record, len(set.rows))
		for _, row := range set.rows {
			id := row.Text(set.idField)
			if _, exists := rows[id]; exists {
				return nil, fmt.Errorf("duplicate %s %q", set.idField, id)
			}
			rows[id] = row
		}
		index[set.kind] = rows
	}
	return index, nil
}

func validateReferences(results []model.Record, index entityIndex) error {
	for _, result := range results {
		id := result.Text("result_id")
		for _, ref := range resultReferences {
			target := result.Text(ref.field)
			if target == "" {
				continue
			}
			if index[ref.kind][target] == nil || (ref.kind == "result" && target == id) {
				return fmt.Errorf("result %q has invalid %s %q", id, ref.field, target)
			}
		}
	}

	checked := make(map[string]bool, len(results))
	for _, result := range results {
		var path []string
		visiting := map[string]bool{}
		for id := result.Text("result_id"); id != "" && !checked[id]; id = index["result"][id].Text("duplicate_of") {
			if visiting[id] {
				return fmt.Errorf("duplicate_of cycle at result %q", id)
			}
			visiting[id] = true
			path = append(path, id)
		}
		for _, id := range path {
			checked[id] = true
		}
	}
	return nil
}

func validateEvidence(doc model.Record, evidence []model.Record, index entityIndex) (map[string][]model.Record, error) {
	pages, ok := doc.Number("page_count")
	if !ok || pages > 1<<31-1 {
		return nil, errors.New("invalid document page_count")
	}
	for _, unreadable := range doc["unreadable_pages"].([]any) {
		if page, ok := model.Number(unreadable); !ok || page > pages {
			return nil, fmt.Errorf("unreadable page %v is outside the document", unreadable)
		}
	}
	byResult := make(map[string][]model.Record)
	for _, item := range evidence {
		id := item.Text("evidence_id")
		kind, entityID := item.Text("entity_type"), item.Text("entity_id")
		entity := index[kind][entityID]
		if entity == nil {
			return nil, fmt.Errorf("evidence %q refers to missing %s %q", id, kind, entityID)
		}
		field := item.Text("field")
		if field != "" && entity[field] == nil {
			return nil, fmt.Errorf("evidence %q refers to an absent field %q", id, field)
		}
		if page, ok := item.Number("page"); !ok || page > pages {
			return nil, fmt.Errorf("evidence %q is outside the document", id)
		}
		box, hasBox := item["bbox"].([]any)
		if hasBox {
			x0, validX0 := model.Number(box[0])
			y0, validY0 := model.Number(box[1])
			x1, validX1 := model.Number(box[2])
			y1, validY1 := model.Number(box[3])
			if !validX0 || !validY0 || !validX1 || !validY1 || x0 >= x1 || y0 >= y1 {
				return nil, fmt.Errorf("evidence %q has invalid bbox", id)
			}
		}
		hasLocator := hasBox || item.Text("table") != "" || item.Text("figure") != "" || item.Text("block_id") != ""
		if item.Text("verification") == "locator_only" && !hasLocator {
			return nil, fmt.Errorf("evidence %q has no locator", id)
		}
		if kind == "result" && (field == "" || field == "raw_value") &&
			(hasLocator || strings.TrimSpace(item.Text("fragment")) != "") {
			byResult[entityID] = append(byResult[entityID], item)
		}
	}
	return byResult, nil
}

func makeResultSummaries(doc model.Document, results []model.Record, index entityIndex) []model.ResultSummary {
	summaries := make([]model.ResultSummary, 0, len(results))
	for _, result := range results {
		config := index["configuration"][result.Text("config_id")]
		task := index["task"][result.Text("task_id")]
		dataset := index["dataset"][result.Text("dataset_id")]
		summary := model.ResultSummary{
			DocumentID: doc.ID, ExtractionMode: doc.ExtractionMode, Result: result,
			Normalized: model.Normalized(result), DatasetName: dataset.TextPtr("dataset_name"),
			TaskType: task.TextPtr("task_type"), ConfigurationName: config.TextPtr("rag_variant"),
		}
		summaries = append(summaries, summary)
	}
	return summaries
}

func makeIssues(doc model.Document, publication, sourceDocument model.Record, rows []model.ResultSummary, evidence map[string][]model.Record) []model.Issue {
	issues := []model.Issue{}
	for i := range rows {
		row := &rows[i]
		result := row.Result
		localID := result.Text("result_id")
		sources := evidence[localID]
		sourceIDs := make([]string, 0, len(sources))
		verifiedSource := false
		for _, source := range sources {
			sourceIDs = append(sourceIDs, scopedID(doc.ID, "evidence", source.Text("evidence_id")))
			verification := source.Text("verification")
			verifiedSource = verifiedSource || verification == "text_match" || verification == "ocr_match"
		}
		add := func(rule, message string) {
			issues = append(issues, model.Issue{
				ID: scopedID(doc.ID, "issue", localID+":"+rule), CheckName: rule,
				Severity: "warning", Description: message,
				ResultIDs: []string{scopedID(doc.ID, "result", localID)}, EvidenceIDs: sourceIDs,
			})
			row.IssueCount++
		}
		if row.Normalized != nil && row.Normalized.Unit == "fraction" && model.BoundedFraction(result) &&
			(row.Normalized.Value < 0 || row.Normalized.Value > 1) {
			add("metric_scale", "Значение выходит за диапазон указанной метрики и шкалы.")
		}
		if result.Text("metric_k_applicability") == "applicable" && result["metric_k"] == nil {
			add("metric_k", "Метрика использует k, но его значение не указано.")
		}
		if result.Text("extraction_status") == "extracted" && !verifiedSource && doc.ExtractionMode == "python" {
			add("source_unverified", "Источник значения требует сверки по странице PDF.")
		}
		if result.Text("result_origin") == "cited" && result.Text("cited_publication_id") == "" {
			add("citation_missing", "Обозначение цитируемой публикации не установлено.")
		}
		if result["reported_value"] != nil && row.Normalized == nil {
			add("normalization", "Нормализация недоступна для указанного значения, единицы или шкалы.")
		}
	}
	if pages := sourceDocument["unreadable_pages"].([]any); len(pages) != 0 {
		issues = append(issues, documentIssue(doc.ID, "unreadable_pages", fmt.Sprintf("Не удалось прочитать страницы PDF: %v.", pages)))
	}
	if publication["in_scope"] == true && len(rows) == 0 {
		issues = append(issues, documentIssue(doc.ID, "no_results", "В публикации не извлечено ни одного результата."))
	}
	return issues
}

func documentIssue(docID, rule, message string) model.Issue {
	return model.Issue{
		ID: scopedID(docID, "issue", ":"+rule), CheckName: rule,
		Description: message, Severity: "warning", ResultIDs: []string{}, EvidenceIDs: []string{},
	}
}

func scopedID(docID, kind, id string) string {
	sum := sha256.Sum256([]byte(docID + "\x00" + kind + "\x00" + id))
	return kind + "_" + hex.EncodeToString(sum[:16])
}
