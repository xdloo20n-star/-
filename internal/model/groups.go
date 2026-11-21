package model

import (
	"crypto/sha256"
	"encoding/hex"
	"sort"
	"strconv"
	"strings"
)

type GroupResult struct {
	ResultID          string         `json:"result_id"`
	DocumentID        string         `json:"document_id"`
	PublicationID     string         `json:"publication_id"`
	PublicationTitle  string         `json:"publication_title"`
	DatasetName       string         `json:"dataset_name"`
	ConfigurationName string         `json:"configuration_name"`
	MetricName        string         `json:"metric_name"`
	RawValue          string         `json:"raw_value"`
	Normalized        *MeasuredValue `json:"normalized"`
}

type ComparableGroup struct {
	GroupID        string        `json:"group_id"`
	ExtractionMode string        `json:"extraction_mode"`
	Criteria       []Criterion   `json:"criteria"`
	Results        []GroupResult `json:"results"`
}

// AppendGroupCandidates builds the comparison context without loading sources or issues.
func (data DocumentData) AppendGroupCandidates(details []ResultDetail) []ResultDetail {
	datasets := recordsByID(data.Datasets, "dataset_id")
	tasks := recordsByID(data.Tasks, "task_id")
	experiments := recordsByID(data.Experiments, "experiment_id")
	for _, summary := range data.Results {
		if !groupCandidate(summary) {
			continue
		}
		result := summary.Result
		details = append(details, ResultDetail{
			ResultSummary: summary,
			Publication:   data.Publication,
			Dataset:       datasets[result.Text("dataset_id")],
			Task:          tasks[result.Text("task_id")],
			Experiment:    experiments[result.Text("experiment_id")],
		})
	}
	return details
}

func recordsByID(records []Record, field string) map[string]Record {
	index := make(map[string]Record, len(records))
	for _, record := range records {
		index[record.Text(field)] = record
	}
	return index
}

func groupCandidate(summary ResultSummary) bool {
	return summary.Normalized != nil && summary.Result.Text("extraction_status") == "extracted" &&
		summary.Result.Text("result_origin") == "primary"
}

type groupKey [11]string

// ComparableGroups groups primary numeric results with matching RAG criteria.
func ComparableGroups(details []ResultDetail) []ComparableGroup {
	// A publication can supply a title alias even when its results have different comparison criteria.
	aliases := collectPublicationTitleAliases(details)
	buckets := map[groupKey][]ResultDetail{}
	for _, detail := range details {
		if !groupCandidate(detail.ResultSummary) {
			continue
		}
		if publicationDOI(detail) == "" && key(detail.Publication.Text("title")) == "" {
			continue
		}
		if fingerprint, ok := groupFingerprint(detail); ok {
			buckets[fingerprint] = append(buckets[fingerprint], detail)
		}
	}

	groups := make([]ComparableGroup, 0, len(buckets))
	for _, bucket := range buckets {
		if len(bucket) < 2 {
			continue
		}
		sort.Slice(bucket, func(i, j int) bool {
			return bucket[i].Result.Text("result_id") < bucket[j].Result.Text("result_id")
		})
		first, second, ok := distinctPublicationPair(bucket, aliases)
		if !ok {
			continue
		}
		comparison := Compare(bucket[first], bucket[second])
		if comparison.Status != Comparable {
			continue
		}
		members := make([]GroupResult, 0, len(bucket))
		ids := make([]string, 0, len(bucket))
		for _, detail := range bucket {
			configurationName := ""
			if detail.ConfigurationName != nil {
				configurationName = *detail.ConfigurationName
			}
			ids = append(ids, detail.Result.Text("result_id"))
			members = append(members, GroupResult{
				ResultID: detail.Result.Text("result_id"), DocumentID: detail.DocumentID,
				PublicationID:    detail.Publication.Text("publication_id"),
				PublicationTitle: detail.Publication.Text("title"), DatasetName: detail.Dataset.Text("dataset_name"),
				ConfigurationName: configurationName,
				MetricName:        detail.Result.Text("metric_name"), RawValue: detail.Result.Text("raw_value"),
				Normalized: detail.Normalized,
			})
		}
		// Put the confirmed pair first so the UI compares distinct publications.
		members[0], members[first] = members[first], members[0]
		members[1], members[second] = members[second], members[1]
		sum := sha256.Sum256([]byte(strings.Join(ids, "\n") + "\n"))
		groups = append(groups, ComparableGroup{
			GroupID:        "group_" + hex.EncodeToString(sum[:8]),
			ExtractionMode: bucket[first].ExtractionMode,
			Criteria:       comparison.Criteria,
			Results:        members,
		})
	}
	sort.Slice(groups, func(i, j int) bool { return groups[i].GroupID < groups[j].GroupID })
	return groups
}

func groupFingerprint(detail ResultDetail) (groupKey, bool) {
	result := detail.Result
	unit, _ := normalizationUnit(result)
	cutoff := result.Text("metric_k_applicability")
	switch cutoff {
	case "not_applicable":
	case "applicable":
		value, ok := result.Number("metric_k")
		if !ok {
			return groupKey{}, false
		}
		cutoff += ":" + strconv.FormatFloat(value+0, 'f', -1, 64)
	default:
		return groupKey{}, false
	}
	// Include every equality criterion from Compare and keep extraction modes separate.
	values := groupKey{
		detail.ExtractionMode,
		detail.Task.Text("task_type"), detail.Dataset.Text("dataset_name"),
		detail.Dataset.Text("dataset_version"), detail.Dataset.Text("data_split"),
		baseMetric(result), result.Text("metric_variant"), cutoff, unit,
		result.Text("aggregation"), detail.Experiment.Text("evaluation_protocol"),
	}
	for index := range values {
		values[index] = key(values[index])
		if values[index] == "" {
			return groupKey{}, false
		}
	}
	return values, true
}

type publicationTitle struct {
	mode, title string
}

// An empty DOI marks a title associated with multiple distinct DOIs.
type publicationTitleAliases map[publicationTitle]string

func collectPublicationTitleAliases(details []ResultDetail) publicationTitleAliases {
	aliases := make(publicationTitleAliases)
	for _, detail := range details {
		if !groupCandidate(detail.ResultSummary) {
			continue
		}
		title, doi := key(detail.Publication.Text("title")), publicationDOI(detail)
		if title == "" || doi == "" {
			continue
		}
		alias := publicationTitle{mode: detail.ExtractionMode, title: title}
		if previous, found := aliases[alias]; found && previous != doi {
			aliases[alias] = ""
		} else {
			aliases[alias] = doi
		}
	}
	return aliases
}

func (aliases publicationTitleAliases) identity(detail ResultDetail) (doi, title string) {
	doi, title = publicationDOI(detail), key(detail.Publication.Text("title"))
	if doi != "" {
		return doi, title
	}
	alias, found := aliases[publicationTitle{mode: detail.ExtractionMode, title: title}]
	if found && alias == "" {
		return "", ""
	}
	return alias, title
}

func distinctPublicationPair(bucket []ResultDetail, aliases publicationTitleAliases) (int, int, bool) {
	firstDOI := ""
	withDOI, withoutDOI := -1, -1
	for i, detail := range bucket {
		doi, title := aliases.identity(detail)
		switch {
		case doi == "":
			if withoutDOI == -1 && title != "" {
				withoutDOI = i
			}
		case withDOI == -1:
			firstDOI, withDOI = doi, i
		case doi != firstDOI:
			return withDOI, i, true
		}
	}
	if withoutDOI != -1 {
		title := key(bucket[withoutDOI].Publication.Text("title"))
		for i, detail := range bucket {
			if _, other := aliases.identity(detail); other != "" && other != title {
				return min(i, withoutDOI), max(i, withoutDOI), true
			}
		}
	}
	return 0, 0, false
}

func publicationDOI(detail ResultDetail) string {
	value := key(detail.Publication.Text("doi"))
	for _, prefix := range []string{"doi:", "https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi.org/"} {
		value = strings.TrimSpace(strings.TrimPrefix(value, prefix))
	}
	return strings.TrimRight(value, ".,; ")
}
