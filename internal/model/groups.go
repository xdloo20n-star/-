package model

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"strconv"
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
	GroupID  string        `json:"group_id"`
	Criteria []Criterion   `json:"criteria"`
	Results  []GroupResult `json:"results"`
}

// ComparableGroups clusters only extracted numeric results whose pairwise
// cross-publication comparisons pass every comparison criterion.
func ComparableGroups(details []ResultDetail) []ComparableGroup {
	sort.Slice(details, func(i, j int) bool {
		return details[i].Result.Text("result_id") < details[j].Result.Text("result_id")
	})
	buckets := map[string][]ResultDetail{}
	for _, detail := range details {
		if detail.Result.Text("extraction_status") != "extracted" || detail.Normalized == nil {
			continue
		}
		if fingerprint, ok := groupFingerprint(detail); ok {
			buckets[fingerprint] = append(buckets[fingerprint], detail)
		}
	}

	groups := make([]ComparableGroup, 0, len(buckets))
	for _, bucket := range buckets {
		clusters := [][]ResultDetail{}
		for _, detail := range bucket {
			placed := false
			for i := range clusters {
				compatible := true
				checkedPublications := map[string]bool{}
				for _, member := range clusters[i] {
					identity := publicationIdentity(member)
					if checkedPublications[identity] || !differentPublications(member, detail) {
						continue
					}
					checkedPublications[identity] = true
					if Compare(member, detail).Status != Comparable {
						compatible = false
						break
					}
				}
				if compatible {
					clusters[i] = append(clusters[i], detail)
					placed = true
					break
				}
			}
			if !placed {
				clusters = append(clusters, []ResultDetail{detail})
			}
		}
		for _, cluster := range clusters {
			publications := map[string]bool{}
			for _, detail := range cluster {
				publications[publicationIdentity(detail)] = true
			}
			if len(publications) < 2 {
				continue
			}
			comparable := true
			for i := range cluster {
				for j := i + 1; j < len(cluster); j++ {
					if differentPublications(cluster[i], cluster[j]) && Compare(cluster[i], cluster[j]).Status != Comparable {
						comparable = false
						break
					}
				}
				if !comparable {
					break
				}
			}
			if !comparable {
				continue
			}
			members := make([]GroupResult, 0, len(cluster))
			ids := make([]string, 0, len(cluster))
			for _, detail := range cluster {
				ids = append(ids, detail.Result.Text("result_id"))
				members = append(members, GroupResult{
					ResultID: detail.Result.Text("result_id"), DocumentID: detail.DocumentID,
					PublicationID:    detail.Publication.Text("publication_id"),
					PublicationTitle: detail.Publication.Text("title"), DatasetName: detail.Dataset.Text("dataset_name"),
					ConfigurationName: detail.Configuration.Text("rag_variant"),
					MetricName:        detail.Result.Text("metric_name"), RawValue: detail.Result.Text("raw_value"),
					Normalized: detail.Normalized,
				})
			}
			criteria := []Criterion{}
			for _, detail := range cluster {
				if differentPublications(cluster[0], detail) {
					criteria = Compare(cluster[0], detail).Criteria
					break
				}
			}
			sum := sha256.Sum256([]byte(joinGroupIDs(ids)))
			groups = append(groups, ComparableGroup{
				GroupID: "group_" + hex.EncodeToString(sum[:8]), Criteria: criteria, Results: members,
			})
		}
	}
	sort.Slice(groups, func(i, j int) bool { return groups[i].GroupID < groups[j].GroupID })
	return groups
}

func groupFingerprint(detail ResultDetail) (string, bool) {
	result := detail.Result
	unit, _ := normalizationUnit(result)
	cutoff := result.Text("metric_k_applicability")
	switch cutoff {
	case "not_applicable":
	case "applicable":
		value, ok := result.Number("metric_k")
		if !ok {
			return "", false
		}
		cutoff += ":" + strconv.FormatFloat(value, 'f', -1, 64)
	default:
		return "", false
	}
	values := []string{
		detail.Task.Text("task_type"), detail.Dataset.Text("dataset_name"),
		detail.Dataset.Text("dataset_version"), detail.Dataset.Text("data_split"),
		baseMetric(result), result.Text("metric_variant"), cutoff, unit,
		result.Text("aggregation"), detail.Experiment.Text("evaluation_protocol"),
	}
	for index := range values {
		values[index] = key(values[index])
		if values[index] == "" {
			return "", false
		}
	}
	encoded, err := json.Marshal(values)
	return string(encoded), err == nil
}

func publicationIdentity(detail ResultDetail) string {
	if doi := normalizeDOI(key(detail.Publication.Text("doi"))); doi != "" {
		return "doi:" + doi
	}
	if title := key(detail.Publication.Text("title")); title != "" {
		return "title:" + title
	}
	if id := detail.Publication.Text("publication_id"); id != "" {
		return "publication:" + id
	}
	return "document:" + detail.DocumentID
}

func joinGroupIDs(ids []string) string {
	sort.Strings(ids)
	result := ""
	for _, id := range ids {
		result += id + "\n"
	}
	return result
}
