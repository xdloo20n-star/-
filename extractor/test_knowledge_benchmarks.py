import unittest

from pdf_pages import PDF, Page
from publication import extract_publication


# Pilot Study on RAG with LLMs, PDF page 4. Tables 1 and 2 share the left
# column, with independent explanations printed alongside them.
GRIDS = """Input Knowledge KP KR KF1 R-L F1                        datasets, K contains text passages consisting of an
GPT-4o-mini
  No knowledge      0 0   0 0.110 0.200 (± .005)        article title and a few knowledge sentences. We per-
 Full knowledge 0.015 1 0.031 0.140 0.251 (± .006)      form knowledge sampling at the sentence level to
 Gold knowledge     1 1   1 0.167 0.276 (± .007)        simulate the end result of applying both the knowl-
LLaMA 3.1 8B
  No knowledge      0 0   0 0.111 0.216 (± 0.005)       edge retrieval and selection steps. More details are
 Full knowledge 0.015 1 0.031 0.138 0.248 (± .005)      in Appendix A.1.1.
 Gold knowledge     1 1   1 0.164 0.278 (± .008)           In our experiments, we refer to using the entire
Mistral 7B Instruct
  No knowledge      0 0   0 0.113 0.203 (± .005)        retrieved knowledge set K provided by the dataset
 Full knowledge     0 1   0 0.131 0.233 (± .005)        as the input knowledge to the generator (i.e. no
 Gold knowledge     1 1   1 0.172 0.268 (± .007)        knowledge selection) as the “full knowledge” set-
Table 1: WoW response generation performance bench-     ting, where both gold and distractor knowledge are
marked by different LLM generators. We measure          present. We also include a “no knowledge” setting,
knowledge precision (KP), recall (KR), and F1 (KF1);    where the generators receive no external knowledge
response ROUGE-L F1 (R-L); and response F1 (and its     at all, as a weak baseline, and “gold knowledge,”
standard error mean).                                   corresponding to perfect knowledge selection, as
                                                        an upper bound.
Input Knowledge KP KR KF1 EM F1
GPT-4o-mini                                             5   Meta-Experimental Results
  No knowledge      0 0   0 0.330 0.437 (± .020)
 Full knowledge 0.065 1 0.120 0.668 0.780 (± .016)      Overall we observe several consistent trends across
 Gold knowledge     1 1   1 0.710 0.828 (± .014)
LLaMA 3.1 8B                                            meta-experiments, except for Mistral-7B-Instruct,
  No knowledge      0 0   0 0.200 0.298 (± .019)        which is a relatively weak-performing LLM. Note
 Full knowledge 0.065 1 0.120 0.372 0.545 (± .019)      that each of the plots in Figures 2-4 corresponds to
 Gold knowledge     1 1   1 0.414 0.671 (± .016)
Mistral 7B Instruct                                     a meta-experiment, and each point corresponds to
  No knowledge      0 0   0 0.208 0.260 (± .019)        a full experiment on the entire test set. The standard
 Full knowledge 0.065 1 0.120 0.046 0.151 (± .011)
 Gold knowledge     1 1   1 0.502 0.627 (± .019)
                                                        error mean values in Tables 1 & 2 indicate that the
                                                        answer F1 scores are robust across all experiments
Table 2: HotpotQA answer generation performance         and the overall observed trends are significant.
benchmarked by different LLM generators. We measure
knowledge precision (KP), recall (KR), and F1 (KF1);    RAG for LLM is beneficial. As Tables 1 & 2
answer exact match (EM); and answer F1 (and its stan-   show, generators without any retrieved knowledge
dard error mean).                                       (“no knowledge”) perform poorly on WoW and
"""

APPENDIX = """We use the API services of OpenAI gpt-4o-mini-2024-0718*, Together AI* meta-llama/Meta-Llama-3.1-8BInstruct-Turbo,
and mistralai/Mistral-7B-Instructv0.1. We report results on the first 500 examples
in the HotpotQA training set and the first 100 conversations (452 wizard utterances) from the “test
seen” set of WoW."""


class KnowledgeBenchmarksTest(unittest.TestCase):
    def extract(self, layout):
        pdf = PDF({"Title": "RAG with LLMs"}, [Page(4, layout, "text_match"), Page(10, APPENDIX, "text_match")])
        return extract_publication(pdf)

    def result_at(self, payload, table, row, column):
        source = next((e for e in payload["evidence"] if e["field"] == "raw_value" and e.get("table") == table
                       and e.get("table_row") == row and e.get("table_column") == column), None)
        self.assertIsNotNone(source, (table, row, column))
        result = next(r for r in payload["results"] if r["result_id"] == source["entity_id"])
        config = next(c for c in payload["configurations"] if c["config_id"] == result["config_id"])
        dataset = next(d for d in payload["datasets"] if d["dataset_id"] == result["dataset_id"])
        return result, config, dataset, source

    def test_knowledge_conditions_and_answer_sem_use_source_layout_and_appendix(self):
        payload = self.extract(GRIDS)
        self.assertEqual(len(payload["results"]), 108)
        result, config, dataset, source = self.result_at(payload, "Table 1", "GPT-4o-mini / No knowledge", "F1 / SEM")
        self.assertEqual((result["metric_name"], result["raw_value"], result["reported_value"], result["raw_unit"],
                          result["reported_scale"], result["aggregation"], result["sample_size"], source["page"],
                          source["verification"]), ("Answer F1 SEM", ".005", .005, None, "fraction",
                                                    "standard error of mean", 452, 4, "text_match"))
        self.assertEqual((config["rag_variant"], config["generator_model"], config["generator_version"], config["top_k"],
                          dataset["dataset_name"], dataset["data_split"]),
                         ("No knowledge", "GPT-4o-mini", "gpt-4o-mini-2024-07-18", None,
                          "WoW", "first 100 conversations from test seen"))
        score, config, dataset, source = self.result_at(payload, "Table 2", "Mistral 7B Instruct / Full knowledge", "EM")
        self.assertEqual((score["metric_name"], score["raw_value"], score["aggregation"], score["sample_size"],
                          config["generator_version"], dataset["dataset_name"], dataset["data_split"], source["page"]),
                         ("EM", "0.046", "mean over examples", 500, "mistralai/Mistral-7B-Instruct-v0.1",
                          "HotpotQA", "first 500 training examples", 4))

    def test_ambiguous_metric_sequence_does_not_claim_a_table(self):
        payload = self.extract(GRIDS.replace("KP KR KF1 R-L", "KR KP KF1 R-L"))
        self.assertEqual(len(payload["results"]), 54)
        self.assertFalse(any(e.get("table") == "Table 1" for e in payload["evidence"]))
