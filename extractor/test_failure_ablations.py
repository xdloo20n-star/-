import unittest

from pdf_pages import PDF, Page
from publication import extract_publication
from test_failure_diagnostics import AUDIT, extract, find


# TrustRAG, PDF page 7, Table 2.
ABLATIONS = """Block          Setting                  Method / Condition                   Report          Leak ↓       Stale ↓       Unsup. ↓
                                                Component ablations
                                        w/o Access Filtering       Range                   29.50–36.50    0.00   13.25–24.50
               T EMPLATE
                                        w/o Conflict Reporting     Range                       0.00       0.00   15.75–25.00
Ablation
               F RESH -S TRESS          w/o Supersession Filtering Range                       0.00    0.00–2.00     0.00
               Matched Sets             TrustRAG                   Range                       0.00       0.00       0.00
                                          Adversarial inadmissible evidence
                                        Naive RAG                    Avg.                    36.17          16.83          0.17
                                        Prompt RAG                   Avg.                     0.17           0.00          0.00
Prompt inj. P ROMPT-I NJECTION
                                        Metadata Filter              Avg.                     0.00           0.00          0.00
                                        TrustRAG                     Avg.                    0.00           0.00           0.00

Table 2: Ablation and adversarial stress-test results. Values are failure rates (%), where lower is better. L EAK,
S TALE, and U NSUP. denote unauthorized leakage, stale citation, and unsupported-answer rates, respectively. Range
reports the minimum–maximum failure rate across the three evaluated LLMs, while Avg. reports their macro-average.
Bold numbers indicate the best result within each block.
"""


# TrustRAG, PDF page 8, Table 3. The adjacent caption establishes the column break.
ROBUSTNESS = """Eval.         Metric               Naive      TrustRAG                                           Unsup.
                                                             Method             Leak ↓ Stale ↓          Valid ↑
                                   RAG         / Audit                                             ↓
                   Robustness checks                         Naive RAG           32.00   12.67    30.67    25.67
              Leak ↓              36–38          0           Permission-Aware
S OURCE       Stale ↓             12–31          0           RAG
              Unsup. ↓             0–12          0           Metadata-Aware
              Leak ↓             25–42.5         0           RAG
SF            Stale ↓              0–7.5         0           TrustRAG            0.00     0.00     3.67    96.33
              Unsup. ↓            2.5–10         0
                                                            Table 4: Final-response results with 50-document can-
                   Utility diagnostics
              Valid ↑             52.7–56.0     100
T EMPLATE                                                   macro-averaged across three LLMs.
              Over-refusal ↓         0–1.1       0
              Valid ↑             31.5–59.0     100
F RESH
              Over-refusal ↓         0–3.8       0
                   Judge reliability
Judge         κ Leak               κ Stale    κ Unsup.
DeepSeek-
V3
              1.000                0.939       0.908
Qwen-Max      1.000                0.958       0.607
GLM-4-32B-    0.884                0.958       0.467
0414

Table 3: Robustness, utility, and multi-judge reliabil-     citations on F RESH -S TRESS.
ity diagnostics. For robustness and utility rows, values    lute failure rate is lower than in the Naive RAG
compare Naive RAG with TrustRAG and are percent-            setting, the result still shows that supersession rout-
ages. Ranges are across Qwen-Max, DeepSeek-V3, and          ing is necessary for controlling stale-citation risk.
GLM-4-32B-0414. For the judge-reliability block, each       Multi-version tests further show that TrustRAG
row reports Cohen’s κ between one automatic judge           routes empty contexts caused by inaccessible super-
and human labels for the three failure types. S OURCE,
SF, and F RESH denote SOURCE_ANCHORED, SIL-
ICONFLOW_HELDOUT, and FRESH_STRESS, re-
spectively.
"""


# TrustRAG, PDF page 18, Table B3.
PER_MODEL = """Model           Variant               N     Leak ↓ Stale ↓ Unsup. ↓
            Access-filter ablation on T EMPLATE
Qwen-Max    w/o Access           400 29.50 0.00               24.50
DeepSeek-V3 w/o Access           400 36.00 0.00               22.25
GLM-4-32B-
            w/o Access           400 36.50 0.00               13.25
0414
All         TrustRAG            1200 0.00     0.00             0.00
        Supersession-filter ablation on F RESH -S TRESS
            w/o Supersession
Qwen-Max                          400 0.00       0.00          0.00
            Filtering
            w/o Supersession
DeepSeek-V3                       400 0.00       2.00          0.00
            Filtering
GLM-4-32B- w/o Supersession
                                  400 0.00       1.50          0.00                                      Leak Stale Over- Empty
0414        Filtering
                                                                         Dataset     Condition       N
All         TrustRAG             1200 0.00       0.00          0.00

Table B3: Per-model ablation and prompt-injection                        T EMPLATE
stress results. Abbreviations: w/o Access, w/o Super-                                  50    400       150         68.00      0.00
session Filtering, and w/o Conflict denote removing                                    5     400       400          1.25      0.00
access filtering, supersession filtering, and conflict re-               F RESH -      10    400       400          2.25      0.00
porting, respectively. Rows labeled MacroAvg report                      S TRESS       20    400       400          2.25      0.00
the arithmetic mean over the three evaluated LLMs.
Rows labeled All summarize the matched TrustRAG                          Table B5: Boundary stress diagnostics. The upper panel
runs across the same model set. All values except N are                  reports routing-level risks under 30% metadata noise.
failure rates (%).                                                       Access-label, missing-link, wrong-link, and mixed de-
"""


class FailureAblationsTest(unittest.TestCase):
    def test_decodes_range_endpoints_and_their_setting(self):
        payload = extract(7, ABLATIONS)
        row, config, dataset, source = find(payload, "Table 2", "Leakage rate maximum", "36.50")
        self.assertEqual((row["reported_value"], row["raw_unit"], row["aggregation"],
                          config["rag_variant"], dataset["dataset_name"]),
                         (36.5, "%", "range across 3 LLMs", "w/o Access Filtering", "TEMPLATE"))
        self.assertEqual((source["page"], source["table_row"], source["table_column"]),
                         (7, "TEMPLATE / w/o Access Filtering", "Leak / maximum (Range)"))
        self.assertTrue(any(r["metric_name"] == "Stale citation rate maximum" and r["raw_value"] == "0.00"
                            for r in payload["results"]))

    def test_keeps_wrapped_ablation_model_variant_and_sample_size(self):
        row, config, dataset, source = find(extract(18, PER_MODEL), "Table B3", "Stale citation rate", "1.50")
        self.assertEqual((row["reported_value"], row["raw_unit"], row["sample_size"], row["aggregation"]),
                         (1.5, "%", 400, "per-model rate"))
        self.assertEqual((config["rag_variant"], config["generator_model"], dataset["dataset_name"]),
                         ("w/o Supersession Filtering", "GLM-4-32B-0414", "FRESH-STRESS"))
        self.assertEqual((source["page"], source["table_row"], source["table_column"]),
                         (18, "FRESH-STRESS / GLM-4-32B-0414 / w/o Supersession Filtering", "Stale"))

    def test_robustness_assigns_kappa_to_wrapped_judge_and_conditions(self):
        payload = extract(8, ROBUSTNESS)
        row, config, _, source = find(payload, "Table 3", "Cohen kappa", "0.884")
        self.assertEqual((row["raw_unit"], row["metric_variant"], config["rag_variant"], config["generator_model"]),
                         (None, "Leakage rate", "GLM-4-32B-0414 judge vs human labels", None))
        self.assertEqual((source["page"], source["table_row"], source["table_column"]),
                         (8, "GLM-4-32B-0414", "κ Leak"))
        row, config, dataset, _ = find(payload, "Table 3", "Leakage rate maximum", "42.5")
        self.assertEqual((row["raw_unit"], config["rag_variant"], dataset["dataset_name"]),
                         ("%", "Naive RAG", "SILICONFLOW_HELDOUT"))

    def test_rejects_reversed_range(self):
        payload = extract(7, ABLATIONS.replace("29.50–36.50", "36.50–29.50"))
        self.assertFalse(any(r["metric_name"] == "Leakage rate maximum" and r["raw_value"] == "29.50"
                             for r in payload["results"]))

    def test_summary_kappa_references_detailed_audit_for_same_sample(self):
        title = Page(1, "Metadata-Governed RAG for Question Answering", "text_match")
        section = ("Table 3 summarizes robustness checks, utility diagnostics, and judge reliability.")
        comparison = ("On the same 200 human-audited responses, Cohen’s κ across the three judges ranges "
                      "from 0.884 to 1.000 for unauthorized leakage and from 0.939 to 0.958 for stale citation, "
                      "compared with 0.467 to 0.908 for unsupported answers (Appendix Table B7).")
        pages = [title, Page(8, ROBUSTNESS, "text_match", section),
                 Page(9, comparison, "text_match"), Page(19, AUDIT, "text_match")]
        payload = extract_publication(PDF({"Title": title.text}, pages))
        detailed, _, _, _ = find(payload, "Table B7", "Cohen kappa", "0.884")
        summary, _, _, source = find(payload, "Table 3", "Cohen kappa", "0.884")
        self.assertEqual((summary["duplicate_of"], summary["sample_size"]), (detailed["result_id"], 200))
        self.assertEqual((source["page"], source["table_row"], source["table_column"]),
                         (8, "GLM-4-32B-0414", "κ Leak"))
        pages[2] = Page(9, comparison.replace("same 200", "same 201"), "text_match")
        payload = extract_publication(PDF({"Title": title.text}, pages))
        summary, _, _, _ = find(payload, "Table 3", "Cohen kappa", "0.884")
        self.assertIsNone(summary.get("duplicate_of"))

    def test_appendix_macro_average_repeats_the_same_prompt_injection_run(self):
        prompt = (
            "                  Prompt-injection stress\n"
            "Qwen-Max    Naive RAG         200 46.50               0.00     0.00\n"
            "MacroAvg    Naive RAG         600 36.17              16.83     0.17\n"
            "MacroAvg    Meta. Filter      600 0.00                0.00     0.00\n\n"
        )
        title = Page(1, "Metadata-Governed RAG for Question Answering", "text_match")
        section = "5.3 Ablations and Adversarial Stress Tests\nTable 2 analyzes the contribution of each TrustRAG component."
        appendix_ref = "Table B3 expands the ablation and adversarial stress-test results from Section 5.3."
        appendix_table = PER_MODEL.replace("Table B3:", prompt + "Table B3:")
        pages = [title, Page(7, ABLATIONS, "text_match"), Page(8, section, "text_match"),
                 Page(16, appendix_ref, "text_match"), Page(18, appendix_table, "text_match")]
        payload = extract_publication(PDF({"Title": title.text}, pages))
        summary, _, _, _ = find(payload, "Table 2", "Leakage rate", "36.17")
        appendix, _, _, evidence = find(payload, "Table B3", "Leakage rate", "36.17")
        self.assertEqual((appendix["duplicate_of"], appendix["sample_size"]), (summary["result_id"], 600))
        self.assertEqual((evidence["page"], evidence["table_column"]), (18, "Leak"))
        pages[3] = Page(16, appendix_ref.replace("5.3", "5.2"), "text_match")
        payload = extract_publication(PDF({"Title": title.text}, pages))
        appendix, _, _, _ = find(payload, "Table B3", "Leakage rate", "36.17")
        self.assertIsNone(appendix.get("duplicate_of"))
