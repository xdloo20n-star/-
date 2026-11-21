import unittest

from pdf_pages import PDF, Page
from publication import extract_publication


# TrustRAG, PDF page 19, with printed line breaks in the model headings.
AUDIT = """                                                                Failure Valid   Over Refusal Refusal
   Dataset                Model                 System
                                                                 -free Answer -refusal Recall Precision
   T EMPLATE              Qwen-Max              Naive RAG 56.75            52.73         0.00           0.00            0.00
                          GLM-4-32B-
   T EMPLATE                                    Naive RAG        59.00      56.00        0.00           0.00              –
                          0414

   Metric               N          TP      FP         FN     TN          Agreement       Precision            Recall          κ
   Leakage              200        21       0         0      179          100.00              1.000           1.000        1.000
   Stale Citation       200        27       1         2      170           98.50              0.964           0.931        0.939
   Unsupported          200        79       8         1      112           95.50              0.908           0.988        0.908


Table B6: Utility diagnostics and human-audit details. The upper panel tests whether low failure rates come from
over-refusal. The lower panel reports confusion-matrix statistics between the primary DeepSeek-V3 judge and
human labels. Percentages are shown for failure-free, valid-answer, over-refusal, refusal, and agreement metrics; “–”
denotes not applicable.


 Judge              Failure Type                N     TP    FP      FN    TN       Agreement      Precision       Recall          κ
                    Unauthorized Leakage        200   21    0       0     179        100.00           1.000       1.000        1.000
 DeepSeek-V3        Stale Citation              200   27    1       2     170        98.50            0.964       0.931        0.939
                    Unsupported Answer          200   79    8       1     112        95.50            0.908       0.988        0.908
                    Unauthorized Leakage        200   17    0        4    179        98.00            1.000       0.810        0.884
 GLM-4-32B-
                    Stale Citation              200   27    0       2     171        99.00            1.000       0.931        0.958
 0414
                    Unsupported Answer          200   37    4       43    116        76.50            0.902       0.463        0.467

                                        Primary-judge agreement stratified by generator
 Generator                         N                  Leak Agr.                    Stale Agr.                          Unsup. Agr.
 DeepSeek-V3                       72                      100.00                        97.22                                 94.44
 GLM-4-32B-
                                   55                      100.00                       100.00                                100.00
 0414

Table B7: Judge reliability on the same 200 human-audited responses. The upper panel compares all three
judges with the human labels; the lower panel stratifies the primary DeepSeek-V3 judge’s agreement by generator.
Agreement is reported as a percentage, while precision, recall, and Cohen’s κ are proportions. Agreement on
DeepSeek-V3-generated responses is not consistently higher than that on the other generators, providing no evidence
of systematic self-preference within the audited sample.
"""


def extract(number, text):
    title = Page(1, "Metadata-Governed RAG for Question Answering", "text_match")
    return extract_publication(PDF({"Title": title.text}, [title, Page(number, text, "text_match")]))


def find(payload, table, name, raw, source_row=None):
    evidence = {row["entity_id"]: row for row in payload["evidence"]
                if row["entity_type"] == "result" and row["field"] == "raw_value"}
    result, = [row for row in payload["results"] if row["metric_name"] == name and row["raw_value"] == raw
               and evidence[row["result_id"]]["table"] == table
               and (source_row is None or evidence[row["result_id"]]["table_row"] == source_row)]
    config = next(row for row in payload["configurations"] if row["config_id"] == result["config_id"])
    dataset = next(row for row in payload["datasets"] if row["dataset_id"] == result["dataset_id"])
    return result, config, dataset, evidence[result["result_id"]]


class FailureDiagnosticsTest(unittest.TestCase):
    def test_reads_audit_counts_proportions_and_generator_strata(self):
        payload = extract(19, AUDIT)
        count, config, dataset, evidence = find(payload, "Table B6", "TN", "170")
        self.assertEqual((count["reported_value"], count["raw_unit"], count["metric_variant"], count["sample_size"]),
                         (170, None, "Stale citation rate", 200))
        self.assertEqual((config["rag_variant"], config["generator_model"], dataset["dataset_name"]),
                         ("DeepSeek-V3 judge vs human labels", None, None))
        self.assertEqual((evidence["page"], evidence["table_row"], evidence["table_column"], evidence["fragment"]),
                         (19, "DeepSeek-V3 / Stale Citation", "TN", "170"))
        kappa, config, _, evidence = find(payload, "Table B7", "Cohen kappa", "0.467")
        self.assertEqual((kappa["reported_scale"], kappa["raw_unit"], kappa["metric_variant"], config["generator_model"]),
                         ("fraction", None, "Unsupported answer rate", None))
        self.assertEqual((evidence["page"], evidence["table_row"], evidence["table_column"]),
                         (19, "GLM-4-32B-0414 / Unsupported Answer", "κ"))
        agreement, config, _, evidence = find(payload, "Table B7", "Agreement", "97.22")
        self.assertEqual((agreement["raw_unit"], agreement["metric_variant"], agreement["sample_size"],
                          config["rag_variant"], config["generator_model"]),
                         ("%", "Stale citation rate", 72, "DeepSeek-V3 judge vs human labels", "DeepSeek-V3"))
        self.assertEqual((evidence["page"], evidence["table_row"], evidence["table_column"]),
                         (19, "DeepSeek-V3", "Stale Agr."))

    def test_utility_keeps_dashes_distinct_from_printed_zeroes(self):
        payload = extract(19, AUDIT)
        dash, config, dataset, evidence = find(payload, "Table B6", "Refusal precision", "–")
        self.assertEqual((dash["reported_value"], dash["extraction_status"], dash["raw_unit"]),
                         (None, "absent", "%"))
        self.assertEqual((config["rag_variant"], config["generator_model"], dataset["dataset_name"]),
                         ("Naive RAG", "GLM-4-32B-0414", "TEMPLATE"))
        self.assertEqual((evidence["page"], evidence["table_row"], evidence["table_column"]),
                         (19, "TEMPLATE / GLM-4-32B-0414 / Naive RAG", "Refusal Precision"))
        zero, _, _, _ = find(payload, "Table B6", "Refusal precision", "0.00")
        self.assertEqual((zero["reported_value"], zero["extraction_status"]), (0, "extracted"))

    def test_rejects_confusion_counts_that_disagree_with_sample_size(self):
        payload = extract(19, AUDIT.replace("2     170", "2     180"))
        self.assertFalse(any(row["metric_name"] == "TN" and row["raw_value"] == "180" for row in payload["results"]))
        self.assertTrue(any(row["metric_name"] == "TN" and row["raw_value"] == "170" for row in payload["results"]))

    def test_marks_only_equal_cells_on_the_same_audit_sample(self):
        payload = extract(19, AUDIT.replace("0.931        0.939", "0.931        0.938", 1))
        prior, _, _, _ = find(payload, "Table B6", "TN", "170")
        repeated, _, _, source = find(payload, "Table B7", "TN", "170")
        self.assertEqual(repeated["duplicate_of"], prior["result_id"])
        self.assertEqual((source["page"], source["table_row"], source["table_column"]),
                         (19, "DeepSeek-V3 / Stale Citation", "TN"))
        changed, _, _, _ = find(payload, "Table B7", "Cohen kappa", "0.939")
        self.assertIsNone(changed.get("duplicate_of"))
        separate = extract(19, AUDIT.replace("the same 200 human-audited", "another 200 human-audited"))
        separate_row, _, _, _ = find(separate, "Table B7", "TN", "170")
        self.assertIsNone(separate_row.get("duplicate_of"))
