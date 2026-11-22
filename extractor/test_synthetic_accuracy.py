import unittest

from pdf_pages import PDF, Page, Word
from publication import extract_publication


CAPTION = 'Table 2: Our training complements retrieval augmentation. “Ours” denotes the trained model without RAG,'
LAYOUT = f'''Benchmark       Model       Ours       Ours + RAG       Vanilla RAG         ∆
                Llama      68.2%          69.7%            65.3%          +4.4
QuaLITY         Qwen       70.1%          73.6%            69.2%          +4.4
{CAPTION}
“Ours + RAG” denotes the same trained model with RAG, and “Vanilla RAG” denotes the RAG baseline. ∆ denotes
the absolute improvement of “Ours + RAG” over “Vanilla RAG”.
'''
METHOD = 'We generate eight responses per question (n=8) and report the average accuracy for a more robust evaluation.'
MODELS = 'We train the Llama 3.1 8B Instruct model. We also use another model (Qwen3 8B; Yang et al., 2025a).'


def grid(dataset_y=362.85):
    rows = (
        (342.54, [(164, 'Benchmark'), (226, 'Model'), (264, 'Ours'), (293, 'Ours'), (315, '+'),
                  (322, 'RAG'), (352, 'Vanilla'), (381, 'RAG'), (423, '∆')]),
        (357.87, [(226, 'Llama'), (260, '68.2%'), (318, '69.7%'), (377, '65.3%'), (415, '+4.4')]),
        (dataset_y, [(164, 'QuaLITY')]),
        (367.83, [(226, 'Qwen'), (260, '70.1%'), (318, '73.6%'), (377, '69.2%'), (415, '+4.4')]),
        (442.39, [(70, 'Table'), (96, '2:'), (107, 'Our'), (127, 'training'), (165, 'complements'),
                  (225, 'retrieval'), (264, 'augmentation.')]),
    )
    return tuple(Word(value, x, y - 3, x + len(value) * 4, y + 3) for y, words in rows for x, value in words)


class SyntheticAccuracyTest(unittest.TestCase):
    def extract(self, words, method=METHOD):
        pages = [Page(1, 'Retrieval Augmented Generation', 'text_match', MODELS),
                 Page(8, LAYOUT, 'text_match', LAYOUT, words=words, size=(595, 842)),
                 Page(14, method, 'text_match', method)]
        return extract_publication(PDF({}, pages))

    def test_positioned_accuracy_and_signed_delta(self):
        payload = self.extract(grid())
        result = next(row for row in payload['results'] if row['raw_value'] == '69.7%')
        config = next(row for row in payload['configurations'] if row['config_id'] == result['config_id'])
        dataset = next(row for row in payload['datasets'] if row['dataset_id'] == result['dataset_id'])
        source = next(row for row in payload['evidence'] if row['entity_id'] == result['result_id']
                      and row['field'] == 'raw_value')
        self.assertEqual((result['metric_name'], result['aggregation'], result['raw_unit']), ('Accuracy', 'mean', '%'))
        self.assertEqual((config['rag_variant'], config['generator_model'], dataset['dataset_name']),
                         ('Ours + RAG', 'Llama 3.1 8B Instruct', 'QuaLITY'))
        self.assertEqual((source['page'], source['table'], source['table_row'], source['table_column']),
                         (8, 'Table 2', 'QuaLITY / Llama', 'Ours + RAG'))
        self.assertAlmostEqual(source['bbox'][0], 318 / 595)
        self.assertEqual([row['raw_value'] for row in payload['results']
                          if row['metric_name'] == 'Accuracy gain over Vanilla RAG'], ['+4.4', '+4.4'])

    def test_missing_protocol_or_unanchored_dataset_does_not_relabel_scores(self):
        for words, method in ((grid(), 'We report the best run.'), (grid(348.0), METHOD)):
            with self.subTest(method=method):
                self.assertFalse(any(row['metric_name'] == 'Accuracy' for row in self.extract(words, method)['results']))
