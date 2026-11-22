import unittest

from pdf_pages import Page, Word
from publication import Payload, add_table
from rag_tables import Column, Header, Table
from retrieval_tables import EFFICIENCY_HEADINGS
from efficiency_summary import read_efficiency_summaries


ROWS = (
    ('Total Input Tokens', '573,954', '861,691', '+50.13%'),
    ('Total Output Tokens', '343,891', '52,816', '−84.64%'),
    ('Average Input Tokens per File', '2,447.93', '3,669.64', '+49.90%'),
    ('Average Output Tokens per File', '1,467.53', '226.82', '−84.54%'),
    ('Total Processing Time (s)', '2,167.52', '875.42', '−59.61%'),
    ('Average Time per File (s)', '9.23', '3.78', '−59.10%'),
    ('P90 Time (s)', '12.78', '5.83', '−54.38%'),
    ('P95 Time (s)', '14.67', '7.17', '−51.12%'),
)


def summary(*, displaced=False):
    def words(y, text, left):
        return [Word(word, left + index * 15, y - 2, left + index * 15 + 10, y + 2)
                for index, word in enumerate(text.split())]

    header = 'Metric Agentic Chunking W-RAC Relative Change'
    caption = 'Table 5: Aggregate Efficiency Summary'
    positioned = words(101, 'Metric', 132) + words(101, 'Agentic Chunking', 324)
    positioned += words(101, 'W-RAC', 386) + words(101, 'Relative Change', 450)
    for index, (metric, agentic, wrac, change) in enumerate(ROWS):
        y = 113 + index * 11
        positioned += words(y, metric, 132) + words(y, agentic, 337) + words(y, change, 462)
        positioned += words(y, wrac, 366 if displaced and index == 4 else 386)
    positioned += words(201, caption, 225)
    return Page(5, f'{header}\n{caption}\n', 'text_match', words=tuple(positioned), size=(612, 792))


def organizations(*, invalid=False):
    caption = 'Table 4: Token and Runtime Comparison by Organization.'
    page = Page(4, caption, 'text_match')
    columns = [Column(Header(label, name, 0, None, unit), None) for label, name, unit, _ in EFFICIENCY_HEADINGS]
    data = {
        'Agentic Chunking': [
            '191162 96826 60202 379.01 38 2548.05 1584.26 9.97 15.13 18.94',
            '291888 116871 77355 474.18 40 2921.78 1933.88 11.85 14.93 17.27',
            '275169 170890 100534 625.41 71 2406.9 1415.97 8.8 11.7 13.5',
            '170022 107454 64602 433.81 50 2149.08 1292.04 8.68 12.69 13.21',
            '133844 81913 41126 255.11 37 2213.86 1111.51 6.89 9.47 10.46',
        ],
        'W-RAC': [
            '191162 133230 7022 118.6 38 3506.05 184.79 3.12 4.86 6.52',
            '291888 198264 16302 253.59 40 4956.60 407.55 6.34 10.80 11.89',
            '275169 259987 15210 237.76 71 3661.79 214.23 3.35 5.07 7.06',
            '170022 153581 8322 156.29 50 3071.62 166.44 3.13 4.14 5.13',
            '133844 116629 5960 109.18 37 3152.14 161.08 2.95 4.28 5.25',
        ],
    }
    if invalid:
        data['W-RAC'][0] = data['W-RAC'][0].replace('118.6', '111.6')
    return [Table(page, 'Table 4', caption, i, columns, [(method, row.split())], row_section=str(i))
            for method, rows in data.items() for i, row in enumerate(rows)]


class EfficiencySummaryTest(unittest.TestCase):
    def test_located_runtime_has_verified_organization_sum(self):
        page = summary()
        tables = read_efficiency_summaries([page], organizations())
        payload = Payload({}, {})
        for table in tables:
            if table.label == 'Table 5':
                add_table(payload, table, False)
        row = next(row for row in payload.results if row['raw_value'] == '875.42')
        source = next(e for e in payload.evidence if e['entity_id'] == row['result_id'] and e['field'] == 'raw_value')
        config = next(item for item in payload.configurations if row['config_id'] == item['config_id'])
        self.assertEqual((row['metric_name'], row['aggregation'], row['raw_unit'], config['rag_variant']),
                         ('Processing time', 'sum across organizations', 's', 'W-RAC'))
        self.assertEqual((source['page'], source['table'], source['table_row'], source['table_column']),
                         (5, 'Table 5', 'Total Processing Time (s)', 'W-RAC'))
        self.assertAlmostEqual(source['bbox'][0], 386 / 612)

    def test_misaligned_grid_or_unverified_aggregation(self):
        self.assertFalse(any(t.label == 'Table 5' for t in read_efficiency_summaries([summary(displaced=True)], organizations())))
        tables = read_efficiency_summaries([summary()], organizations(invalid=True))
        record = next(t for t in tables if t.label == 'Table 5' and t.rows[0][1] == ['875.42'])
        self.assertIsNone(record.columns[0].aggregation)
