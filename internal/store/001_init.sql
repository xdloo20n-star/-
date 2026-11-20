CREATE TABLE documents (
    document_id TEXT PRIMARY KEY,
    original_filename TEXT NOT NULL,
    corpus_id TEXT,
    status TEXT NOT NULL CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
    extraction_mode TEXT NOT NULL CHECK (extraction_mode IN ('mock', 'python')),
    failure_code TEXT,
    failure_message TEXT,
    uploaded_at TEXT NOT NULL,
    data_json TEXT,
    CHECK ((status = 'completed') = (data_json IS NOT NULL)),
    CHECK ((status = 'failed') = (failure_code IS NOT NULL AND failure_message IS NOT NULL))
);

CREATE INDEX documents_queue ON documents (status, uploaded_at, document_id);
CREATE INDEX documents_uploaded ON documents (uploaded_at DESC, document_id DESC);

-- Search and pagination fields for the immutable results in each document snapshot.
CREATE TABLE results (
    result_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents (document_id) ON DELETE CASCADE,
    dataset_id TEXT,
    task_id TEXT,
    config_id TEXT,
    metric_key TEXT NOT NULL,
    has_issues INTEGER NOT NULL CHECK (has_issues IN (0, 1)),
    search_text TEXT NOT NULL,
    summary_json TEXT NOT NULL
);
CREATE INDEX results_document ON results (document_id, result_id);
CREATE INDEX results_dataset ON results (dataset_id, result_id);
CREATE INDEX results_task ON results (task_id, result_id);
CREATE INDEX results_config ON results (config_id, result_id);
CREATE INDEX results_metric ON results (metric_key, result_id);
