"""Validate publication-local duplicate links using the organizer's CSV rules."""

from decimal import Decimal, DecimalException, ROUND_HALF_UP


DUPLICATE_FIELDS = (
    "experiment_id", "config_id", "dataset_id", "task_id", "metric_name",
    "metric_variant", "metric_k", "metric_unit", "metric_value", "aggregation",
    "sample_size", "result_origin", "cited_publication_id",
)


def csv_cell(value):
    return "" if value is None else str(value).strip()


def same_at_reported_precision(first, second):
    try:
        first, second = Decimal(first), Decimal(second)
        if not first.is_finite() or not second.is_finite():
            return False
        if first == second:
            return True
        first_exponent = first.as_tuple().exponent
        second_exponent = second.as_tuple().exponent
        if first_exponent == second_exponent:
            return False
        coarse, fine = (first, second) if first_exponent > second_exponent else (second, first)
        place = Decimal(1).scaleb(max(first_exponent, second_exponent))
        return fine.quantize(place, rounding=ROUND_HALF_UP) == coarse
    except (DecimalException, ValueError):
        return False


def validate_duplicates(rows):
    by_id = {}
    for row in rows:
        key = csv_cell(row["publication_id"]), csv_cell(row["result_id"])
        if key in by_id:
            raise ValueError(f"{key[0]}/{key[1]}: duplicate result ID")
        by_id[key] = row

    links = {}
    for key, row in by_id.items():
        target_id = csv_cell(row.get("duplicate_of"))
        if not target_id:
            continue
        publication_id, result_id = key
        location = f"{publication_id}/{result_id}"
        target_key = publication_id, target_id
        target = by_id.get(target_key)
        if target is None:
            raise ValueError(f"{location}: duplicate_of target {target_id!r} does not exist in this publication")
        if (csv_cell(row.get("extraction_status")) != "extracted"
                or csv_cell(target.get("extraction_status")) != "extracted"):
            raise ValueError(f"{location}: duplicate_of requires two extracted rows (target {target_id!r})")
        for field in DUPLICATE_FIELDS:
            first, second = csv_cell(row.get(field)), csv_cell(target.get(field))
            if field == "metric_value" and first and second:
                equal = same_at_reported_precision(first, second)
            else:
                equal = first == second
            if not equal:
                raise ValueError(f"{location}: duplicate_of differs in {field} (target {target_id!r})")
        links[key] = target_key

    checked = set()
    for key in links:
        path = set()
        while key in links and key not in checked:
            if key in path:
                raise ValueError(f"{key[0]}/{key[1]}: duplicate_of cycle")
            path.add(key)
            key = links[key]
        checked.update(path)
