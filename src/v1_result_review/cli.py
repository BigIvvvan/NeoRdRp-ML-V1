"""Compare recorded intervals on a single input coordinate system."""

import argparse
import json
from importlib.resources import files
from pathlib import Path


def validate_interval(value, length, name):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain two coordinates")
    start, end = value
    if any(type(position) is not int for position in value) or not 1 <= start <= end <= length:
        raise ValueError(f"{name} must be a valid 1-based inclusive interval")
    return start, end


def compare(record):
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("Expected a record with schema_version 1")
    length = record["input_length"]
    if type(length) is not int or length < 1:
        raise ValueError("input_length must be a positive integer")
    ref_start, ref_end = validate_interval(record["reference_interval"], length, "reference_interval")
    start, end = validate_interval(record["predicted_interval"], length, "predicted_interval")
    # Inclusive intervals count both coordinate endpoints as residues.
    overlap = max(0, min(ref_end, end) - max(ref_start, start) + 1)
    reference_span = ref_end - ref_start + 1
    union = reference_span + (end - start + 1) - overlap
    start_error = start - ref_start
    end_error = end - ref_end
    return {
        **record,
        "start_error_aa": start_error,
        "end_error_aa": end_error,
        "interval_iou": overlap / union,
        "reference_coverage": overlap / reference_span,
        "both_within_20_aa": max(abs(start_error), abs(end_error)) <= 20,
        "full_protein_baseline_iou": reference_span / length,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, nargs="?", help="JSON record. Defaults to the bundled demo.")
    args = parser.parse_args()
    try:
        source = args.record if args.record is not None else files(__package__).joinpath("demo.json")
        report = compare(json.loads(source.read_text(encoding="utf-8")))
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
