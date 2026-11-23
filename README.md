# NeoRdRp-ML V1 result review

This offline companion compares a recorded V1 candidate interval with a reference interval and the full-protein baseline. The bundled example uses coordinates from one completed evaluation record. The command computes the comparison live from those coordinates. Its input is a JSON result record.

[日本語](README.ja.md)

## Install and run

From the repository root, using Python 3.11 or later:

```sh
python -m pip install .
neordrp-ml-v1-review
```

A built wheel supports the same command after installation. The package requires no GPU or network access at runtime.

## Single-record demonstration

The bundled record has an input length of 1,699 residues, a reference interval of 1,405–1,694 and a recorded candidate interval of 14–1,676. The demo checks whether both endpoints fall within 20 residues of the reference. The expected signed errors are −1,391 and −18 residues. The intersection-over-union is about 0.1618 for the candidate and 0.1707 for the full-protein baseline. The reference is a structure-module envelope. All coordinates are 1-based and inclusive.

Use `neordrp-ml-v1-review path/to/record.json` to assess another saved interval. The required fields are `schema_version` (currently `1`), `input_length`, `reference_interval`, and `predicted_interval`. Each interval is a pair of integers within `[1, input_length]`. Additional provenance fields are carried into the report. The included JSON carries the frozen checkpoint SHA-256.
