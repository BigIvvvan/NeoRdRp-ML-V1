# Data card

## Public bundle

The processed training bundle is distributed at
`BigIvvvan/NeoRdRp-ML-V1/data/neordrp_ml_v1_training_bundle.tar.gz`.

- compressed size: 866,069,978 bytes;
- uncompressed size: 878,681,002 bytes;
- files: 19 (18 parquet tables plus the bundle manifest);
- total parquet rows across splits and feature tables: 1,007,076;
- bundle SHA-256: `9a7d22406267bd2ae83b7bd98b8fa0fb257f6c62ae471714efdbb9cb576fd473`.

The archive contains:

| Group | Rows | Purpose |
| --- | ---: | --- |
| family training | 105,318 | 65-family positives |
| family development | 14,938 | checkpoint selection |
| family calibration | 14,913 | threshold and uncertainty fitting |
| research positive slices | 60,000 | auxiliary research partitions |
| research negative slices | 29,461 | detection negatives |
| weak boundary consensus | 422,204 | start/end weak supervision |
| ESM-C feature caches | 184,121 | 1,152-dimensional pooled features |
| ProTrek feature caches | 184,121 | 1,280-dimensional pooled features |

Some rows are intentionally shared between separate tables, so summing rows
does not estimate biological sample size.

## Provenance

- Family labels represent a curated protein split across 65 viral families.
  The split was constructed with a 90% identity clustering rule.
- Detection negatives are fixed research slices from the project's frozen
  negative pool.
- Weak boundaries are multi-tool consensus intervals retained at agreement
  greater than or equal to 0.50.
- ESM-C and ProTrek embeddings are pooled over 2,048-residue windows with a
  512-residue stride.

The release bundle excludes evaluation-only locked tables, unrelated research
artifacts, and the raw third-party archives. Users remain responsible for
following the terms of the upstream protein, taxonomy, and model sources.

## Intended use

The data support reproduction of the V1 training recipe and methodological
research. They are not a substitute for experimentally curated catalytic-domain
annotations. Weak boundaries are noisy supervision, not exact structural truth.
