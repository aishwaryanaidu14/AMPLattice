# Training data

The generators use the positive AMP sequences published in OmegAMP's `data/generative-model-data/AMPs.fasta`. Curated negatives from its `data/activity-data/curated-Non-AMPs.fasta` were used for the local pilot classifier. Sources and citations are listed in [DATA_SOURCES.md](DATA_SOURCES.md).

Sequences were restricted to the 20 standard amino acids and lengths of 8–50 residues, then deduplicated. Negative records conflicting with positive labels were removed. The prepared data contain 30,449 positives and 270 negatives; 607 conflicting negative records were excluded.

Sequences connected by normalized Indel similarity of at least 80% were grouped into 14,123 families. Families were assigned to splits using seed 42.

| Split | Positive sequences | Negative sequences |
|---|---:|---:|
| Training | 24,359 | 216 |
| Development | 3,045 | 27 |
| Audit | 3,045 | 27 |

The generators train on positive training sequences. Development data inform checkpoint evaluation, additional sampling and library selection. Audit data are reserved for diagnostics. Negative examples are not used as generator potency labels.

Prepared sequences and the split manifest are included in `assets/pilot/`. Original download URLs and checksums are recorded in `provenance/data_raw_downloads.json`; record-level source identifiers are retained in `provenance/data_record_sources.tsv`.

AMPlify, MBC-Attention and ESM2 use externally pretrained weights. Their training-data sources are included in the source table. The submitted peptides are represented as unmodified linear sequences with free termini; the training FASTAs do not provide complete chemical-modification or assay metadata.
