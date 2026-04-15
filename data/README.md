# Data

## proteins.csv

| Column | Type | Content |
|--------|------|---------|
| `id` | string | UniProt accession |
| `sequence` | string | Amino acid sequence |
| `label` | int | 1 = thermophilic, 0 = mesophilic |

16,500 proteins: 8,250 thermophilic + 8,250 mesophilic. Balanced, standard AAs only (ACDEFGHIKLMNPQRSTVWY), length 50-5000.

## features.csv

429 features per protein, all derived from sequence only.

**Amino acid composition (20):** `AA_A`, `AA_C`, ... `AA_Y` — fraction of each AA.

**Dipeptide composition (400):** `DP_AA`, `DP_AC`, ... `DP_YY` — fraction of each pair. Captures local context.

**Physicochemical (9):** molecular_weight, isoelectric_point, gravy, aromaticity, instability_index, helix_fraction, turn_fraction, sheet_fraction, aliphatic_index.

**Label:** 1 or 0 (same as proteins.csv).

Total: 430 columns (429 features + 1 label).

## Generate

```bash
python data/download_data.py        # → proteins.csv
python src/feature_extraction.py    # → features.csv
```

## Why UniProt?

Manually curated (expert verified), standardized, reproducible, used in published research.
