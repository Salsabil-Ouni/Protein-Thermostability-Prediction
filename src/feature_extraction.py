"""
feature_extraction.py — Read data/proteins.csv and compute a 27-column
feature matrix for each protein sequence.  Saves data/features.csv.

Run from the project root:
    python src/feature_extraction.py
"""

import sys
from pathlib import Path

import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Allow importing utils.py from this same src/ directory
sys.path.insert(0, str(Path(__file__).parent))
from utils import compute_aliphatic_index, compute_dipeptide_composition  # noqa: E402

# ── File paths (all relative to the project root) ─────────────────
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_PATH   = PROJECT_ROOT / "data" / "proteins.csv"
OUTPUT_PATH  = PROJECT_ROOT / "data" / "features.csv"


def extract_features(seq: str) -> dict:
    """Extract 429 numerical features from a protein sequence.
    
    Returns:
        dict with keys like 'AA_A' (amino acid %), 'gravy', 'DP_AK' (dipeptides), etc.
    """
    # Create a ProteinAnalysis object for this sequence
    pa = ProteinAnalysis(seq)

    # Get amino acid composition (what % of sequence is each AA)
    aa_comp = pa.get_amino_acids_percent()
    aa_order = list("ACDEFGHIKLMNPQRSTVWY")
    aa_features = {f"AA_{aa}": aa_comp.get(aa, 0.0) for aa in aa_order}

    # Calculate molecular weight
    mw = pa.molecular_weight()

    # Isoelectric point - pH where protein has zero charge
    pi = pa.isoelectric_point()

    # GRAVY score - measures how hydrophobic the protein is
    gravy = pa.gravy()

    # Aromaticity - % of F, W, Y amino acids
    arom = pa.aromaticity()

    # Instability index - predicts if protein is stable (< 40 = stable, >= 40 = unstable)
    instab = pa.instability_index()

    # Estimate secondary structure fractions (helix, turn, sheet)
    helix, turn, sheet = pa.secondary_structure_fraction()

    # Aliphatic index - formula: (A + 2.9*V + 3.9*(I+L)) * 100
    aliphatic = compute_aliphatic_index(seq)

    # Dipeptide composition - frequency of all 400 adjacent AA pairs
    dipeptides = compute_dipeptide_composition(seq)

    # Combine all features into one dictionary
    features = {
        **aa_features,
        "molecular_weight":  mw,
        "isoelectric_point": pi,
        "gravy":             gravy,
        "aromaticity":       arom,
        "instability_index": instab,
        "helix_fraction":    helix,
        "turn_fraction":     turn,
        "sheet_fraction":    sheet,
        "aliphatic_index":   aliphatic,
        **dipeptides,        # 400 DP_XY features
    }

    return features


def main():
    print("=" * 60)
    print("Protein Thermostability — Feature Extraction")
    print("=" * 60)

    # Step 1: Load proteins.csv
    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found.")
        print("Run  python data/download_data.py  first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} proteins from {INPUT_PATH}")

    # Step 2: Extract features for every sequence
    feature_rows = []   # list of dicts, one per protein
    n_skipped = 0       # count of sequences that raised errors

    for idx, row in df.iterrows():
        seq = str(row["sequence"]).upper().strip()

        try:
            feats = extract_features(seq)
        except Exception as e:
            # Some sequences may cause BioPython errors (e.g.,
            # sequences with unusual characters that slipped through).
            # We log the ID and skip rather than crashing the pipeline.
            print(f"  SKIPPED sequence at index {idx} "
                  f"(id={row['id']}): {e}")
            n_skipped += 1
            continue

        # Attach the label (0 or 1) so features.csv is self-contained
        feats["label"] = int(row["label"])
        feature_rows.append(feats)

    print(f"\nExtracted features for {len(feature_rows)} proteins.")
    if n_skipped > 0:
        print(f"Skipped {n_skipped} sequences due to computation errors.")

    # Step 3: Build feature DataFrame
    features_df = pd.DataFrame(feature_rows)

    # Step 4: Save to data/features.csv
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUTPUT_PATH, index=False)

    # Step 5: Print summary
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETE — Summary")
    print("=" * 60)
    print(f"  Output shape : {features_df.shape}  "
          f"({features_df.shape[0]} proteins × "
          f"{features_df.shape[1] - 1} features + 1 label)")
    print(f"  Saved to     : {OUTPUT_PATH.resolve()}")
    print("\nFirst 3 rows (selected columns):")
    preview_cols = ["AA_A", "molecular_weight", "gravy",
                    "isoelectric_point", "aliphatic_index", "label"]
    print(features_df[preview_cols].head(3).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
