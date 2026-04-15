"""
predict.py — Load the saved model and classify one or more protein
sequences as thermophilic (1) or mesophilic (0).

Accepts input in two ways:
  1. A raw sequence string via --sequence
  2. A FASTA file via --fasta  (predicts every sequence in the file)

Usage examples:
    python src/predict.py --sequence MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPSLMLFGSIGQIZSGDAAS

    python src/predict.py --fasta my_proteins.fasta

Run from the project root:
    python src/predict.py --sequence <AMINO_ACID_STRING>
"""

import argparse
import json
import sys
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Allow importing utils.py from the src/ directory
sys.path.insert(0, str(Path(__file__).parent))
from utils import compute_aliphatic_index, compute_dipeptide_composition  # noqa: E402

try:
    from Bio import SeqIO
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
except ImportError:
    print("ERROR: biopython is not installed.")
    print("Run: pip install biopython==1.83")
    sys.exit(1)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"

# Standard 20 amino acids
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


# Load model artefacts
def load_artefacts():
    """
    Load the trained HistGradientBoosting model, scaler, selector, and feature name list
    from the models/ directory.

    Returns a tuple: (model, scaler, selector, all_feature_names)

    Raises SystemExit with a helpful message if any file is missing
    (i.e. the user has not run src/train.py yet).
    """
    required = [
        MODELS_DIR / "gb_model.joblib",
        MODELS_DIR / "scaler.joblib",
        MODELS_DIR / "selector.joblib",
        MODELS_DIR / "selected_features.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("ERROR: Model artefacts not found. Run training first:")
        print("  python src/train.py")
        print(f"Missing files: {missing}")
        sys.exit(1)

    model    = joblib.load(MODELS_DIR / "gb_model.joblib")
    scaler   = joblib.load(MODELS_DIR / "scaler.joblib")
    selector = joblib.load(MODELS_DIR / "selector.joblib")

    with open(MODELS_DIR / "selected_features.json") as f:
        feature_info = json.load(f)

    # all_features is the full ordered list of 29 feature names
    # (the same order used when features.csv was built)
    all_features = feature_info["all_features"]

    return model, scaler, selector, all_features


#  Feature extraction for a single sequence 
def sequence_to_features(seq: str, all_feature_names: list) -> np.ndarray:
    """
    Compute the same 429 numerical features used during training for
    a single amino acid sequence.

    Features: 20 amino acid composition + 9 physicochemical +
              400 dipeptide composition = 429 total.

    Parameters
    ----------
    seq              : uppercase amino acid string
    all_feature_names: ordered list of feature column names from training

    Returns
    -------
    np.ndarray, shape (1, 429)  — one row ready for scaler.transform()

    Raises
    ------
    ValueError  if the sequence contains non-standard characters
    ValueError  if the sequence is shorter than 50 or longer than 15000 aa
    """
    seq = seq.upper().strip()

    # Validate length
    if len(seq) < 50:
        raise ValueError(
            f"Sequence length {len(seq)} is too short (minimum 50 amino acids). "
            f"Shorter sequences produce unreliable feature estimates."
        )
    if len(seq) > 15000:
        raise ValueError(
            f"Sequence length {len(seq)} exceeds the 15,000 amino acid limit."
        )

    # Validate characters
    invalid = set(seq) - STANDARD_AA
    if invalid:
        raise ValueError(
            f"Sequence contains non-standard characters: {invalid}. "
            f"Only the 20 standard amino acids are supported."
        )

    pa = ProteinAnalysis(seq)

    # Amino acid composition (20 features, alphabetical order)
    # Use fractions (0-1) to match training (feature_extraction.py: count / len)
    aa_count = pa.count_amino_acids()
    aa_order = list("ACDEFGHIKLMNPQRSTVWY")
    aa_feats = {f"AA_{aa}": aa_count.get(aa, 0) / len(seq) for aa in aa_order}

    # Physicochemical features (9 features)
    helix, turn, sheet = pa.secondary_structure_fraction()
    phys_feats = {
        "molecular_weight":  pa.molecular_weight(),
        "isoelectric_point": pa.isoelectric_point(),
        "gravy":             pa.gravy(),
        "aromaticity":       pa.aromaticity(),
        "instability_index": pa.instability_index(),
        "helix_fraction":    helix,
        "turn_fraction":     turn,
        "sheet_fraction":    sheet,
        "aliphatic_index":   compute_aliphatic_index(seq),
    }

    # Dipeptide composition: 400 features (DP_AA … DP_YY)
    dipeptides = compute_dipeptide_composition(seq)

    # Combine into one dict and convert to a DataFrame row
    all_feats = {**aa_feats, **phys_feats, **dipeptides}

    # Build a single-row DataFrame with columns in the exact training order
    row = pd.DataFrame([all_feats])[all_feature_names]

    return row  # DataFrame shape (1, 429) — keeps feature names for scaler


# Predict one sequence
def predict_sequence(seq: str,
                     model, scaler, selector,
                     all_feature_names: list,
                     seq_id: str = "input") -> dict:
    """
    Run the full inference pipeline for one sequence.

    Pipeline:
        raw sequence
        → 429 features
        → StandardScaler (same parameters as training)
        → SelectKBest   (same top 80 features as training)
        → model.predict() and model.predict_proba()

    Returns a dict with keys:
        id, sequence_length, prediction, label,
        prob_mesophilic, prob_thermophilic
    """
    try:
        X_raw = sequence_to_features(seq, all_feature_names)
    except ValueError as e:
        return {
            "id":               seq_id,
            "sequence_length":  len(seq),
            "prediction":       "ERROR",
            "label":            -1,
            "prob_mesophilic":  None,
            "prob_thermophilic": None,
            "error":            str(e),
        }

    # Apply the same preprocessing as during training
    X_scaled   = scaler.transform(X_raw)       # subtract mean, divide by std
    X_selected = selector.transform(X_scaled)  # keep top 80 features

    label = int(model.predict(X_selected)[0])
    probs = model.predict_proba(X_selected)[0]  # [P(mesophilic), P(thermo)]

    return {
        "id":                seq_id,
        "sequence_length":   len(seq),
        "prediction":        "Thermophilic" if label == 1 else "Mesophilic",
        "label":             label,
        "prob_mesophilic":   round(float(probs[0]), 4),
        "prob_thermophilic": round(float(probs[1]), 4),
    }


#  Print a nicely formatted result
def print_result(r: dict):
    """Print one prediction result to stdout in a readable format."""
    print(f"\n{'-'*50}")
    print(f"  ID              : {r['id']}")
    print(f"  Length          : {r['sequence_length']} amino acids")
    if r["prediction"] == "ERROR":
        print(f"  ERROR           : {r.get('error', 'unknown error')}")
        return
    # Confidence bar (filled blocks proportional to winning probability)
    conf = max(r["prob_mesophilic"], r["prob_thermophilic"])
    filled = int(conf * 20)
    bar  = "#" * filled + "-" * (20 - filled)
    print(f"  Prediction      : {r['prediction'].upper()}")
    print(f"  Confidence      : {conf*100:.1f}%  [{bar}]")
    print(f"  P(Mesophilic)   : {r['prob_mesophilic']:.4f}")
    print(f"  P(Thermophilic) : {r['prob_thermophilic']:.4f}")
    print("-" * 50)


# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Predict whether a protein sequence is thermophilic or mesophilic "
            "using a trained Histogram-based Gradient Boosting classifier."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sequence", "-s",
        type=str,
        help="Single amino acid sequence as a string (uppercase, standard AA).",
    )
    group.add_argument(
        "--fasta", "-f",
        type=str,
        help="Path to a FASTA file. Each sequence will be predicted.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Optional: save results as a CSV file at this path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load trained model artefacts
    model, scaler, selector, all_features = load_artefacts()

    results = []

    if args.sequence:
        # Single sequence mode
        r = predict_sequence(
            args.sequence, model, scaler, selector, all_features,
            seq_id="command_line_input"
        )
        print_result(r)
        results.append(r)

    elif args.fasta:
        # FASTA file mode
        fasta_path = Path(args.fasta)
        if not fasta_path.exists():
            print(f"ERROR: FASTA file not found: {fasta_path}")
            sys.exit(1)

        records = list(SeqIO.parse(str(fasta_path), "fasta"))
        print(f"Found {len(records)} sequences in {fasta_path.name}")

        for record in records:
            r = predict_sequence(
                str(record.seq), model, scaler, selector, all_features,
                seq_id=record.id
            )
            print_result(r)
            results.append(r)

    # Optional CSV output
    if args.output and results:
        out_path = Path(args.output)
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"\nResults saved to: {out_path.resolve()}")

    # Summary when processing multiple sequences
    if len(results) > 1:
        valid = [r for r in results if r["prediction"] != "ERROR"]
        n_thermo = sum(1 for r in valid if r["label"] == 1)
        n_meso   = sum(1 for r in valid if r["label"] == 0)
        n_error  = len(results) - len(valid)
        print(f"\nSummary: {len(valid)} predicted "
              f"({n_thermo} thermophilic, {n_meso} mesophilic)"
              + (f", {n_error} errors" if n_error else ""))


if __name__ == "__main__":
    main()
