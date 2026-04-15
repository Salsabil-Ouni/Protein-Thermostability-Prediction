"""
utils.py — Shared helper functions for the thermostability ML project.

Each function is self-contained and can be imported from any script
or notebook in the project.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test):
    """Calculate accuracy, F1 score, and ROC-AUC on test data."""
    # Generate hard class predictions (0 or 1)
    y_pred = model.predict(X_test)

    # Generate probability estimates
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "f1":       round(f1_score(y_test, y_pred), 3),
        "roc_auc":  round(roc_auc_score(y_test, y_prob), 3),
    }


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot a confusion matrix heatmap."""
    # Compute the 2×2 confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Class names for axis labels
    labels = ["Mesophilic (0)", "Thermophilic (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))

    # Draw the heatmap; annot=True writes the count inside each cell
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",             # show integers, not floats
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
    )

    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Actual Class", fontsize=11)
    ax.set_xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()
    plt.show()


def compute_aliphatic_index(sequence):
    """Compute aliphatic index - a measure of protein thermal stability.
    
    Uses formula: Aliphatic Index = (A + 2.9*V + 3.9*(I+L)) * 100
    where A, V, I, L are fractions of Ala, Val, Ile, Leu.
    """
    # Guard against empty strings to avoid ZeroDivisionError
    if not sequence:
        return 0.0

    # Convert to uppercase to handle lower-case input
    seq = sequence.upper()
    n = len(seq)

    # Count occurrences of each relevant amino acid
    a = seq.count("A") / n  # Alanine mole fraction
    v = seq.count("V") / n  # Valine mole fraction
    i = seq.count("I") / n  # Isoleucine mole fraction
    l = seq.count("L") / n  # Leucine mole fraction

    # Apply Ikai's formula and scale by 100
    aliphatic = (a + 2.9 * v + 3.9 * (i + l)) * 100

    return round(aliphatic, 4)


# Alphabetical order of the 20 standard amino acids — used by both
# dipeptide composition and feature extraction to guarantee consistent
# column ordering across all scripts.
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def compute_dipeptide_composition(sequence: str) -> dict:
    """Compute dipeptide composition - frequency of all adjacent AA pairs.
    
    There are 400 possible dipeptides (20 * 20). Each feature is the 
    fraction of that pair in the sequence. Dipeptides capture local 
    sequence patterns that single amino acids miss.
    """
    seq = sequence.upper()
    n_pairs = len(seq) - 1  # number of adjacent pairs

    # Initialise all 400 counts to zero
    counts = {}
    for a in AA_ORDER:
        for b in AA_ORDER:
            counts[a + b] = 0

    if n_pairs <= 0:
        return {f"DP_{pair}": 0.0 for pair in counts}

    # Count every adjacent pair in one pass through the sequence
    for i in range(n_pairs):
        pair = seq[i: i + 2]
        if pair in counts:          # ignore pairs with non-standard AA
            counts[pair] += 1

    # Convert counts to fractions
    return {f"DP_{pair}": round(cnt / n_pairs, 6)
            for pair, cnt in counts.items()}


# Pre-compute all 8000 tripeptide keys once at module level for speed
_TRIPEPTIDES = [a + b + c
                for a in AA_ORDER
                for b in AA_ORDER
                for c in AA_ORDER]


def compute_tripeptide_composition(sequence: str) -> dict:
    """Compute tripeptide composition - frequency of all adjacent AA triplets.
    
    There are 8,000 possible tripeptides (20 * 20 * 20).
    Captures longer-range sequence patterns than dipeptides.
    """
    from collections import Counter
    seq = sequence.upper()
    n_triplets = len(seq) - 2

    if n_triplets <= 0:
        return {f"TP_{t}": 0.0 for t in _TRIPEPTIDES}

    counter = Counter(
        seq[i:i + 3] for i in range(n_triplets)
        if seq[i] in AA_ORDER and seq[i + 1] in AA_ORDER and seq[i + 2] in AA_ORDER
    )
    return {f"TP_{t}": round(counter.get(t, 0) / n_triplets, 6)
            for t in _TRIPEPTIDES}
