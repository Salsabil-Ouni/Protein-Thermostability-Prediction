# Protein Thermostability Prediction

Salsabil Ouni — April 2026

## What's this?

Thermophilic proteins come from organisms living in extreme heat (hot springs, deep sea vents, boiling water). They stay folded and work at high temps where normal proteins fall apart. We're trying to predict if a protein is thermophilic just from its amino acid sequence.

Thermophiles have more hydrophobic residues (aliphatic amino acids), more salt bridges, higher GRAVY scores. These patterns show up in the sequence, so ML models can detect them.

Useful for: PCR, industrial enzymes, biofuel production.

## Task

Binary classification: given a protein sequence, predict **Thermophilic** (label 1) or **Mesophilic** (label 0).

## Data

16,500 proteins from UniProt SwissProt (8,250 thermophilic, 8,250 mesophilic). 50-5000 amino acids, standard 20 AAs only. 80/20 train/test split.

---

## File structure

```
data/
  download_data.py      # Get proteins from UniProt
  proteins.csv          # 16.5k sequences
  features.csv          # 429 features
src/
  feature_extraction.py # Compute features
  train.py              # Train model
  predict.py            # Make predictions
  utils.py              # Helpers
app.py                  # Streamlit UI
notebooks/
  01_EDA.ipynb          # Data exploration
  02_Modeling.ipynb     # Model comparison
```

## Setup

```bash
pip install -r requirements.txt
# or conda env create -f environment.yml
```

## Run

1. Download data: `python data/download_data.py`
2. Extract features: `python src/feature_extraction.py`
3. Train model: `python src/train.py`
4. Predict: `python src/predict.py --sequence MKTAYIAK...`
5. Web UI: `streamlit run app.py`
6. EDA: `jupyter notebook notebooks/01_EDA.ipynb`
7. Modeling: `jupyter notebook notebooks/02_Modeling.ipynb`

## Results

Best model: HistGradientBoosting with top 80 features.

- Test accuracy: 93.8%
- F1: 93.8%
- ROC-AUC: 0.983

Selected features: 63 dipeptides, 11 amino acid composition, 6 physicochemical.

## Features

- 20 amino acid composition fractions
- 9 physicochemical properties (GRAVY, aliphatic index, etc.)
- 400 dipeptide frequencies

Total: 429 features, select top 80 with ANOVA F-test.

## Limits

- Org-level labels (not all proteins from thermophiles are actually heat-stable)
- Sequence only (no 3D structure)
- Limited coverage of some organisms
