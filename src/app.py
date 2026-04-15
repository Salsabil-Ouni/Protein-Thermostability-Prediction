"""
app.py — Streamlit web interface for the Protein Thermostability Predictor.

Allows users to:
  - Paste a raw amino acid sequence
  - Upload a FASTA file (single or multi-sequence)
  - See prediction, confidence gauge, probability bar, and feature breakdown

Run from the project root:
    python -m streamlit run app.py
"""

import json
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).parent / "src"))
from predict import load_artefacts, sequence_to_features  # noqa: E402
from Bio import SeqIO                                      # noqa: E402

#  Config 
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR   = PROJECT_ROOT / "models"
STANDARD_AA  = set("ACDEFGHIKLMNPQRSTVWY")

st.set_page_config(
    page_title="Protein Thermostability Predictor",
    page_icon="P",
    layout="wide",
)


#  Load artefacts 
def load_model():
    """Load model, scaler, selector, and feature names from models/."""
    required = [
        MODELS_DIR / "gb_model.joblib",
        MODELS_DIR / "scaler.joblib",
        MODELS_DIR / "selector.joblib",
        MODELS_DIR / "selected_features.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return None, None, None, None, missing
    model, scaler, selector, all_features = load_artefacts()
    return model, scaler, selector, all_features, []


def clean_sequence(seq: str) -> tuple[str, set]:
    """Strip non-standard characters; return (cleaned_seq, removed_chars)."""
    standard = set("ACDEFGHIKLMNPQRSTVWY")
    removed  = set(seq) - standard
    cleaned  = "".join(c for c in seq if c in standard)
    return cleaned, removed


#  Predict one sequence
def predict(seq: str, model, scaler, selector, all_features: list) -> dict:
    """Run the full inference pipeline and return a result dict."""
    try:
        X_raw      = sequence_to_features(seq, all_features)
        X_scaled   = scaler.transform(X_raw)
        X_selected = selector.transform(X_scaled)

        label = int(model.predict(X_selected)[0])
        probs = model.predict_proba(X_selected)[0]

        return {
            "ok":           True,
            "label":        label,
            "prediction":   "Thermophilic" if label == 1 else "Mesophilic",
            "prob_meso":    float(probs[0]),
            "prob_thermo":  float(probs[1]),
            "features":     X_raw,
            "error":        None,
        }
    except ValueError as e:
        return {"ok": False, "error": str(e)}


#  Render a single result card 
def render_result(r: dict, seq: str, seq_id: str = ""):
    """Display prediction result using native Streamlit components."""

    if not r["ok"]:
        st.error(f"**Error:** {r['error']}")
        return

    # Warn when sequence is longer than the training range (50–5000 aa)
    if len(seq) > 5000:
        st.warning(
            f"Sequence length ({len(seq)} aa) exceeds the training range "
            f"(max 5,000 aa). The prediction may be less reliable — "
            f"all features are fraction-based so they still work, but "
            f"this length was not seen during training."
        )

    is_thermo = r["label"] == 1
    conf      = max(r["prob_meso"], r["prob_thermo"]) * 100
    

    # Result banner
    banner_text = (
        f"{r['prediction'].upper()}  "
        f"— Confidence: **{conf:.1f}%**  "
        f"| Length: **{len(seq)} aa**"
        + (f"  | `{seq_id}`" if seq_id else "")
    )
    if is_thermo:
        st.success(banner_text)
    else:
        st.info(banner_text)

    #  Probability metrics + progress bars 
    col1, col2 = st.columns(2)
    with col1:
        st.metric(" P(Thermophilic)", f"{r['prob_thermo']*100:.2f}%")
        st.progress(r["prob_thermo"])
    with col2:
        st.metric(" P(Mesophilic)", f"{r['prob_meso']*100:.2f}%")
        st.progress(r["prob_meso"])

    #  Key physicochemical features 
    with st.expander("Feature breakdown", expanded=False):
        feat_df = r["features"]

        # 9 interpretable physicochemical features as a tidy table
        phys_cols = [
            "gravy", "aliphatic_index", "instability_index",
            "isoelectric_point", "molecular_weight", "aromaticity",
            "helix_fraction", "turn_fraction", "sheet_fraction",
        ]
        phys_vals = (
            feat_df[phys_cols]
            .T
            .rename(columns={0: "Value"})
        )
        phys_vals["Value"] = phys_vals["Value"].round(4)

        st.markdown("**Physicochemical features**")
        st.dataframe(phys_vals, use_container_width=True)

        # Amino acid composition bar chart
        aa_cols = [c for c in feat_df.columns if c.startswith("AA_")]
        aa_df   = feat_df[aa_cols].T.rename(columns={0: "Fraction"})
        aa_df.index = [c.replace("AA_", "") for c in aa_df.index]
        aa_df["Fraction"] = aa_df["Fraction"].round(4)

        st.markdown("**Amino acid composition**")
        st.bar_chart(aa_df["Fraction"])


# PAGEEEEE

# Header
st.title("Protein Thermostability Predictor")
st.markdown(
    "Classify a protein as **Thermophilic** (heat-stable) or "
    "**Mesophilic** (normal temperature) from its amino acid sequence. "
    "Uses Histogram-based Gradient Boosting trained on 16,500 UniProt-reviewed proteins with 429 sequence features."
)
st.divider()

# Load model
model, scaler, selector, all_features, missing = load_model()

if missing:
    st.error(
        "**Model not found.** Run training first:\n\n"
        "```bash\npython src/train.py\n```"
    )
    st.stop()

#  Sidebar: model info 
with st.sidebar:
    st.header("Model Info")
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        st.metric("Test Accuracy",   f"{m['test_accuracy']*100:.1f}%")
        st.metric("Test F1",         f"{m['test_f1']*100:.1f}%")
        st.metric("Test ROC-AUC",    f"{m['test_roc_auc']:.3f}")
        st.metric("CV F1 (5-fold)",  f"{m['cv_f1_mean']*100:.1f}% ± {m['cv_f1_std']*100:.1f}%")
        st.caption(f"Dataset: {m['dataset_size']:,} proteins total")
        st.caption(f"Train: {m['train_size']:,}  |  Test: {m['test_size']:,}")
        st.caption(f"Features: top {m['n_features_selected']} of {m['n_features_input']} (AA + dipeptide + physicochemical)")
    st.divider()
    st.markdown(
        "**Data source:** [UniProt Swiss-Prot](https://www.uniprot.org/)  \n"

        "**Features:** top 80 of 429 sequence-derived properties"
    )

#  Tabs 
tab1, tab2 = st.tabs(["Paste Sequence", "Upload FASTA"])

#  Tab 1: single sequence input 
with tab1:
    st.markdown("Paste a single amino acid sequence (50–5,000 residues, standard AA only).")

    example_seq = (
        "MEAMLPLFEPKGRVLLVDGHHLAYRTFHALKGLTTSRGEPVQAVYGFAKSLLKALKEDGY"
        "KAVFVVFDAKAPSFRHEAYEAYKAGRAPTPEDFPRQLALIKELVDLLGFTRLEVPGYEADD"
        "VLATLAKKAEKEGYEVRILTADRDLYQLVSDRVAVLHPEGHLITPEWLWEKYGLRPEQWVD"
        "FRALVGDPSDNIPGVKGIGEKTALKLLEEWGSLENLLKNLDRVKPENVREKIKAHLEDLRL"
        "SLELSRVRTDLPLEVDLAQGREPDREGLRAFLERLEFGSLLHEFGLLEAPAPLEEAPWPPP"
        "EGAFVGFVLSRPEPMWAELKALAACRGRVHGRPDDLVAVLGRLRGLEVPAGRPALEFAYEL"
        "GRLEEARGLLALPLAAEVVAGSVARVLRAGADGRLEPAAVLLREALEAAPPEAGPWLEAVRA"
        "GPDRALVLLPPDLPLEPLAVPVLLAWVDAERPVLGRGRVVVPGTARAAVDAAARAAVLREAGR"
    )

    col_btn, _ = st.columns([1, 4])
    if col_btn.button("Load Taq polymerase example"):
        st.session_state["seq_input"] = example_seq

    seq_input = st.text_area(
        "Amino acid sequence",
        height=180,
        placeholder="MKTAYIAKQRQISFVKSHFSRQ...",
        key="seq_input",
    )

    if st.button("Predict", type="primary", key="btn_single"):
        raw = seq_input.strip()
        if not raw:
            st.warning("Please enter a sequence.")
        else:
            # Accept both raw sequences and FASTA-formatted input
            if raw.startswith(">"):
                records = list(SeqIO.parse(StringIO(raw), "fasta"))
                if not records:
                    st.error("Could not parse FASTA input. Check the format.")
                    st.stop()
                clean = str(records[0].seq).upper().strip()
            else:
                clean = raw.replace(" ", "").replace("\n", "").replace("\r", "").upper()
            clean, removed = clean_sequence(clean)
            if removed:
                st.warning(f"Removed non-standard characters: `{''.join(sorted(removed))}`")
            with st.spinner("Running prediction..."):
                result = predict(clean, model, scaler, selector, all_features)
            render_result(result, clean)

#  Tab 2: FASTA upload 
with tab2:
    st.markdown(
        "Upload a FASTA file. Each sequence will be predicted individually. "
        "Multi-sequence files are supported."
    )

    uploaded = st.file_uploader("Choose a FASTA file", type=["fasta", "fa", "txt"])

    if uploaded is not None:
        raw_bytes = uploaded.read()
        raw_text  = raw_bytes.decode("utf-8", errors="ignore").encode("ascii", errors="ignore").decode("ascii")
        records   = list(SeqIO.parse(StringIO(raw_text), "fasta"))

        if not records:
            st.error("No sequences found. Make sure the file is in FASTA format.")
        else:
            st.info(f"Found **{len(records)}** sequence(s) in `{uploaded.name}`.")

            if st.button("Predict All", type="primary", key="btn_fasta"):
                results_rows = []

                progress = st.progress(0, text="Predicting...")
                for i, rec in enumerate(records):
                    seq, removed = clean_sequence(str(rec.seq).upper().strip())
                    r   = predict(seq, model, scaler, selector, all_features)

                    st.markdown(f"#### {rec.id}")
                    render_result(r, seq, seq_id=rec.description[:80])

                    results_rows.append({
                        "id":               rec.id,
                        "length":           len(seq),
                        "prediction":       r.get("prediction", "ERROR"),
                        "confidence":       f"{max(r.get('prob_meso',0), r.get('prob_thermo',0))*100:.1f}%",
                        "prob_thermophilic": round(r.get("prob_thermo", 0), 4),
                        "prob_mesophilic":   round(r.get("prob_meso",   0), 4),
                        "error":            r.get("error", ""),
                    })
                    progress.progress((i + 1) / len(records),
                                      text=f"Predicted {i+1}/{len(records)}")

                progress.empty()

                # Summary
                st.divider()
                summary_df = pd.DataFrame(results_rows)
                valid = summary_df[summary_df["prediction"] != "ERROR"]
                n_t = (valid["prediction"] == "Thermophilic").sum()
                n_m = (valid["prediction"] == "Mesophilic").sum()

                c1, c2, c3 = st.columns(3)
                c1.metric("Total sequences",  len(summary_df))
                c2.metric(" Thermophilic",  n_t)
                c3.metric(" Mesophilic",    n_m)

                # Download CSV
                csv = summary_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name="thermostability_predictions.csv",
                    mime="text/csv",
                )
