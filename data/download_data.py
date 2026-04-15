"""
download_data.py — Download thermophilic and mesophilic protein
sequences from the UniProt REST API and save a balanced, filtered
dataset to data/proteins.csv.

Run from the project root:
    python data/download_data.py
"""

import random
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from Bio import SeqIO

# Reproducibility
random.seed(42)

# Constants
# Standard 20 amino acids in single-letter IUPAC code
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Sequence length bounds (too short = unreliable features,
# too long = likely multi-domain complexes that add noise)
MIN_LEN = 50
MAX_LEN = 5000   # raised from 2000 — covers large multi-domain proteins

# Where to save the output CSV
DATA_DIR = Path(__file__).parent          # same folder as this script
OUTPUT_PATH = DATA_DIR / "proteins.csv"

# UniProt REST API URLs
# reviewed:true  → Swiss-Prot only (manually curated, highest quality)
# format=fasta   → standard FASTA text format
# size=800       → request up to 800 sequences per class
THERMO_URL = (
    "https://rest.uniprot.org/uniprotkb/stream?"
    "query=%28"
    # 27 confirmed hyperthermophiles — ALL Topt >= 75°C
    # Removed borderline organisms (59–70°C) that blurred the class boundary
    "taxonomy_id%3A300852+OR+taxonomy_id%3A262724"   # T. thermophilus HB8/HB27 (75C)
    "+OR+taxonomy_id%3A273057+OR+taxonomy_id%3A2261"  # S. solfataricus / P. furiosus (80-100C)
    "+OR+taxonomy_id%3A2234+OR+taxonomy_id%3A243274"  # A. fulgidus / T. maritima (80-83C)
    "+OR+taxonomy_id%3A69014+OR+taxonomy_id%3A272557" # T. kodakarensis / A. pernix (85-90C)
    "+OR+taxonomy_id%3A178306+OR+taxonomy_id%3A273068"# P. aerophilum / T. tengcongensis (75-100C)
    "+OR+taxonomy_id%3A190192+OR+taxonomy_id%3A63363" # M. kandleri / A. aeolicus (95-98C)
    "+OR+taxonomy_id%3A33015+OR+taxonomy_id%3A70601"  # S. acidocaldarius / P. horikoshii (75-98C)
    "+OR+taxonomy_id%3A228908+OR+taxonomy_id%3A2285"  # N. equitans / M. jannaschii (85-90C)
    "+OR+taxonomy_id%3A35543"                         # Pyrococcus abyssi GE5 (103C)
    # 10 additional strict hyperthermophiles (Topt >= 75°C)
    "+OR+taxonomy_id%3A111955"                        # Sulfolobus tokodaii str. 7 (80C)
    "+OR+taxonomy_id%3A2277"                          # Pyrobaculum islandicum DSM 4184 (100C)
    "+OR+taxonomy_id%3A2282"                          # Thermoproteus tenax Kra 1 (86C)
    "+OR+taxonomy_id%3A2224"                          # Methanothermus fervidus DSM 2088 (83C)
    "+OR+taxonomy_id%3A36419"                         # Desulfurococcus mobilis DSM 2161 (85C)
    "+OR+taxonomy_id%3A2203"                          # Staphylothermus marinus F1 (92C)
    "+OR+taxonomy_id%3A54033"                         # Hyperthermus butylicus DSM 5456 (101C)
    "+OR+taxonomy_id%3A453591"                        # Ignicoccus hospitalis KIN4/I (90C)
    "+OR+taxonomy_id%3A593117"                        # Thermococcus gammatolerans EJ3 (88C)
    "+OR+taxonomy_id%3A110163"                        # Metallosphaera sedula DSM 5348 (75C)
    "%29+AND+reviewed%3Atrue"
    "&format=fasta"
)
# Thermophilic organisms — strict cutoff Topt >= 75°C.
# Removed 7 borderline organisms (59-70°C) that caused label noise.
# 27 total organisms:
#   300852 = Thermus thermophilus HB8            (Topt 75C)
#   262724 = Thermus thermophilus HB27           (Topt 75C)
#   273057 = Sulfolobus solfataricus P2          (Topt 80C)
#   2261   = Pyrococcus furiosus DSM 3638        (Topt 100C)
#   2234   = Archaeoglobus fulgidus DSM 4304     (Topt 83C)
#   243274 = Thermotoga maritima MSB8            (Topt 80C)
#   69014  = Thermococcus kodakarensis KOD1      (Topt 85C)
#   272557 = Aeropyrum pernix K1                 (Topt 90C)
#   178306 = Pyrobaculum aerophilum IM2          (Topt 100C)
#   273068 = Thermoanaerobacter tengcongensis     (Topt 75C)
#   190192 = Methanopyrus kandleri AV19          (Topt 98C)
#   63363  = Aquifex aeolicus VF5               (Topt 95C)
#   33015  = Sulfolobus acidocaldarius DSM 639   (Topt 75C)
#   70601  = Pyrococcus horikoshii OT3           (Topt 98C)
#   228908 = Nanoarchaeum equitans Kin4-M        (Topt 90C)
#   2285   = Methanocaldococcus jannaschii JAL-1  (Topt 85C)
#   35543  = Pyrococcus abyssi GE5              (Topt 103C)
#   111955 = Sulfolobus tokodaii str. 7          (Topt 80C)  NEW
#   2277   = Pyrobaculum islandicum DSM 4184     (Topt 100C) NEW
#   2282   = Thermoproteus tenax Kra 1           (Topt 86C)  NEW
#   2224   = Methanothermus fervidus DSM 2088    (Topt 83C)  NEW
#   36419  = Desulfurococcus mobilis DSM 2161    (Topt 85C)  NEW
#   2203   = Staphylothermus marinus F1          (Topt 92C)  NEW
#   54033  = Hyperthermus butylicus DSM 5456     (Topt 101C) NEW
#   453591 = Ignicoccus hospitalis KIN4/I        (Topt 90C)  NEW
#   593117 = Thermococcus gammatolerans EJ3      (Topt 88C)  NEW
#   110163 = Metallosphaera sedula DSM 5348      (Topt 75C)  NEW
# REMOVED (borderline, caused noise):
#   187420 = Methanothermobacter thermautotroph. (Topt 65C)
#   273075 = Thermoplasma acidophilum            (Topt 59C)
#   273116 = Thermoplasma volcanium              (Topt 60C)
#   1515   = Clostridium thermocellum            (Topt 60C)
#   235909 = Geobacillus kaustophilus            (Topt 60C)
#   1422   = Geobacillus stearothermophilus      (Topt 65C)
#   195522 = Carboxydothermus hydrogenoformans   (Topt 70C)

MESO_URL = (
    "https://rest.uniprot.org/uniprotkb/stream?"
    "query=%28"
    # Original 5 organisms
    "taxonomy_id%3A562+OR+taxonomy_id%3A9606"
    "+OR+taxonomy_id%3A559292+OR+taxonomy_id%3A10090"
    "+OR+taxonomy_id%3A1423"
    # 15 existing mesophilic organisms across bacteria, fungi, plants, animals
    "+OR+taxonomy_id%3A83332"   # Mycobacterium tuberculosis H37Rv (Topt 37C)
    "+OR+taxonomy_id%3A208964"  # Pseudomonas aeruginosa PAO1     (Topt 37C)
    "+OR+taxonomy_id%3A99287"   # Salmonella typhimurium LT2      (Topt 37C)
    "+OR+taxonomy_id%3A85962"   # Helicobacter pylori 26695       (Topt 37C)
    "+OR+taxonomy_id%3A169963"  # Listeria monocytogenes EGD-e    (Topt 37C)
    "+OR+taxonomy_id%3A7227"    # Drosophila melanogaster         (Topt 25C)
    "+OR+taxonomy_id%3A3702"    # Arabidopsis thaliana            (Topt 22C)
    "+OR+taxonomy_id%3A6239"    # Caenorhabditis elegans          (Topt 20C)
    "+OR+taxonomy_id%3A7955"    # Danio rerio (zebrafish)         (Topt 28C)
    "+OR+taxonomy_id%3A9031"    # Gallus gallus (chicken)         (Topt 41C)
    "+OR+taxonomy_id%3A39947"   # Oryza sativa (rice)             (Topt 30C)
    "+OR+taxonomy_id%3A284812"  # Schizosaccharomyces pombe 972   (Topt 30C)
    "+OR+taxonomy_id%3A1280"    # Staphylococcus aureus MRSA252   (Topt 37C)
    "+OR+taxonomy_id%3A71421"   # Haemophilus influenzae Rd KW20  (Topt 37C)
    "+OR+taxonomy_id%3A243277"  # Vibrio cholerae O1 El Tor N169  (Topt 37C)
    # 5 additional mesophilic organisms for more kingdom diversity
    "+OR+taxonomy_id%3A9913"    # Bos taurus (cow)                (Topt 37C)
    "+OR+taxonomy_id%3A10116"   # Rattus norvegicus (rat)         (Topt 37C)
    "+OR+taxonomy_id%3A9823"    # Sus scrofa (pig)                (Topt 37C)
    "+OR+taxonomy_id%3A100226"  # Streptomyces coelicolor A3(2)   (Topt 28C)
    "+OR+taxonomy_id%3A5141"    # Neurospora crassa OR74A         (Topt 25C)
    "%29+AND+reviewed%3Atrue"
    "&format=fasta"
)
# Mesophilic organisms — all experimentally confirmed < 45 °C optimum.
# Original 5 + 15 + 5 new = 25 total organisms spanning all kingdoms of life.
#   562    = Escherichia coli K-12              (Topt 37C)
#   9606   = Homo sapiens                       (Topt 37C)
#   559292 = Saccharomyces cerevisiae S288C     (Topt 30C)
#   10090  = Mus musculus                       (Topt 37C)
#   1423   = Bacillus subtilis 168              (Topt 37C)
#   83332  = Mycobacterium tuberculosis H37Rv   (Topt 37C)
#   208964 = Pseudomonas aeruginosa PAO1        (Topt 37C)
#   99287  = Salmonella typhimurium LT2         (Topt 37C)
#   85962  = Helicobacter pylori 26695          (Topt 37C)
#   169963 = Listeria monocytogenes EGD-e       (Topt 37C)
#   7227   = Drosophila melanogaster            (Topt 25C)
#   3702   = Arabidopsis thaliana               (Topt 22C)
#   6239   = Caenorhabditis elegans             (Topt 20C)
#   7955   = Danio rerio                        (Topt 28C)
#   9031   = Gallus gallus                      (Topt 41C)
#   39947  = Oryza sativa Japonica              (Topt 30C)
#   284812 = Schizosaccharomyces pombe          (Topt 30C)
#   1280   = Staphylococcus aureus              (Topt 37C)
#   71421  = Haemophilus influenzae             (Topt 37C)
#   243277 = Vibrio cholerae                    (Topt 37C)
#   9913   = Bos taurus (cow)                   (Topt 37C)  NEW
#   10116  = Rattus norvegicus (rat)            (Topt 37C)  NEW
#   9823   = Sus scrofa (pig)                   (Topt 37C)  NEW
#   100226 = Streptomyces coelicolor A3(2)      (Topt 28C)  NEW
#   5141   = Neurospora crassa OR74A            (Topt 25C)  NEW


# Helper: download one FASTA file from a URL
def download_fasta(url: str, label_name: str) -> str:
    """
    Download raw FASTA text from a URL.

    Parameters
    ----------
    url        : full UniProt REST API URL
    label_name : human-readable name used only for print messages

    Returns
    -------
    str : raw FASTA text (may be empty if server returns nothing)
    """
    print(f"\nDownloading {label_name} sequences from UniProt ...")
    print(f"  URL: {url}")

    try:
        # timeout=60 seconds — UniProt can be slow for large queries
        response = requests.get(url, timeout=60)
        response.raise_for_status()   # raises HTTPError for 4xx/5xx
    except requests.exceptions.Timeout:
        print(f"ERROR: Request timed out for {label_name}.")
        print("Check your internet connection and try again.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: Server returned an error for {label_name}: {e}")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to UniProt for {label_name}.")
        print("Check your internet connection.")
        sys.exit(1)

    text = response.text
    print(f"  Received {len(text):,} bytes.")

    if not text.strip():
        print(f"WARNING: UniProt returned an empty response for "
              f"{label_name}. Continuing with 0 sequences.")

    return text


# Helper: parse, filter, and deduplicate a FASTA string
def parse_and_filter(fasta_text: str, label_name: str) -> list:
    """
    Parse a raw FASTA string and apply quality filters.

    Filters applied (in order):
    1. Length: keep only sequences between MIN_LEN and MAX_LEN
    2. Standard residues: discard sequences with non-standard
       characters (X, B, Z, U, etc.)
    3. Deduplication: remove identical sequences

    Returns a list of (id, sequence) tuples.
    """
    # Step 1: Parse FASTA using BioPython
    # StringIO wraps the raw text so SeqIO can treat it like a file
    records = list(SeqIO.parse(StringIO(fasta_text), "fasta"))
    n_downloaded = len(records)
    print(f"  Parsed {n_downloaded} raw sequences for {label_name}.")

    if n_downloaded == 0:
        print(f"WARNING: No sequences parsed for {label_name}.")
        return []

    kept = []
    n_short_long = 0    # removed for length
    n_nonstandard = 0   # removed for non-standard amino acids
    seen_sequences = set()
    n_duplicate = 0

    for record in records:
        seq = str(record.seq).upper()

        # Step 2: Filter by length
        if not (MIN_LEN <= len(seq) <= MAX_LEN):
            n_short_long += 1
            continue

        # Step 3: Remove sequences with non-standard characters
        # set(seq) gives unique characters; they must all be in
        # STANDARD_AA for the sequence to pass this filter
        if not set(seq).issubset(STANDARD_AA):
            n_nonstandard += 1
            continue

        # Step 4: Deduplicate — skip if we have seen this exact sequence
        if seq in seen_sequences:
            n_duplicate += 1
            continue

        seen_sequences.add(seq)
        kept.append((record.id, seq))

    n_removed = n_short_long + n_nonstandard + n_duplicate
    print(f"  Removed {n_short_long} for length, "
          f"{n_nonstandard} for non-standard AA, "
          f"{n_duplicate} duplicates.")
    print(f"  Kept {len(kept)} sequences after filtering.")

    return kept


# Main pipeline
def main():
    print("=" * 60)
    print("Protein Thermostability — Data Download Script")
    print("=" * 60)

    # Step 1: Download thermophilic sequences
    thermo_fasta = download_fasta(THERMO_URL, "thermophilic")

    # Step 2: Download mesophilic sequences
    meso_fasta = download_fasta(MESO_URL, "mesophilic")

    # Step 3: Parse and filter both classes
    thermo_seqs = parse_and_filter(thermo_fasta, "thermophilic")
    meso_seqs   = parse_and_filter(meso_fasta,   "mesophilic")

    # Guard: both classes must have at least some sequences
    if len(thermo_seqs) == 0 or len(meso_seqs) == 0:
        print("\nERROR: One or both classes have zero sequences after "
              "filtering. Cannot build a balanced dataset.")
        print("Possible causes: UniProt API unavailable, empty response,")
        print("or all sequences were filtered out.")
        sys.exit(1)

    # Step 4: Balance classes
    # Take the smaller class size from both, so label counts are equal.
    # Equal class balance produces the best per-class F1.
    n_balanced = min(len(thermo_seqs), len(meso_seqs))

    if n_balanced < len(thermo_seqs):
        print(f"\nBalancing: randomly sampling {n_balanced} thermophilic "
              f"sequences (from {len(thermo_seqs)}).")
        thermo_seqs = random.sample(thermo_seqs, n_balanced)

    if n_balanced < len(meso_seqs):
        print(f"Balancing: randomly sampling {n_balanced} mesophilic "
              f"sequences (from {len(meso_seqs)}).")
        meso_seqs = random.sample(meso_seqs, n_balanced)

    # Step 5: Assign labels and combine into one list
    # thermophilic = 1, mesophilic = 0
    rows = []
    for seq_id, seq in thermo_seqs:
        rows.append({"id": seq_id, "sequence": seq, "label": 1})
    for seq_id, seq in meso_seqs:
        rows.append({"id": seq_id, "sequence": seq, "label": 0})

    # Step 6: Create DataFrame and shuffle rows
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Step 7: Save to data/proteins.csv
    # DATA_DIR already points to the data/ folder
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    # Step 8: Print final summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE — Summary")
    print("=" * 60)
    print(f"  Total proteins saved : {len(df)}")
    print(f"  Thermophilic (1)     : {(df['label'] == 1).sum()}")
    print(f"  Mesophilic   (0)     : {(df['label'] == 0).sum()}")
    print(f"  Saved to             : {OUTPUT_PATH.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
