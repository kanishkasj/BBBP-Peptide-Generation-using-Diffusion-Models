"""
generation.py  —  Generate BBBP peptide datasets with random lengths 5-20.
               —  Uses diffusion model + SVM (AAIndex top-17) + B3PPS.
Uses the REAL diffusion model architecture from diffusion_model.py.
"""
import ast
import os, pickle
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import EsmTokenizer, EsmForSequenceClassification
from aaindex import aaindex1
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from diffusion_model import PeptideDiffusionModel, DiscreteDiffusion, load_diffusion_model
from config import DEVICE, MAX_SEQ_LEN, AA_TO_IDX, AMINO_ACIDS

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
B3PPS_PATH  = r"E:\COLLEGE\AMRITA VISHWA VIDYPEETHAM _ NEW\New folder\DD\diffusion_bbbp\outputs\b3pps\best_model5"
OUTPUT_DIR  = r"E:\COLLEGE\AMRITA VISHWA VIDYPEETHAM _ NEW\New folder\DD\diffusion_bbbp\outputs\generated"
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "prob_plots")
PROJECT_DIR = os.path.dirname(__file__)

SVM_TOP17_FEATURES_PATH = os.path.join(PROJECT_DIR, "svm_fixed_top17_features.csv")
SVM_PARAMS_SOURCE_PATH  = os.path.join(PROJECT_DIR, "svm_10_runs_results.csv")
SVM_TRAIN_CSV_PATH      = os.path.join(OUTPUT_DIR, "bbbp_1000_dual.csv")
SVM_MODEL_CACHE_PATH    = os.path.join(OUTPUT_DIR, "svm_top17_model.pkl")
SVM_SCALER_CACHE_PATH   = os.path.join(OUTPUT_DIR, "svm_top17_scaler.pkl")
SVM_FEATURES_CACHE_PATH = os.path.join(OUTPUT_DIR, "svm_top17_feature_ids.pkl")

# ── Toxicity classifier paths ─────────────────────────────────────────────────
TOXICITY_TOP17_FEATURES_PATH = os.path.join(PROJECT_DIR, "toxicity_svm_fixed_top17_features.csv")
TOXICITY_TRAIN_CSV_PATH      = os.path.join(PROJECT_DIR, "toxicity_train_aaindex_encoded.csv")
TOXICITY_TEST_CSV_PATH       = os.path.join(PROJECT_DIR, "toxicity_test_aaindex_encoded.csv")
TOXICITY_MODEL_CACHE_PATH    = os.path.join(OUTPUT_DIR, "toxicity_svm_top17_model.pkl")
TOXICITY_SCALER_CACHE_PATH   = os.path.join(OUTPUT_DIR, "toxicity_svm_top17_scaler.pkl")
TOXICITY_FEATURES_CACHE_PATH = os.path.join(OUTPUT_DIR, "toxicity_svm_top17_feature_ids.pkl")

# ── Toxicity threshold ────────────────────────────────────────────────────────
TOXICITY_THRESH = 0.5  # Peptides with toxicity_prob >= this are filtered out

IDX_TO_AA        = {i: aa for aa, i in AA_TO_IDX.items()}
VOCAB_SIZE       = len(AMINO_ACIDS)          # 20
MIN_LEN, MAX_LEN = 5, 20
TARGET_TOTALS    = [1000, 5000]
POSITIVE_RATIO   = 0.70

# ── Classifier thresholds ─────────────────────────────────────────────────────
INT_THRESH  = 0.5
B3P_THRESH  = 0.5

# For label=0: peptide must score BELOW both thresholds (truly negative)
INT_NEG_THRESH = 0.35      # internal_prob  < this  →  confident negative
B3P_NEG_THRESH = 0.35      # b3pps_prob     < this  →  confident negative

CLF_LOGIC   = "or"         # label=1 logic: "or"=either passes | "and"=both pass
TRIMER_MAX_LEN = 12
BATCH_CLF      = 64
MAX_ROUNDS     = 60


# ─────────────────────────────────────────────────────────────────────────────
# LENGTH QUOTAS
# ─────────────────────────────────────────────────────────────────────────────
def get_random_length_quota(total: int, seed: int) -> dict:
    """
    Randomly assign each peptide a length in [5, 20] and return per-length counts.
    """
    rng = np.random.default_rng(seed)
    lengths = rng.integers(MIN_LEN, MAX_LEN + 1, size=total)
    quota = {l: int(np.sum(lengths == l)) for l in range(MIN_LEN, MAX_LEN + 1)}
    assert sum(quota.values()) == total, f"Quota sum {sum(quota.values())} ≠ {total}"
    return quota


def _pick_aaindex_features(target_n: int = 86) -> list[str]:
    selected = []
    standard_aa = list("ACDEFGHIKLMNPQRSTVWY")
    for fid in aaindex1.record_codes():
        try:
            rec = aaindex1[fid]
            vals = np.asarray([float(rec.values[aa]) for aa in standard_aa], dtype=float)
            if np.isfinite(vals).all():
                selected.append(fid)
        except Exception:
            continue
        if len(selected) >= target_n:
            break
    return selected


def aaindex_encode_sequence(seq: str, feature_ids: list[str]) -> np.ndarray | None:
    residues = [aa for aa in str(seq).upper() if aa in AA_TO_IDX]
    if not residues:
        return None

    values = []
    for fid in feature_ids:
        try:
            rec = aaindex1[fid]
            values.append(float(np.mean([float(rec.values[aa]) for aa in residues])))
        except Exception:
            return None
    return np.asarray(values, dtype=float)


def load_top17_feature_ids() -> list[str]:
    if os.path.exists(SVM_TOP17_FEATURES_PATH):
        df = pd.read_csv(SVM_TOP17_FEATURES_PATH)
        if "Feature" in df.columns and not df.empty:
            return df["Feature"].head(17).astype(str).tolist()
    return _pick_aaindex_features(17)


def load_default_svm_params() -> dict:
    default_params = {"C": 10, "kernel": "rbf", "gamma": "scale"}
    if os.path.exists(SVM_PARAMS_SOURCE_PATH):
        try:
            df = pd.read_csv(SVM_PARAMS_SOURCE_PATH)
            if "BestParams" in df.columns and len(df) > 0:
                parsed = ast.literal_eval(str(df.loc[0, "BestParams"]))
                if isinstance(parsed, dict):
                    return parsed
        except Exception:
            pass
    return default_params


def load_or_train_svm_classifier() -> tuple[object, StandardScaler, list[str]]:
    if (
        os.path.exists(SVM_MODEL_CACHE_PATH)
        and os.path.exists(SVM_SCALER_CACHE_PATH)
        and os.path.exists(SVM_FEATURES_CACHE_PATH)
    ):
        with open(SVM_MODEL_CACHE_PATH, "rb") as f:
            svm_model = pickle.load(f)
        with open(SVM_SCALER_CACHE_PATH, "rb") as f:
            svm_scaler = pickle.load(f)
        with open(SVM_FEATURES_CACHE_PATH, "rb") as f:
            top17_features = pickle.load(f)
        print("[✓] Loaded cached SVM top-17 classifier")
        return svm_model, svm_scaler, top17_features

    if not os.path.exists(SVM_TRAIN_CSV_PATH):
        raise FileNotFoundError(
            f"SVM training CSV not found: {SVM_TRAIN_CSV_PATH}. "
            "Generate bbbp_1000_dual.csv first or provide this file."
        )

    df = pd.read_csv(SVM_TRAIN_CSV_PATH).copy()
    if "seq" not in df.columns or "label" not in df.columns:
        raise ValueError("SVM training CSV must contain `seq` and `label` columns")

    top17_features = load_top17_feature_ids()
    X_rows, y_rows = [], []
    for _, row in df.iterrows():
        vec = aaindex_encode_sequence(row["seq"], top17_features)
        if vec is None:
            continue
        X_rows.append(vec)
        y_rows.append(int(row["label"]))

    if not X_rows:
        raise ValueError("No valid rows found to train SVM classifier")

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=int)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    svm_params = load_default_svm_params()
    svm_model = SVC(probability=True, random_state=42, **svm_params)
    svm_model.fit(Xs, y)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SVM_MODEL_CACHE_PATH, "wb") as f:
        pickle.dump(svm_model, f)
    with open(SVM_SCALER_CACHE_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(SVM_FEATURES_CACHE_PATH, "wb") as f:
        pickle.dump(top17_features, f)

    print("[✓] Trained and cached SVM top-17 classifier")
    return svm_model, scaler, top17_features


def load_toxicity_svm_top17_features() -> list[str]:
    """Load top-17 toxicity features from CSV file."""
    if os.path.exists(TOXICITY_TOP17_FEATURES_PATH):
        df = pd.read_csv(TOXICITY_TOP17_FEATURES_PATH)
        if "Feature" in df.columns and not df.empty:
            return df["Feature"].head(17).astype(str).tolist()
    return _pick_aaindex_features(17)


def load_or_train_toxicity_svm_classifier() -> tuple[object, StandardScaler, list[str]]:
    """Load or train toxicity SVM classifier using top-17 AAIndex features."""
    if (
        os.path.exists(TOXICITY_MODEL_CACHE_PATH)
        and os.path.exists(TOXICITY_SCALER_CACHE_PATH)
        and os.path.exists(TOXICITY_FEATURES_CACHE_PATH)
    ):
        with open(TOXICITY_MODEL_CACHE_PATH, "rb") as f:
            tox_model = pickle.load(f)
        with open(TOXICITY_SCALER_CACHE_PATH, "rb") as f:
            tox_scaler = pickle.load(f)
        with open(TOXICITY_FEATURES_CACHE_PATH, "rb") as f:
            tox_features = pickle.load(f)
        print("[✓] Loaded cached toxicity SVM top-17 classifier")
        return tox_model, tox_scaler, tox_features

    # Try to load from encoded CSV files (from toxicity_aaindex_classification.ipynb)
    if os.path.exists(TOXICITY_TRAIN_CSV_PATH):
        print(f"[*] Loading toxicity training data from {TOXICITY_TRAIN_CSV_PATH}")
        df_train = pd.read_csv(TOXICITY_TRAIN_CSV_PATH)
        if "label" not in df_train.columns:
            raise ValueError("Toxicity training CSV must contain `label` column")
        
        # Extract feature columns (everything except 'label')
        feature_cols = [c for c in df_train.columns if c != "label"]
        X_train = df_train[feature_cols].values.astype(float)
        y_train = df_train["label"].values.astype(int)
        
        print(f"[*] Toxicity train shape: {X_train.shape}, labels: {np.unique(y_train, return_counts=True)}")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train SVM with best params from toxicity notebook
        tox_model = SVC(probability=True, C=10, kernel='rbf', gamma='scale', random_state=42)
        tox_model.fit(X_train_scaled, y_train)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(TOXICITY_MODEL_CACHE_PATH, "wb") as f:
            pickle.dump(tox_model, f)
        with open(TOXICITY_SCALER_CACHE_PATH, "wb") as f:
            pickle.dump(scaler, f)
        with open(TOXICITY_FEATURES_CACHE_PATH, "wb") as f:
            pickle.dump(feature_cols, f)
        
        print("[✓] Trained and cached toxicity SVM classifier")
        return tox_model, scaler, feature_cols
    else:
        raise FileNotFoundError(
            f"Toxicity training data not found at {TOXICITY_TRAIN_CSV_PATH}. "
            "Run toxicity_aaindex_classification.ipynb first to generate this file."
        )


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLING
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def diffusion_sample(model, diffusion, length, num_samples, device):
    """
    Reverse discrete diffusion: t=T → predict x0 → re-noise to t-1 → … → t=0
    Returns a list of decoded amino acid strings.
    """
    B = num_samples
    T = diffusion.timesteps

    mask      = torch.zeros(B, MAX_SEQ_LEN, dtype=torch.bool, device=device)
    mask[:, length:] = True
    lengths_t = torch.full((B,), length, dtype=torch.long, device=device)
    labels_t  = torch.ones((B,),          dtype=torch.long, device=device)

    x = torch.zeros(B, MAX_SEQ_LEN, dtype=torch.long, device=device)
    x[:, :length] = torch.randint(1, VOCAB_SIZE + 1, (B, length), device=device)

    for t_val in range(T - 1, -1, -1):
        t_tensor = torch.full((B,), t_val, dtype=torch.long, device=device)
        logits   = model(x, t_tensor, lengths_t, labels_t, mask)
        logits   = logits[:, :length, :]

        x0_pred  = torch.multinomial(
            F.softmax(logits.contiguous().view(B * length, 20), dim=-1), 1
        ).view(B, length)

        if t_val == 0:
            x[:, :length] = logits.argmax(dim=-1) + 1
            break

        alpha_prev   = diffusion.alpha_cumprod[t_val - 1].to(device)
        corrupt_prob = 1.0 - alpha_prev
        keep_mask    = torch.rand(B, length, device=device) >= corrupt_prob
        random_toks  = torch.randint(0, VOCAB_SIZE, (B, length), device=device)
        x_prev       = torch.where(keep_mask, x0_pred, random_toks)
        x[:, :length] = x_prev + 1

    seqs = []
    for b in range(B):
        seq = "".join(IDX_TO_AA.get((x[b, p] - 1).item(), "") for p in range(length))
        seq = "".join(c for c in seq if c in AA_TO_IDX)
        if seq:
            seqs.append(seq)
    return seqs


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL FILTER  (same as before)
# ─────────────────────────────────────────────────────────────────────────────
MAX_SINGLE_AA_FRAC = 0.4
MIN_UNIQUE_AA      = 4
MAX_NGRAM_FRAC     = 0.35

def is_valid(seq, length):
    if not isinstance(seq, str) or len(seq) != length: return False
    if any(c not in AA_TO_IDX for c in seq):           return False
    if seq.count("CK") > 1:                            return False

    counts = Counter(seq)
    if counts.most_common(1)[0][1] / length > MAX_SINGLE_AA_FRAC: return False
    if len(counts) < MIN_UNIQUE_AA:                                return False

    trimers = [seq[i:i+3] for i in range(length - 2)]
    tri_counts = Counter(trimers)
    most_common_tri_frac = tri_counts.most_common(1)[0][1] / len(trimers)
    if length <= TRIMER_MAX_LEN:
        if most_common_tri_frac > 1 / len(trimers): return False
    else:
        if most_common_tri_frac > MAX_NGRAM_FRAC:   return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIERS
# ─────────────────────────────────────────────────────────────────────────────
def load_b3pps(path, device):
    if not os.path.exists(path):
        print(f"[!] B3PPs not found at {path}"); return None
    tok = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D", do_lower_case=False)
    mdl = EsmForSequenceClassification.from_pretrained(path).to(device).eval()
    print("[✓] B3PPs ESM classifier loaded")
    return (tok, mdl)


def score_batch(seqs, svm_model, svm_scaler, svm_feature_ids, b3pps, device, 
                 toxicity_model=None, toxicity_scaler=None, toxicity_feature_ids=None):
    """
    Returns list of dicts with 'seq', 'internal_prob', 'b3pps_prob', 'toxicity_prob' for every seq.
    (No filtering here — caller decides label=0 or label=1 based on scores.)
    """
    svm_vecs = []
    valid_idx = []
    for i, seq in enumerate(seqs):
        vec = aaindex_encode_sequence(seq, svm_feature_ids)
        if vec is not None:
            svm_vecs.append(vec)
            valid_idx.append(i)

    int_probs = np.zeros(len(seqs), dtype=float)
    if svm_vecs:
        Xs = svm_scaler.transform(np.asarray(svm_vecs, dtype=float))
        probs = svm_model.predict_proba(Xs)[:, 1]
        for j, p in enumerate(probs):
            int_probs[valid_idx[j]] = float(p)

    if b3pps:
        tok, mdl = b3pps
        enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            b3_probs = torch.sigmoid(mdl(**enc).logits)[:, 1].cpu().numpy()
    else:
        b3_probs = int_probs.copy()

    # ── Score toxicity ────────────────────────────────────────────────────
    toxicity_probs = np.zeros(len(seqs), dtype=float)
    if toxicity_model is not None and toxicity_scaler is not None and toxicity_feature_ids is not None:
        tox_vecs = []
        tox_valid_idx = []
        for i, seq in enumerate(seqs):
            # Encode sequence using toxicity feature IDs (AAIndex features)
            vec = aaindex_encode_sequence(seq, toxicity_feature_ids)
            if vec is not None:
                tox_vecs.append(vec)
                tox_valid_idx.append(i)
        
        if tox_vecs:
            X_tox_scaled = toxicity_scaler.transform(np.asarray(tox_vecs, dtype=float))
            tox_probs = toxicity_model.predict_proba(X_tox_scaled)[:, 1]
            for j, p in enumerate(tox_probs):
                toxicity_probs[tox_valid_idx[j]] = float(p)

    results = []
    for i, seq in enumerate(seqs):
        results.append({
            "seq"          : seq,
            "internal_prob": float(int_probs[i]),
            "b3pps_prob"   : float(b3_probs[i]),
            "toxicity_prob": float(toxicity_probs[i]),
        })
    return results


def is_positive(r: dict) -> bool:
    """
    Label=1 rule: passes at least one classifier (OR logic) AND is low-toxicity.
    """
    int_pass = r["internal_prob"] >= INT_THRESH
    b3p_pass = r["b3pps_prob"]    >= B3P_THRESH
    tox_pass = r.get("toxicity_prob", 0.0) < TOXICITY_THRESH  # Must be LOW toxicity
    
    if CLF_LOGIC == "and":
        clf_pass = int_pass and b3p_pass
    else:
        clf_pass = int_pass or b3p_pass
    
    return clf_pass and tox_pass


def is_negative(r: dict) -> bool:
    """
    Label=0 rule: BOTH classifiers score below their negative thresholds AND is low-toxicity.
    This ensures the peptide is genuinely BBBP-negative and not toxic.
    """
    tox_pass = r.get("toxicity_prob", 0.0) < TOXICITY_THRESH  # Must be LOW toxicity
    return (r["internal_prob"] < INT_NEG_THRESH and
            r["b3pps_prob"]    < B3P_NEG_THRESH and tox_pass)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-LABEL GENERATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def generate_for_label(
    label: int,
    quota: dict,
    diff_model, diffusion,
    svm_model, svm_scaler, svm_feature_ids, b3pps,
    toxicity_model, toxicity_scaler, toxicity_feature_ids,
    seen: set,
    device,
) -> list[dict]:
    """
    Generate peptides for one class (label=0 or label=1).

    label=1 → accept peptides that pass is_positive() and are low-toxicity
    label=0 → accept peptides that pass is_negative() and are low-toxicity

    `seen` is shared across both calls so no duplicates appear in either class.
    """
    accept_fn   = is_positive if label == 1 else is_negative
    label_name  = "POSITIVE (label=1)" if label == 1 else "NEGATIVE (label=0)"
    records     = []

    print(f"\n{'='*60}")
    print(f"  Generating {sum(quota.values())} {label_name} peptides (LOW TOXICITY)")
    print(f"{'='*60}")

    for length in range(MIN_LEN, MAX_LEN + 1):
        need, bucket, rounds = quota[length], [], 0
        print(f"\n── Length {length:2d}  (need {need}) ──────────")

        while len(bucket) < need and rounds < MAX_ROUNDS:
            rounds += 1
            raw   = diffusion_sample(diff_model, diffusion, length,
                                     max(256, need * 8), device)
            cands = [s for s in raw if s not in seen and is_valid(s, length)]

            hit = 0
            for i in range(0, len(cands), BATCH_CLF):
                if len(bucket) >= need:
                    break
                batch   = cands[i : i + BATCH_CLF]
                scored  = score_batch(
                    batch,
                    svm_model,
                    svm_scaler,
                    svm_feature_ids,
                    b3pps,
                    device,
                    toxicity_model,
                    toxicity_scaler,
                    toxicity_feature_ids,
                )
                for r in scored:
                    if len(bucket) >= need:
                        break
                    if accept_fn(r):
                        seen.add(r["seq"])
                        bucket.append({
                            **r,
                            "len"  : length,
                            "label": label,
                        })
                        hit += 1

            print(f"  r{rounds:02d}: raw={len(raw):4d}  "
                  f"struct={len(cands):4d}  accepted={hit:3d}  "
                  f"bucket={len(bucket)}/{need}")

        if len(bucket) < need:
            print(f"  ⚠ WARNING: only {len(bucket)}/{need} collected for "
                  f"length={length}, label={label}")

        records.extend(bucket[:need])

    return records


# ─────────────────────────────────────────────────────────────────────────────
# MASTER GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_all(total_target, diff_model, diffusion, svm_model, svm_scaler,
                 svm_feature_ids, b3pps, toxicity_model, toxicity_scaler, 
                 toxicity_feature_ids, device) -> pd.DataFrame:
    """
    Generate target positives + negatives for the requested total.
    Both sets share a `seen` set to guarantee no duplicates across classes.
    All generated peptides are filtered to be low-toxicity.
    """
    target_label1 = int(round(total_target * POSITIVE_RATIO))
    target_label0 = int(total_target - target_label1)
    quota1 = get_random_length_quota(target_label1, seed=42 + total_target)
    quota0 = get_random_length_quota(target_label0, seed=99 + total_target)

    seen = set()   # shared across both classes

    # ── Generate label=1 first ────────────────────────────────────────────
    pos_records = generate_for_label(
        label=1, quota=quota1,
        diff_model=diff_model, diffusion=diffusion,
        svm_model=svm_model, svm_scaler=svm_scaler,
        svm_feature_ids=svm_feature_ids, b3pps=b3pps,
        toxicity_model=toxicity_model, toxicity_scaler=toxicity_scaler,
        toxicity_feature_ids=toxicity_feature_ids,
        seen=seen, device=device,
    )

    # ── Generate label=0 ──────────────────────────────────────────────────
    neg_records = generate_for_label(
        label=0, quota=quota0,
        diff_model=diff_model, diffusion=diffusion,
        svm_model=svm_model, svm_scaler=svm_scaler,
        svm_feature_ids=svm_feature_ids, b3pps=b3pps,
        toxicity_model=toxicity_model, toxicity_scaler=toxicity_scaler,
        toxicity_feature_ids=toxicity_feature_ids,
        seen=seen, device=device,
    )

    # ── Combine & shuffle ─────────────────────────────────────────────────
    df = pd.DataFrame(pos_records + neg_records)
    df = df.drop_duplicates("seq").reset_index(drop=True)
    df = df[df.apply(lambda r: is_valid(r["seq"], int(r["len"])), axis=1)].copy()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)   # shuffle

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def plot_distributions(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    bins = np.linspace(0, 1, 21)

    def side_bars(ax, a, b, bins):
        ca, cb = "steelblue", "darkorange"
        na, _  = np.histogram(a, bins)
        nb, e  = np.histogram(b, bins)
        w  = (e[1] - e[0]) * 0.42
        cx = (e[:-1] + e[1:]) / 2
        ax.bar(cx - w/2, na, w, color=ca, alpha=0.85, label="Internal", edgecolor="white")
        ax.bar(cx + w/2, nb, w, color=cb, alpha=0.85, label="B3PPs",    edgecolor="white")

    # Per-length plots split by label
    for length in sorted(df["len"].unique()):
        sub  = df[df["len"] == length]
        pos  = sub[sub["label"] == 1]
        neg  = sub[sub["label"] == 0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        side_bars(axes[0], pos["internal_prob"], pos["b3pps_prob"], bins)
        axes[0].set(title=f"Len {length} — Label=1 (n={len(pos)})",
                    xlabel="Probability", ylabel="Count", xlim=(0, 1))
        axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

        if len(neg) > 0:
            side_bars(axes[1], neg["internal_prob"], neg["b3pps_prob"], bins)
            axes[1].set(title=f"Len {length} — Label=0 (n={len(neg)})",
                        xlabel="Probability", ylabel="Count", xlim=(0, 1))
            axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)
        else:
            axes[1].set_visible(False)

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"len{length:02d}.png"), dpi=150)
        plt.close(fig)

    # Overview: probability distribution by class
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, title in zip(axes,
                               ["internal_prob", "b3pps_prob"],
                               ["Internal Classifier", "B3PPs Classifier"]):
        ax.hist(df[df["label"]==1][col], bins=40, alpha=0.7,
                color="steelblue", label="Label=1 (positive)")
        ax.hist(df[df["label"]==0][col], bins=40, alpha=0.7,
                color="darkorange", label="Label=0 (negative)")
        ax.set(title=title, xlabel="Probability", ylabel="Count")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Score distributions — {len(df)} peptides  "
                 f"(+{(df.label==1).sum()} / −{(df.label==0).sum()})",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "overview_by_label.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Plots saved → {save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Using device: {DEVICE}")
    print(f"Targets: {TARGET_TOTALS} (random lengths between {MIN_LEN} and {MAX_LEN})")

    diff_model, diffusion       = load_diffusion_model()
    svm_model, svm_scaler, svm_feature_ids = load_or_train_svm_classifier()
    toxicity_model, toxicity_scaler, toxicity_feature_ids = load_or_train_toxicity_svm_classifier()
    b3pps                       = load_b3pps(B3PPS_PATH, DEVICE)
    print("[✓] All models loaded (including toxicity filter)\n")

    for total_target in TARGET_TOTALS:
        print("\n" + "=" * 70)
        print(f"Generating dataset with total peptides = {total_target}")
        print(f"Filtering for LOW TOXICITY only (threshold: {TOXICITY_THRESH})")
        print("=" * 70)

        df = generate_all(
            total_target=total_target,
            diff_model=diff_model,
            diffusion=diffusion,
            svm_model=svm_model,
            svm_scaler=svm_scaler,
            svm_feature_ids=svm_feature_ids,
            b3pps=b3pps,
            toxicity_model=toxicity_model,
            toxicity_scaler=toxicity_scaler,
            toxicity_feature_ids=toxicity_feature_ids,
            device=DEVICE,
        )

        # ── Summary ───────────────────────────────────────────────────────
        print("\n── Final counts ──────────────────────────────────")
        print(f"  Total        : {len(df)}")
        print(f"  Label=1      : {(df.label==1).sum()}")
        print(f"  Label=0      : {(df.label==0).sum()}")
        
        # Show toxicity statistics
        if 'toxicity_prob' in df.columns:
            print(f"\n  Toxicity stats:")
            print(f"    Min toxicity_prob : {df['toxicity_prob'].min():.4f}")
            print(f"    Max toxicity_prob : {df['toxicity_prob'].max():.4f}")
            print(f"    Mean toxicity_prob: {df['toxicity_prob'].mean():.4f}")
            print(f"    All below {TOXICITY_THRESH}: {(df['toxicity_prob'] < TOXICITY_THRESH).all()}")
        
        print("\n  Per length:")
        for l in range(MIN_LEN, MAX_LEN + 1):
            sub = df[df["len"] == l]
            n1  = (sub.label == 1).sum()
            n0  = (sub.label == 0).sum()
            print(f"    len {l:2d}: total={len(sub):4d}  +{n1:3d}  -{n0:3d}")
        print("──────────────────────────────────────────────────")

        # ── Save ──────────────────────────────────────────────────────────
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_csv = os.path.join(OUTPUT_DIR, f"bbbp_{total_target}_dual_randomlen_lowtox.csv")
        out_txt = os.path.join(OUTPUT_DIR, f"bbbp_{total_target}_dual_randomlen_lowtox.txt")

        df.to_csv(out_csv, index=False)
        df["seq"].to_csv(out_txt, index=False, header=False)

        # Also save standard names for backward compatibility with notebooks
        if total_target == 1000:
            compat_csv = os.path.join(OUTPUT_DIR, "bbbp_1000_dual.csv")
            compat_txt = os.path.join(OUTPUT_DIR, "bbbp_1000_dual.txt")
            df.to_csv(compat_csv, index=False)
            df["seq"].to_csv(compat_txt, index=False, header=False)

        print(f"\n[✓] CSV saved  → {out_csv}")
        print(f"[✓] TXT saved  → {out_txt}")

        plot_dir = os.path.join(PLOTS_DIR, f"total_{total_target}")
        plot_distributions(df, plot_dir)


if __name__ == "__main__":
    main()