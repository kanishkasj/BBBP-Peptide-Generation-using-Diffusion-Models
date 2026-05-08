"""
generation_5000.py
Generate exactly 5000 BBBP peptides with random lengths from 5 to 20.
Uses diffusion model + AAIndex top-17 SVM + B3PPS.
Keeps the B3PPS GitHub model unchanged.
"""

import ast
import os
import pickle
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from aaindex import aaindex1
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import EsmForSequenceClassification, EsmTokenizer

from config import AA_TO_IDX, AMINO_ACIDS, DEVICE, MAX_SEQ_LEN
from diffusion_model import load_diffusion_model

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "generated")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "prob_plots")
B3PPS_PATH = os.path.join(OUTPUT_DIR, "../b3pps/best_model5")
B3PPS_PATH = os.path.normpath(B3PPS_PATH)

SVM_TOP17_FEATURES_PATH = os.path.join(PROJECT_DIR, "svm_fixed_top17_features.csv")
SVM_PARAMS_SOURCE_PATH = os.path.join(PROJECT_DIR, "svm_10_runs_results.csv")
SVM_TRAIN_CSV_PATH = os.path.join(OUTPUT_DIR, "bbbp_1000_dual.csv")
SVM_MODEL_CACHE_PATH = os.path.join(OUTPUT_DIR, "svm_top17_model.pkl")
SVM_SCALER_CACHE_PATH = os.path.join(OUTPUT_DIR, "svm_top17_scaler.pkl")
SVM_FEATURES_CACHE_PATH = os.path.join(OUTPUT_DIR, "svm_top17_feature_ids.pkl")

IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}
VOCAB_SIZE = len(AMINO_ACIDS)
MIN_LEN, MAX_LEN = 5, 20
TARGET_TOTAL = 5000
POSITIVE_RATIO = 0.70
INT_THRESH = 0.5
B3P_THRESH = 0.5
INT_NEG_THRESH = 0.35
B3P_NEG_THRESH = 0.35
CLF_LOGIC = "or"
BATCH_CLF = 64
MAX_ROUNDS = 60

# ─────────────────────────────────────────────────────────────────────────────
# LENGTH QUOTAS
# ─────────────────────────────────────────────────────────────────────────────
def get_random_length_quota(total: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(MIN_LEN, MAX_LEN + 1, size=total)
    quota = {length: int(np.sum(lengths == length)) for length in range(MIN_LEN, MAX_LEN + 1)}
    assert sum(quota.values()) == total, f"Quota sum {sum(quota.values())} != {total}"
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


# ─────────────────────────────────────────────────────────────────────────────
# B3PPS MODEL
# ─────────────────────────────────────────────────────────────────────────────
def load_b3pps(path, device):
    if not os.path.exists(path):
        print(f"[!] B3PPs not found at {path}")
        return None
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D", do_lower_case=False)
    model = EsmForSequenceClassification.from_pretrained(path).to(device).eval()
    print("[✓] B3PPs ESM classifier loaded")
    return tokenizer, model


# ─────────────────────────────────────────────────────────────────────────────
# SVM MODEL
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# DIFFUSION SAMPLING
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def diffusion_sample(model, diffusion, length, num_samples, device):
    B = num_samples
    T = diffusion.timesteps

    mask = torch.zeros(B, MAX_SEQ_LEN, dtype=torch.bool, device=device)
    mask[:, length:] = True
    lengths_t = torch.full((B,), length, dtype=torch.long, device=device)
    labels_t = torch.ones((B,), dtype=torch.long, device=device)

    x = torch.zeros(B, MAX_SEQ_LEN, dtype=torch.long, device=device)
    x[:, :length] = torch.randint(1, VOCAB_SIZE + 1, (B, length), device=device)

    for t_val in range(T - 1, -1, -1):
        t_tensor = torch.full((B,), t_val, dtype=torch.long, device=device)
        logits = model(x, t_tensor, lengths_t, labels_t, mask)
        logits = logits[:, :length, :]

        x0_pred = torch.multinomial(
            F.softmax(logits.contiguous().view(B * length, 20), dim=-1), 1
        ).view(B, length)

        if t_val == 0:
            x[:, :length] = logits.argmax(dim=-1) + 1
            break

        alpha_prev = diffusion.alpha_cumprod[t_val - 1].to(device)
        corrupt_prob = 1.0 - alpha_prev
        keep_mask = torch.rand(B, length, device=device) >= corrupt_prob
        random_toks = torch.randint(0, VOCAB_SIZE, (B, length), device=device)
        x_prev = torch.where(keep_mask, x0_pred, random_toks)
        x[:, :length] = x_prev + 1

    seqs = []
    for b in range(B):
        seq = "".join(IDX_TO_AA.get((x[b, p] - 1).item(), "") for p in range(length))
        seq = "".join(c for c in seq if c in AA_TO_IDX)
        if seq:
            seqs.append(seq)
    return seqs


# ─────────────────────────────────────────────────────────────────────────────
# FILTERS + SCORING
# ─────────────────────────────────────────────────────────────────────────────
MAX_SINGLE_AA_FRAC = 0.4
MIN_UNIQUE_AA = 4
MAX_NGRAM_FRAC = 0.35


def is_valid(seq, length):
    if not isinstance(seq, str) or len(seq) != length:
        return False
    if any(c not in AA_TO_IDX for c in seq):
        return False
    if seq.count("CK") > 1:
        return False

    counts = Counter(seq)
    if counts.most_common(1)[0][1] / length > MAX_SINGLE_AA_FRAC:
        return False
    if len(counts) < MIN_UNIQUE_AA:
        return False

    trimers = [seq[i:i + 3] for i in range(length - 2)]
    tri_counts = Counter(trimers)
    most_common_tri_frac = tri_counts.most_common(1)[0][1] / len(trimers)
    if length <= 12:
        if most_common_tri_frac > 1 / len(trimers):
            return False
    else:
        if most_common_tri_frac > MAX_NGRAM_FRAC:
            return False
    return True


def score_batch(seqs, svm_model, svm_scaler, svm_feature_ids, b3pps, device):
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

    results = []
    for i, seq in enumerate(seqs):
        results.append({
            "seq": seq,
            "internal_prob": float(int_probs[i]),
            "b3pps_prob": float(b3_probs[i]),
        })
    return results


def is_positive(r: dict) -> bool:
    int_pass = r["internal_prob"] >= INT_THRESH
    b3p_pass = r["b3pps_prob"] >= B3P_THRESH
    return int_pass or b3p_pass if CLF_LOGIC == "or" else int_pass and b3p_pass


def is_negative(r: dict) -> bool:
    return r["internal_prob"] < INT_NEG_THRESH and r["b3pps_prob"] < B3P_NEG_THRESH


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_for_label(label, quota, diff_model, diffusion, svm_model, svm_scaler, svm_feature_ids, b3pps, seen, device):
    accept_fn = is_positive if label == 1 else is_negative
    records = []

    print(f"\n{'=' * 60}")
    print(f"  Generating {sum(quota.values())} {'POSITIVE' if label == 1 else 'NEGATIVE'} peptides")
    print(f"{'=' * 60}")

    for length in range(MIN_LEN, MAX_LEN + 1):
        need, bucket, rounds = quota[length], [], 0
        print(f"\n── Length {length:2d}  (need {need}) ──────────")

        while len(bucket) < need and rounds < MAX_ROUNDS:
            rounds += 1
            raw = diffusion_sample(diff_model, diffusion, length, max(256, need * 8), device)
            cands = [s for s in raw if s not in seen and is_valid(s, length)]

            hit = 0
            for i in range(0, len(cands), BATCH_CLF):
                if len(bucket) >= need:
                    break
                batch = cands[i:i + BATCH_CLF]
                scored = score_batch(batch, svm_model, svm_scaler, svm_feature_ids, b3pps, device)
                for r in scored:
                    if len(bucket) >= need:
                        break
                    if accept_fn(r):
                        seen.add(r["seq"])
                        bucket.append({**r, "len": length, "label": label})
                        hit += 1

            print(f"  r{rounds:02d}: raw={len(raw):4d}  struct={len(cands):4d}  accepted={hit:3d}  bucket={len(bucket)}/{need}")

        if len(bucket) < need:
            print(f"  ⚠ WARNING: only {len(bucket)}/{need} collected for length={length}, label={label}")

        records.extend(bucket[:need])

    return records


def generate_all(total_target, diff_model, diffusion, svm_model, svm_scaler, svm_feature_ids, b3pps, device):
    target_label1 = int(round(total_target * POSITIVE_RATIO))
    target_label0 = total_target - target_label1
    quota1 = get_random_length_quota(target_label1, seed=42 + total_target)
    quota0 = get_random_length_quota(target_label0, seed=99 + total_target)

    seen = set()

    pos_records = generate_for_label(1, quota1, diff_model, diffusion, svm_model, svm_scaler, svm_feature_ids, b3pps, seen, device)
    neg_records = generate_for_label(0, quota0, diff_model, diffusion, svm_model, svm_scaler, svm_feature_ids, b3pps, seen, device)

    df = pd.DataFrame(pos_records + neg_records)
    df = df.drop_duplicates("seq").reset_index(drop=True)
    df = df[df.apply(lambda r: is_valid(r["seq"], int(r["len"])), axis=1)].copy()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def plot_distributions(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    bins = np.linspace(0, 1, 21)

    def side_bars(ax, a, b, bins):
        ca, cb = "steelblue", "darkorange"
        na, _ = np.histogram(a, bins)
        nb, e = np.histogram(b, bins)
        w = (e[1] - e[0]) * 0.42
        cx = (e[:-1] + e[1:]) / 2
        ax.bar(cx - w / 2, na, w, color=ca, alpha=0.85, label="Internal", edgecolor="white")
        ax.bar(cx + w / 2, nb, w, color=cb, alpha=0.85, label="B3PPs", edgecolor="white")

    for length in sorted(df["len"].unique()):
        sub = df[df["len"] == length]
        pos = sub[sub["label"] == 1]
        neg = sub[sub["label"] == 0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        side_bars(axes[0], pos["internal_prob"], pos["b3pps_prob"], bins)
        axes[0].set(title=f"Len {length} — Label=1 (n={len(pos)})", xlabel="Probability", ylabel="Count", xlim=(0, 1))
        axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)
        if len(neg) > 0:
            side_bars(axes[1], neg["internal_prob"], neg["b3pps_prob"], bins)
            axes[1].set(title=f"Len {length} — Label=0 (n={len(neg)})", xlabel="Probability", ylabel="Count", xlim=(0, 1))
            axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)
        else:
            axes[1].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"len{length:02d}.png"), dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, title in zip(axes, ["internal_prob", "b3pps_prob"], ["Internal Classifier", "B3PPs Classifier"]):
        ax.hist(df[df["label"] == 1][col], bins=40, alpha=0.7, color="steelblue", label="Label=1 (positive)")
        ax.hist(df[df["label"] == 0][col], bins=40, alpha=0.7, color="darkorange", label="Label=0 (negative)")
        ax.set(title=title, xlabel="Probability", ylabel="Count")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Score distributions — {len(df)} peptides  (+{(df.label==1).sum()} / −{(df.label==0).sum()})", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "overview_by_label.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Plots saved → {save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Using device: {DEVICE}")
    print(f"Generating exactly {TARGET_TOTAL} peptides with random lengths {MIN_LEN}-{MAX_LEN}")

    diff_model, diffusion = load_diffusion_model()
    svm_model, svm_scaler, svm_feature_ids = load_or_train_svm_classifier()
    b3pps = load_b3pps(B3PPS_PATH, DEVICE)
    print("[✓] All models loaded\n")

    df = generate_all(
        total_target=TARGET_TOTAL,
        diff_model=diff_model,
        diffusion=diffusion,
        svm_model=svm_model,
        svm_scaler=svm_scaler,
        svm_feature_ids=svm_feature_ids,
        b3pps=b3pps,
        device=DEVICE,
    )

    print("\n── Final counts ──────────────────────────────────")
    print(f"  Total        : {len(df)}")
    print(f"  Label=1      : {(df.label==1).sum()}")
    print(f"  Label=0      : {(df.label==0).sum()}")
    print("\n  Per length:")
    for l in range(MIN_LEN, MAX_LEN + 1):
        sub = df[df["len"] == l]
        n1 = (sub.label == 1).sum()
        n0 = (sub.label == 0).sum()
        print(f"    len {l:2d}: total={len(sub):4d}  +{n1:3d}  -{n0:3d}")
    print("──────────────────────────────────────────────────")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUT_DIR, "bbbp_5000_dual_randomlen.csv")
    out_txt = os.path.join(OUTPUT_DIR, "bbbp_5000_dual_randomlen.txt")
    df.to_csv(out_csv, index=False)
    df["seq"].to_csv(out_txt, index=False, header=False)

    print(f"\n[✓] CSV saved  → {out_csv}")
    print(f"[✓] TXT saved  → {out_txt}")

    plot_distributions(df, os.path.join(PLOTS_DIR, "total_5000"))


if __name__ == "__main__":
    main()
