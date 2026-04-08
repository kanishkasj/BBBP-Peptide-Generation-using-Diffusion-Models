"""
generation.py  —  Generate 1000 BBBP peptides (lengths 5-20)
Uses the REAL diffusion model architecture from diffusion_model.py
"""
import os, pickle
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import EsmTokenizer, EsmForSequenceClassification

# ── Import real model (no re-definition needed) ──────────────────────────────
from diffusion_model import PeptideDiffusionModel, DiscreteDiffusion, load_diffusion_model
from classifier import load_classifier
from ifeature_descriptors import iFeatureExtractor
from config import DEVICE, MAX_SEQ_LEN, AA_TO_IDX, AMINO_ACIDS

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
B3PPS_PATH  = r"E:\COLLEGE\AMRITA VISHWA VIDYPEETHAM _ NEW\New folder\DD\diffusion_bbbp\outputs\b3pps\best_model5"
OUTPUT_DIR  = r"E:\COLLEGE\AMRITA VISHWA VIDYPEETHAM _ NEW\New folder\DD\diffusion_bbbp\outputs\generated"
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "prob_plots")

IDX_TO_AA       = {i: aa for aa, i in AA_TO_IDX.items()}
VOCAB_SIZE      = len(AMINO_ACIDS)          # 20
MIN_LEN, MAX_LEN = 5, 20
TARGET_TOTAL    = 1000
INT_THRESH      = 0.5
B3P_THRESH      = 0.5
CLF_LOGIC       = "or"      # "or" = either passes | "and" = both must pass
TRIMER_MAX_LEN  = 12        # trimer filter only for seqs shorter than this
BATCH_CLF       = 64
MAX_ROUNDS      = 60

def get_quota():
    q = {l: 63 for l in range(MIN_LEN, 13)}
    q.update({l: 62 for l in range(13, MAX_LEN + 1)})
    assert sum(q.values()) == TARGET_TOTAL
    return q

# ─────────────────────────────────────────────────────────────────────────────
# SAMPLING  (uses real model signature + 1-indexed tokens)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def diffusion_sample(model, diffusion, length, num_samples, device):
    """
    Correct reverse discrete diffusion:
      t=T → predict clean x0 → re-noise to t-1 → repeat → t=0
    Uses the actual DiscreteDiffusion noise schedule (alpha_cumprod).
    """
    B = num_samples
    T = diffusion.timesteps

    mask = torch.zeros(B, MAX_SEQ_LEN, dtype=torch.bool, device=device)
    mask[:, length:] = True
    lengths_t = torch.full((B,), length, dtype=torch.long, device=device)
    labels_t  = torch.ones((B,),          dtype=torch.long, device=device)

    # Start: fully noisy (random tokens 1-20)
    x = torch.zeros(B, MAX_SEQ_LEN, dtype=torch.long, device=device)
    x[:, :length] = torch.randint(1, VOCAB_SIZE + 1, (B, length), device=device)

    for t_val in range(T - 1, -1, -1):
        t_tensor = torch.full((B,), t_val, dtype=torch.long, device=device)

        # 1. Predict clean tokens x0 from noisy x_t
        logits = model(x, t_tensor, lengths_t, labels_t, mask)  # (B, MAX_SEQ_LEN, 20)
        logits = logits[:, :length, :]                           # (B, L, 20)

        # Sample predicted x0 (0-indexed)
        x0_pred = torch.multinomial(
            F.softmax(logits.contiguous().view(B * length, 20), dim=-1), 1
        ).view(B, length)  # 0-indexed (0–19)

        if t_val == 0:
            # Final step: just use argmax, no re-noising
            x[:, :length] = logits.argmax(dim=-1) + 1  # back to 1-indexed
            break

        # 2. Re-noise x0_pred to level t-1 using the noise schedule
        alpha_prev = diffusion.alpha_cumprod[t_val - 1].to(device)
        corrupt_prob = 1.0 - alpha_prev  # probability of replacing with random token

        keep_mask = torch.rand(B, length, device=device) >= corrupt_prob
        random_tokens = torch.randint(0, VOCAB_SIZE, (B, length), device=device)  # 0-indexed

        # Where keep_mask=True → use x0_pred, else → random token
        x_prev = torch.where(keep_mask, x0_pred, random_tokens)
        x[:, :length] = x_prev + 1  # shift to 1-indexed

    # Decode
    seqs = []
    for b in range(B):
        seq = "".join(IDX_TO_AA.get((x[b, p] - 1).item(), "") for p in range(length))
        seq = "".join(c for c in seq if c in AA_TO_IDX)
        if seq:
            seqs.append(seq)
    return seqs

# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL FILTER
# ─────────────────────────────────────────────────────────────────────────────
MAX_SINGLE_AA_FRAC = 0.4   # no single AA can be >40% of the sequence
MIN_UNIQUE_AA      = 4     # sequence must contain at least 4 distinct AAs
MAX_NGRAM_FRAC     = 0.35  # no single 3-mer can appear in >35% of positions

def is_valid(seq, length):
    if not isinstance(seq, str) or len(seq) != length: return False
    if any(c not in AA_TO_IDX for c in seq):           return False

    # 1. Block "CK" dimers
    if seq.count("CK") > 1:                            return False

    # 2. Diversity: no single AA dominates (catches NNNNN, QQQQQ, TTTTT)
    counts = Counter(seq)
    if counts.most_common(1)[0][1] / length > MAX_SINGLE_AA_FRAC: return False

    # 3. Minimum unique AAs (catches low-complexity sequences)
    if len(counts) < MIN_UNIQUE_AA:                    return False

    # 4. Soft ngram repetition — scaled to length
    #    Short seqs: strict (no repeated trimers)
    #    Long seqs:  allow some repeats but cap fraction
    trimers = [seq[i:i+3] for i in range(length - 2)]
    tri_counts = Counter(trimers)
    most_common_tri_frac = tri_counts.most_common(1)[0][1] / len(trimers)
    if length <= TRIMER_MAX_LEN:
        if most_common_tri_frac > 1 / len(trimers):   return False  # strict: no repeats
    else:
        if most_common_tri_frac > MAX_NGRAM_FRAC:      return False  # soft cap

    return True

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIERS
# ─────────────────────────────────────────────────────────────────────────────
def load_b3pps(path, device):
    if not os.path.exists(path):
        print(f"[!] B3PPs not found at {path}"); return None
    tok = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D", do_lower_case=False)
    mdl = EsmForSequenceClassification.from_pretrained(path).to(device).eval()
    print("[✓] B3PPs ESM classifier loaded"); return (tok, mdl)

def score_batch(seqs, int_model, scaler, biovec, extractor, b3pps, device):
    from classifier import BBBPClassifier
    # Internal classifier
    ngram = biovec.ngram
    ifeats = scaler.transform(np.array([extractor.extract_all_features(s) for s in seqs]))
    bvecs  = np.array([biovec.embed_sequence_padded(s, MAX_SEQ_LEN) for s in seqs])
    lens   = [len(s) - ngram + 1 for s in seqs]
    with torch.no_grad():
        int_probs = int_model(
            torch.tensor(bvecs,  dtype=torch.float32).to(device),
            torch.tensor(ifeats, dtype=torch.float32).to(device),
            torch.tensor(lens,   dtype=torch.long).to(device),
        ).cpu().numpy()
    int_preds = (int_probs >= INT_THRESH).astype(int)

    # B3PPs classifier
    if b3pps:
        tok, mdl = b3pps
        enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            b3_probs = torch.sigmoid(mdl(**enc).logits)[:, 1].cpu().numpy()
        b3_preds = (b3_probs >= B3P_THRESH).astype(int)
    else:
        b3_probs, b3_preds = int_probs.copy(), int_preds.copy()

    results = []
    for i, seq in enumerate(seqs):
        ok = (int_preds[i] or b3_preds[i]) if CLF_LOGIC == "or" else (int_preds[i] and b3_preds[i])
        results.append({"seq": seq, "internal_prob": float(int_probs[i]),
                        "b3pps_prob": float(b3_probs[i])} if ok else None)
    return results

# ─────────────────────────────────────────────────────────────────────────────
# GENERATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def generate_all(diff_model, diffusion, int_model, scaler, biovec, b3pps, extractor, quota, device):
    seen, records = set(), []
    for length in range(MIN_LEN, MAX_LEN + 1):
        need, bucket, rounds = quota[length], [], 0
        print(f"\n── Length {length:2d}  (need {need}) ──────────")
        while len(bucket) < need and rounds < MAX_ROUNDS:
            rounds += 1
            raw  = diffusion_sample(diff_model, diffusion, length, max(256, need * 8), device)
            cands = [s for s in raw if s not in seen and is_valid(s, length)]
            hit = 0
            for i in range(0, len(cands), BATCH_CLF):
                if len(bucket) >= need: break
                for seq, r in zip(cands[i:i+BATCH_CLF],
                                  score_batch(cands[i:i+BATCH_CLF], int_model, scaler,
                                              biovec, extractor, b3pps, device)):
                    if r and len(bucket) < need:
                        seen.add(seq); bucket.append({**r, "seq": seq, "len": length, "label": 1}); hit += 1
            print(f"  r{rounds:02d}: raw={len(raw)} struct={len(cands)} clf={hit} total={len(bucket)}/{need}")
        if len(bucket) < need: print(f"  ⚠ WARNING: {len(bucket)}/{need}")
        records.extend(bucket[:need])
    return pd.DataFrame(records).sort_values("len").reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_distributions(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    bins = np.linspace(0, 1, 21)
    def side_bars(ax, a, b, bins):
        ca, cb = "steelblue", "darkorange"
        na, _ = np.histogram(a, bins); nb, e = np.histogram(b, bins)
        w = (e[1]-e[0]) * 0.42; cx = (e[:-1]+e[1:])/2
        ax.bar(cx-w/2, na, w, color=ca, alpha=0.85, label="Internal", edgecolor="white")
        ax.bar(cx+w/2, nb, w, color=cb, alpha=0.85, label="B3PPs",    edgecolor="white")

    for length in sorted(df["len"].unique()):
        sub = df[df["len"] == length]
        fig, ax = plt.subplots(figsize=(7, 4))
        side_bars(ax, sub["internal_prob"], sub["b3pps_prob"], bins)
        ax.set(title=f"Length {length} (n={len(sub)})", xlabel="Probability",
               ylabel="Count", xlim=(0,1)); ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"len{length:02d}.png"), dpi=150); plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    for ax, l in zip(axes.flatten(), sorted(df["len"].unique())):
        sub = df[df["len"] == l]; side_bars(ax, sub["internal_prob"], sub["b3pps_prob"], bins)
        ax.set_title(f"Len {l} (n={len(sub)})", fontsize=9); ax.set_xlim(0,1); ax.grid(axis="y", alpha=0.3)
    fig.legend(handles=[plt.Rectangle((0,0),1,1,color="steelblue",alpha=0.85,label="Internal"),
                        plt.Rectangle((0,0),1,1,color="darkorange",alpha=0.85,label="B3PPs")],
               loc="lower right"); fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "overview.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[✓] Plots saved → {save_dir}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Using device: {DEVICE}")
    diff_model, diffusion           = load_diffusion_model()          # from diffusion_model.py
    int_model, scaler, biovec       = load_classifier()               # from classifier.py
    b3pps                           = load_b3pps(B3PPS_PATH, DEVICE)
    extractor                       = iFeatureExtractor()
    print("[✓] All models loaded")

    df = generate_all(diff_model, diffusion, int_model, scaler, biovec, b3pps, extractor, get_quota(), DEVICE)
    df = df.drop_duplicates("seq").reset_index(drop=True)
    df = df[df.apply(lambda r: is_valid(r["seq"], int(r["len"])), axis=1)].copy()

    counts = df.groupby("len").size().to_dict()
    print("\n── Final counts ──────────────────")
    for l in range(MIN_LEN, MAX_LEN+1): print(f"  len {l:2d}: {counts.get(l,0):3d}")
    print(f"  TOTAL : {len(df)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "bbbp_1000_dual.csv"), index=False)
    df["seq"].to_csv(os.path.join(OUTPUT_DIR, "bbbp_1000_dual.txt"), index=False, header=False)
    plot_distributions(df, PLOTS_DIR)

if __name__ == "__main__":
    main()
