import os
from collections import Counter, defaultdict

import pandas as pd

from config import GENERATED_DIR, MIN_SEQ_LEN, MAX_SEQ_LEN, AMINO_ACIDS
from classifier import load_classifier, predict_bbbp
from diffusion_model import load_diffusion_model
from generation import generate_peptides


TARGET_TOTAL = 1000
THRESHOLD = 0.8
FIRST_BUCKET_COUNT = 63  # lengths 5-12 (8 lengths)
SECOND_BUCKET_COUNT = 62  # lengths 13-20 (8 lengths)


def has_repeated_trimer(seq: str) -> bool:
    c = Counter(seq[i:i + 3] for i in range(len(seq) - 2))
    return any(v > 1 for v in c.values())


def is_valid_sequence(seq: str, length: int) -> bool:
    if not isinstance(seq, str):
        return False
    if len(seq) != length:
        return False
    if any(ch not in AMINO_ACIDS for ch in seq):
        return False
    if has_repeated_trimer(seq):
        return False
    return True


def get_target_quota() -> dict:
    quota = {}
    for l in range(5, 13):
        quota[l] = FIRST_BUCKET_COUNT
    for l in range(13, 21):
        quota[l] = SECOND_BUCKET_COUNT
    return quota


def load_existing_validated() -> dict:
    pools = defaultdict(list)

    for length in range(MIN_SEQ_LEN, MAX_SEQ_LEN + 1):
        path = os.path.join(GENERATED_DIR, f"validated_len_{length}.csv")
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        # Keep only validated positives from existing run
        df = df[df["valid"] == True].copy()

        # Ensure we have numeric probability and keep stronger candidates first
        if "probability" in df.columns:
            df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
            df = df.sort_values("probability", ascending=False)
        else:
            df["probability"] = 1.0

        seen = set()
        for _, row in df.iterrows():
            seq = str(row["seq"])
            prob = float(row["probability"]) if pd.notna(row["probability"]) else 0.0
            if prob < THRESHOLD:
                continue
            if seq in seen:
                continue
            if not is_valid_sequence(seq, length):
                continue
            seen.add(seq)
            pools[length].append({
                "seq": seq,
                "len": length,
                "label": 1,
                "probability": prob,
                "source": "existing_validated",
            })

    return pools


def top_up_with_generation(pools: dict, quota: dict) -> dict:
    classifier_model, scaler, biovec_model = load_classifier()
    diffusion_model, diffusion = load_diffusion_model()

    # Global uniqueness across the final set
    global_seen = set()
    for length in range(MIN_SEQ_LEN, MAX_SEQ_LEN + 1):
        for item in pools.get(length, []):
            global_seen.add(item["seq"])

    for length in range(MIN_SEQ_LEN, MAX_SEQ_LEN + 1):
        need = quota[length] - len(pools.get(length, []))
        if need <= 0:
            continue

        print(f"\nLength {length}: need {need} more")
        rounds = 0

        while need > 0 and rounds < 60:
            rounds += 1
            # Oversample because classifier filtering is strict.
            sample_size = max(150, need * 10)
            candidates = generate_peptides(
                diffusion_model,
                diffusion,
                target_length=length,
                num_samples=sample_size,
                bbbp_label=1,
                temperature=0.8,
                top_p=0.9,
            )

            accepted_in_round = 0
            for seq in candidates:
                if seq in global_seen:
                    continue
                if not is_valid_sequence(seq, length):
                    continue

                prob, pred = predict_bbbp(seq, classifier_model, scaler, biovec_model)
                if pred != 1 or prob < THRESHOLD:
                    continue

                record = {
                    "seq": seq,
                    "len": length,
                    "label": 1,
                    "probability": float(prob),
                    "source": "new_generation",
                }
                pools[length].append(record)
                global_seen.add(seq)
                accepted_in_round += 1
                need -= 1

                if need == 0:
                    break

            print(f"  round {rounds:02d}: +{accepted_in_round}, remaining {need}")

        if need > 0:
            print(f"WARNING: Length {length} still short by {need} after max rounds")

    return pools


def build_final_dataset(pools: dict, quota: dict) -> pd.DataFrame:
    rows = []

    for length in range(MIN_SEQ_LEN, MAX_SEQ_LEN + 1):
        candidates = pools.get(length, [])
        # Prefer high-confidence peptides when there are extras.
        candidates = sorted(candidates, key=lambda x: x["probability"], reverse=True)
        selected = candidates[: quota[length]]
        rows.extend(selected)

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["len", "probability"], ascending=[True, False]).reset_index(drop=True)
    return out_df


def main():
    quota = get_target_quota()
    assert sum(quota.values()) == TARGET_TOTAL, "Quota must sum to 1000"

    print("Loading existing validated peptides...")
    pools = load_existing_validated()

    print("Initial availability by length:")
    for length in range(MIN_SEQ_LEN, MAX_SEQ_LEN + 1):
        print(f"  len {length:2d}: {len(pools.get(length, []))}")

    pools = top_up_with_generation(pools, quota)

    out_df = build_final_dataset(pools, quota)

    # Final checks
    out_df = out_df.drop_duplicates(subset=["seq"]).reset_index(drop=True)
    out_df["label"] = 1

    # Re-verify constraints after dedupe
    out_df = out_df[
        out_df.apply(lambda r: is_valid_sequence(r["seq"], int(r["len"])), axis=1)
    ].copy()

    # If any count drift happened due dedupe/filter, report it.
    counts = out_df.groupby("len").size().to_dict()
    print("\nFinal counts by length:")
    for length in range(MIN_SEQ_LEN, MAX_SEQ_LEN + 1):
        print(f"  len {length:2d}: {counts.get(length, 0)}")
    print(f"Total: {len(out_df)}")

    output_csv = os.path.join(GENERATED_DIR, "balanced_1000_label1_no_repeat3.csv")
    output_txt = os.path.join(GENERATED_DIR, "balanced_1000_label1_no_repeat3.txt")

    out_df.to_csv(output_csv, index=False)
    out_df["seq"].to_csv(output_txt, index=False, header=False)

    print(f"\nSaved CSV: {output_csv}")
    print(f"Saved TXT: {output_txt}")


if __name__ == "__main__":
    main()
