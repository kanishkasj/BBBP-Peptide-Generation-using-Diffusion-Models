import argparse
import os
from typing import Iterable

import pandas as pd
import torch
from transformers import EsmForSequenceClassification, EsmTokenizer


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def is_valid_sequence(seq: str) -> bool:
    if not isinstance(seq, str) or len(seq) == 0:
        return False
    if len(seq) > 30:
        return False
    return all(ch in AMINO_ACIDS for ch in seq)


def load_b3pps_model(model_dir: str, device: torch.device):
    tokenizer = EsmTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = EsmForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def predict_b3pps(
    sequences: Iterable[str],
    tokenizer: EsmTokenizer,
    model: EsmForSequenceClassification,
    device: torch.device,
    batch_size: int = 128,
):
    seqs = list(sequences)
    probs = []
    labels = []

    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=30,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        logits = model(**inputs).logits
        p = torch.sigmoid(logits)[:, 1].detach().cpu().numpy()
        probs.extend(p.tolist())
        labels.extend(["B3PP" if x > 0.5 else "Non-B3PP" for x in p])

    return probs, labels


def main():
    parser = argparse.ArgumentParser(
        description="Validate generated peptides using B3PPs classifier from GitHub"
    )
    parser.add_argument(
        "--input",
        default="outputs/generated/bbbp_1000.csv",
        help="Input CSV containing generated peptides (must include 'seq' column)",
    )
    parser.add_argument(
        "--model-dir",
        default="B3PPs/Prediction/model/best_model5",
        help="Path to local B3PPs model directory",
    )
    parser.add_argument(
        "--output-all",
        default="outputs/generated/b3pps_predictions_all.csv",
        help="Output CSV path for all predictions",
    )
    parser.add_argument(
        "--output-positive",
        default="outputs/generated/b3pps_predictions_positive.csv",
        help="Output CSV path for predicted B3PP peptides",
    )
    parser.add_argument(
        "--exclude-motif",
        default="",
        help="Optional motif to exclude from final positive set (example: CK)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for model inference",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    os.makedirs(os.path.dirname(args.output_all), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_positive), exist_ok=True)

    df = pd.read_csv(args.input)
    if "seq" not in df.columns and "Sequence" in df.columns:
        df = df.rename(columns={"Sequence": "seq"})
    if "seq" not in df.columns:
        raise ValueError("Input CSV must contain 'seq' or 'Sequence' column")

    df["seq"] = df["seq"].astype(str).str.upper().str.strip()
    df = df.drop_duplicates(subset=["seq"]).reset_index(drop=True)

    valid_mask = df["seq"].apply(is_valid_sequence)
    invalid_count = int((~valid_mask).sum())
    df = df[valid_mask].copy().reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Valid sequences for prediction: {len(df)}")
    if invalid_count:
        print(f"Dropped invalid sequences: {invalid_count}")

    tokenizer, model = load_b3pps_model(args.model_dir, device)
    probs, labels = predict_b3pps(df["seq"].tolist(), tokenizer, model, device, args.batch_size)

    df["b3pps_probability"] = probs
    df["b3pps_class"] = labels

    positive = df[df["b3pps_class"] == "B3PP"].copy()

    motif = args.exclude_motif.strip().upper()
    if motif:
        before = len(positive)
        positive = positive[~positive["seq"].str.contains(motif, regex=False)]
        print(f"Excluded motif '{motif}': {before - len(positive)} peptides removed")

    df.to_csv(args.output_all, index=False)
    positive.to_csv(args.output_positive, index=False)

    print("\nSummary")
    print(f"Total input unique valid sequences: {len(df)}")
    print(f"Predicted B3PP: {len(positive)}")
    print(f"Saved all predictions: {args.output_all}")
    print(f"Saved B3PP positives: {args.output_positive}")


if __name__ == "__main__":
    main()
