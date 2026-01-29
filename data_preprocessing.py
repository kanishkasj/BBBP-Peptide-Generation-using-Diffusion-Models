import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    DATA_PATH, AMINO_ACIDS, MIN_SEQ_LEN, MAX_SEQ_LEN, 
    TRAIN_VAL_SPLIT, BATCH_SIZE, RANDOM_SEED, DEVICE
)
from utils import set_seed, tokenize_sequence, is_valid_sequence


def load_and_preprocess_data():
    """
    Load CSV and preprocess the peptide dataset
    
    Returns:
        DataFrame with cleaned peptide data
    """
    print("=" * 60)
    print("STEP 1: Dataset Preprocessing")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Rename columns if needed (type -> label)
    if 'type' in df.columns and 'label' not in df.columns:
        df = df.rename(columns={'type': 'label'})
    
    print(f"Original dataset size: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Step 1: Remove sequences with non-standard amino acids
    print("\n[1/5] Removing sequences with non-standard amino acids...")
    df['valid'] = df['seq'].apply(is_valid_sequence)
    invalid_count = (~df['valid']).sum()
    df = df[df['valid']].drop('valid', axis=1)
    print(f"  Removed {invalid_count} sequences with non-standard AAs")
    
    # Step 2: Filter by sequence length (5-20)
    print("\n[2/5] Filtering by sequence length (5-20)...")
    # Recalculate length to be safe
    df['len'] = df['seq'].apply(len)
    len_before = len(df)
    df = df[(df['len'] >= MIN_SEQ_LEN) & (df['len'] <= MAX_SEQ_LEN)]
    print(f"  Removed {len_before - len(df)} sequences outside length range")
    
    # Step 3: Ensure labels are binary
    print("\n[3/5] Ensuring binary labels...")
    df['label'] = df['label'].astype(int)
    unique_labels = df['label'].unique()
    print(f"  Unique labels: {unique_labels}")
    assert all(l in [0, 1] for l in unique_labels), "Labels must be binary (0/1)"
    
    # Step 4: Remove duplicates
    print("\n[4/5] Removing duplicate sequences...")
    dup_count = df.duplicated(subset=['seq']).sum()
    df = df.drop_duplicates(subset=['seq'])
    print(f"  Removed {dup_count} duplicate sequences")
    
    # Step 5: Dataset statistics
    print("\n[5/5] Final dataset statistics:")
    print(f"  Total sequences: {len(df)}")
    print(f"  BBBP positive (label=1): {(df['label'] == 1).sum()}")
    print(f"  BBBP negative (label=0): {(df['label'] == 0).sum()}")
    print(f"  Length distribution:")
    print(df['len'].value_counts().sort_index().to_string())
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


def split_dataset(df, test_size=TRAIN_VAL_SPLIT, random_state=RANDOM_SEED):
    """
    Split dataset into training and validation sets
    
    Args:
        df: Preprocessed DataFrame
        test_size: Proportion for validation
        random_state: Random seed
        
    Returns:
        train_df, val_df
    """
    print("\nSplitting dataset into train/validation...")
    
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['label']  
    )
    
    print(f"  Training set: {len(train_df)} samples")
    print(f"  Validation set: {len(val_df)} samples")
    print(f"  Train BBBP+: {(train_df['label'] == 1).sum()}, BBBP-: {(train_df['label'] == 0).sum()}")
    print(f"  Val BBBP+: {(val_df['label'] == 1).sum()}, BBBP-: {(val_df['label'] == 0).sum()}")
    
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


class PeptideDataset(Dataset):
    """PyTorch Dataset for peptide sequences"""
    
    def __init__(self, df, max_len=MAX_SEQ_LEN):
        """
        Args:
            df: DataFrame with 'seq', 'len', 'label' columns
            max_len: Maximum sequence length for padding
        """
        self.sequences = df['seq'].values
        self.lengths = df['len'].values
        self.labels = df['label'].values
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        length = self.lengths[idx]
        label = self.labels[idx]
        
        # Tokenize sequence
        tokens = tokenize_sequence(seq, self.max_len)
        
        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'length': torch.tensor(length, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float),
            'sequence': seq
        }


def create_dataloaders(train_df, val_df, batch_size=BATCH_SIZE):
    """
    Create DataLoaders for training and validation
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        batch_size: Batch size
        
    Returns:
        train_loader, val_loader
    """
    train_dataset = PeptideDataset(train_df)
    val_dataset = PeptideDataset(val_df)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


def get_bbbp_positive_data(df):
    """
    Get only BBBP positive sequences for diffusion training
    
    Args:
        df: Full DataFrame
        
    Returns:
        DataFrame with only BBBP positive samples
    """
    bbbp_df = df[df['label'] == 1].reset_index(drop=True)
    print(f"\nBBBP positive samples for diffusion training: {len(bbbp_df)}")
    return bbbp_df


if __name__ == "__main__":
    # Test preprocessing
    set_seed()
    df = load_and_preprocess_data()
    train_df, val_df = split_dataset(df)
    train_loader, val_loader = create_dataloaders(train_df, val_df)
    
    # Test a batch
    batch = next(iter(train_loader))
    print("\nSample batch:")
    print(f"  Tokens shape: {batch['tokens'].shape}")
    print(f"  Lengths: {batch['length'][:5]}")
    print(f"  Labels: {batch['label'][:5]}")
    print(f"  Sequences: {batch['sequence'][:3]}")
