"""
Step 8: Validation of Generated Peptides
Validates generated peptides using the trained BBBP classifier
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from config import (
    DEVICE, BBBP_THRESHOLD, GENERATED_DIR, MIN_SEQ_LEN, MAX_SEQ_LEN,
    CLASSIFIER_DIR, AMINO_ACIDS
)
from classifier import load_classifier, predict_bbbp
from ifeature_descriptors import iFeatureExtractor


def validate_peptides(peptides, classifier_model, scaler, biovec_model, 
                      threshold=BBBP_THRESHOLD):
    """
    Validate a list of peptides using the trained classifier
    
    Args:
        peptides: List of peptide sequences
        classifier_model: Trained BBBPClassifier
        scaler: Feature scaler
        biovec_model: BioVec model
        threshold: BBBP probability threshold
        
    Returns:
        List of (sequence, probability, is_valid) tuples
    """
    results = []
    
    for seq in tqdm(peptides, desc="Validating"):
        # Check if sequence is valid
        if not all(aa in AMINO_ACIDS for aa in seq.upper()):
            results.append((seq, 0.0, False))
            continue
            
        try:
            prob, pred = predict_bbbp(seq, classifier_model, scaler, biovec_model)
            is_valid = prob >= threshold
            results.append((seq, prob, is_valid))
        except Exception as e:
            print(f"Error validating {seq}: {e}")
            results.append((seq, 0.0, False))
    
    return results


def validate_all_generated(threshold=BBBP_THRESHOLD,return_scores=False):
    """
    Validate all generated peptides from the generated_peptides directory
    
    Args:
        threshold: BBBP probability threshold
        
    Returns:
        Dictionary with validation statistics
    """
    print("\n" + "=" * 60)
    print("STEP 8: Validating Generated Peptides")
    print("=" * 60)
    all_scores = []
    # Load classifier
    print("\nLoading classifier...")
    classifier_model, scaler, biovec_model = load_classifier()
    
    # Track statistics
    stats = {
        'total': 0,
        'valid': 0,
        'by_length': {}
    }
    
    validated_peptides = []
    
    # Process each length file
    for length in range(MIN_SEQ_LEN, MAX_SEQ_LEN + 1):
        csv_path = os.path.join(GENERATED_DIR, f"len_{length}.csv")
        
        if not os.path.exists(csv_path):
            print(f"  No file for length {length}")
            continue
            
        print(f"\nValidating length {length}...")
        df = pd.read_csv(csv_path)
        peptides = df['seq'].tolist()
        
        # Validate
        results = validate_peptides(
            peptides, classifier_model, scaler, biovec_model, threshold
        )
        
        # Count valid
        valid_count = sum(1 for _, _, is_valid in results if is_valid)
        
        stats['total'] += len(peptides)
        stats['valid'] += valid_count
        stats['by_length'][length] = {
            'total': len(peptides),
            'valid': valid_count,
            'percentage': valid_count / len(peptides) * 100 if peptides else 0
        }
        
        print(f"  Valid: {valid_count}/{len(peptides)} ({valid_count/len(peptides)*100:.1f}%)")
        
        # Save validated results
        validated_df = pd.DataFrame([
            {'seq': seq, 'len': length, 'probability': prob, 'valid': is_valid}
            for seq, prob, is_valid in results
        ])
        
        validated_path = os.path.join(GENERATED_DIR, f"validated_len_{length}.csv")
        validated_df.to_csv(validated_path, index=False)
        
        # Collect valid peptides
        for seq, prob, is_valid in results:
            if is_valid:
                validated_peptides.append({
                    'seq': seq,
                    'len': length,
                    'probability': prob
                })
    
    # Save all valid peptides
    if validated_peptides:
        valid_df = pd.DataFrame(validated_peptides)
        valid_path = os.path.join(GENERATED_DIR, "validated_bbbp_peptides.csv")
        valid_df.to_csv(valid_path, index=False)
        print(f"\nValid peptides saved to {valid_path}")
    if return_scores:
        return stats, valid_peptides, all_scores  
    return stats, valid_peptides

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total peptides: {stats['total']}")
    print(f"Valid BBBP+ (≥{threshold:.0%}): {stats['valid']}")
    print(f"Overall acceptance rate: {stats['valid']/stats['total']*100:.1f}%")
    
    print("\nBy length:")
    for length in sorted(stats['by_length'].keys()):
        length_stats = stats['by_length'][length]
        print(f"  Length {length:2d}: {length_stats['valid']:4d}/{length_stats['total']:4d} "
              f"({length_stats['percentage']:.1f}%)")
    
    return stats, validated_peptides


def batch_predict(sequences, classifier_model, scaler, biovec_model):
    """
    Batch prediction for multiple sequences
    
    Args:
        sequences: List of peptide sequences
        classifier_model: Trained classifier
        scaler: Feature scaler
        biovec_model: BioVec model
        
    Returns:
        List of (probability, prediction) tuples
    """
    results = []
    for seq in sequences:
        prob, pred = predict_bbbp(seq, classifier_model, scaler, biovec_model)
        results.append((prob, pred))
    return results


if __name__ == "__main__":
    # Validate all generated peptides
    stats, valid_peptides = validate_all_generated()
    
    print(f"\n\nValidation complete!")
    print(f"Total valid BBBP+ peptides: {len(valid_peptides)}")
