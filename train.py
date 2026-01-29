"""
Main Training and Generation Pipeline
Runs all steps sequentially to train models and generate BBBP peptides
"""

import os
import sys
import argparse
from datetime import datetime

# Ensure the script directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEVICE, CLASSIFIER_DIR, DIFFUSION_DIR, GENERATED_DIR,
    OUTPUT_DIR, BBBP_THRESHOLD, NUM_PEPTIDES_PER_LENGTH,
    MIN_SEQ_LEN, MAX_SEQ_LEN
)
from utils import set_seed
from diffusion_model import train_diffusion_model
from diffusion_evaluation import DiffusionTrainingTracker  
        

def run_full_pipeline(skip_training=False, skip_generation=False, skip_validation=False):
    """
    Run the complete peptide BBBP generation pipeline
    """
    print("=" * 70)
    print("PEPTIDE BBBP GENERATION SYSTEM - DIFFUSION MODEL")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    set_seed()
    
    # ========================================
    # STEP 1: Data Preprocessing
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 1: DATA PREPROCESSING")
    print("=" * 70)
    
    from data_preprocessing import (
        load_and_preprocess_data, split_dataset, 
        create_dataloaders, get_bbbp_positive_data
    )
    
    df = load_and_preprocess_data()
    train_df, val_df = split_dataset(df)
    bbbp_df = get_bbbp_positive_data(df)
    
    print(f"\n✓ Data preprocessing complete")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    print(f"  BBBP+ samples for diffusion: {len(bbbp_df)}")
    
    training_sequences = bbbp_df['seq'].tolist()
    
    # ========================================
    # STEP 2-4: Train BBBP Classifier
    # ========================================
    if not skip_training:
        print("\n" + "=" * 70)
        print("PHASE 2: TRAINING BBBP CLASSIFIER")
        print("=" * 70)
        
        from classifier import train_classifier
        
        classifier_model, scaler, biovec_model = train_classifier(train_df, val_df)
        
        print(f"\n✓ Classifier training complete")
        print(f"  Model saved to: {CLASSIFIER_DIR}")
    else:
        print("\n[SKIP] Classifier training skipped (using pre-trained models)")
    
    # ========================================
    # STEP 5: Train Diffusion Model
    # ========================================
    diffusion_tracker = None
    
    if not skip_training:
        print("\n" + "=" * 70)
        print("PHASE 3: TRAINING DIFFUSION MODEL")
        print("=" * 70)
        
        diffusion_tracker = DiffusionTrainingTracker()
        
        diffusion_model, diffusion = train_diffusion_model(
            bbbp_df['seq'].tolist(),
            bbbp_df['label'].tolist(),
            tracker=diffusion_tracker  
        )
        
        eval_dir = os.path.join(OUTPUT_DIR, 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)
        diffusion_tracker.plot_training_curves(
            save_path=os.path.join(eval_dir, 'diffusion_training_curves.png')
        )
        
        print(f"\n✓ Diffusion model training complete")
        print(f"  Model saved to: {DIFFUSION_DIR}")
        print(f"  Training plots saved to: {eval_dir}")
    else:
        print("\n[SKIP] Diffusion training skipped (using pre-trained models)")
    
    # ========================================
    # STEP 6: Generate Peptides
    # ========================================
    generated_sequences = []  
    bbbp_scores = []          
    
    if not skip_generation:
        print("\n" + "=" * 70)
        print("PHASE 4: PEPTIDE GENERATION")
        print("=" * 70)
        
        from diffusion_model import load_diffusion_model
        from generation import generate_all_lengths
        
        diffusion_model, diffusion = load_diffusion_model()
        
        generated = generate_all_lengths(
            diffusion_model, diffusion,
            num_per_length=NUM_PEPTIDES_PER_LENGTH,
            min_len=MIN_SEQ_LEN,
            max_len=MAX_SEQ_LEN
        )
        
        for length, peptides in generated.items():
            generated_sequences.extend(peptides)
        
        total_generated = sum(len(peps) for peps in generated.values())
        print(f"\n✓ Peptide generation complete")
        print(f"  Total peptides generated: {total_generated}")
        print(f"  Saved to: {GENERATED_DIR}")
    else:
        print("\n[SKIP] Peptide generation skipped")
    
    # ========================================
    # STEP 7: Validate Generated Peptides
    # ========================================
    if not skip_validation and not skip_generation:
        print("\n" + "=" * 70)
        print("PHASE 5: VALIDATION")
        print("=" * 70)
        
        from validation import validate_all_generated
        
        stats, valid_peptides, all_scores = validate_all_generated(
            threshold=BBBP_THRESHOLD,
            return_scores=True  
        )
        bbbp_scores = all_scores  
        
        print(f"\n✓ Validation complete")
        print(f"  Valid BBBP+ peptides: {len(valid_peptides)}")
        print(f"  Acceptance rate: {stats['valid']/stats['total']*100:.1f}%")
    else:
        print("\n[SKIP] Validation skipped")
    
    # ========================================
    # STEP 8: Diffusion Model Evaluation
    # ========================================
    if not skip_generation and generated_sequences and bbbp_scores:
        print("\n" + "=" * 70)
        print("PHASE 6: DIFFUSION MODEL EVALUATION")
        print("=" * 70)
        
        from diffusion_evaluation import evaluate_diffusion_model
        
        eval_dir = os.path.join(OUTPUT_DIR, 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)
        
        metrics = evaluate_diffusion_model(
            generated_seqs=generated_sequences,
            training_seqs=training_sequences,
            bbbp_scores=bbbp_scores,
            threshold=BBBP_THRESHOLD,
            save_dir=eval_dir
        )
        
        import json
        metrics_path = os.path.join(eval_dir, 'diffusion_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Evaluation metrics saved to: {metrics_path}")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput files:")
    print(f"  Classifier: {CLASSIFIER_DIR}/")
    print(f"    - biovec.model")
    print(f"    - classifier.pth")
    print(f"    - scaler.pkl")
    print(f"  Diffusion: {DIFFUSION_DIR}/")
    print(f"    - diffusion_model.pth")
    print(f"  Generated: {GENERATED_DIR}/")
    print(f"    - len_5.csv ... len_20.csv")
    print(f"    - all_generated.csv")
    print(f"    - validated_bbbp_peptides.csv")
    print(f"  Evaluation: {OUTPUT_DIR}/evaluation/")  
    print(f"    - diffusion_training_curves.png")
    print(f"    - aa_composition.png")
    print(f"    - bbbp_distribution.png")
    print(f"    - length_distribution.png")
    print(f"    - diffusion_metrics.json")


if __name__ == "__main__":
    run_full_pipeline()