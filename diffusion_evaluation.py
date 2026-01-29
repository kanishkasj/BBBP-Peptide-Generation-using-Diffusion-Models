"""
Diffusion Model Evaluation Metrics and Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.spatial.distance import pdist
import pandas as pd
import os

from config import AMINO_ACIDS, DIFFUSION_DIR


# ============================================
# TRAINING METRICS
# ============================================

class DiffusionTrainingTracker:
    """Track and plot training metrics"""
    
    def __init__(self):
        self.losses = []
        self.token_accuracies = []
        self.learning_rates = []
        self.epochs = []
        
    def update(self, epoch, loss, token_acc=None, lr=None):
        self.epochs.append(epoch)
        self.losses.append(loss)
        if token_acc is not None:
            self.token_accuracies.append(token_acc)
        if lr is not None:
            self.learning_rates.append(lr)
    
    def plot_training_curves(self, save_path=None):
        """Plot loss and accuracy curves"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curve
        axes[0].plot(self.epochs, self.losses, 'b-', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Diffusion Training Loss')
        axes[0].grid(True, alpha=0.3)
        
        # Token accuracy (if tracked)
        if self.token_accuracies:
            axes[1].plot(self.epochs, self.token_accuracies, 'g-', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Token Accuracy')
            axes[1].set_title('Denoising Accuracy')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


# ============================================
# GENERATION QUALITY METRICS
# ============================================

def compute_validity(sequences):
    """Check if all characters are valid amino acids"""
    valid_aas = set(AMINO_ACIDS)
    valid_count = sum(1 for seq in sequences if all(aa in valid_aas for aa in seq))
    return valid_count / len(sequences) * 100


def compute_novelty(generated_seqs, training_seqs):
    """Percentage of generated sequences NOT in training set"""
    training_set = set(training_seqs)
    novel_count = sum(1 for seq in generated_seqs if seq not in training_set)
    return novel_count / len(generated_seqs) * 100


def compute_diversity(sequences):
    """
    Compute diversity metrics:
    - Uniqueness: % unique sequences
    - Internal diversity: mean pairwise edit distance
    """
    unique_seqs = set(sequences)
    uniqueness = len(unique_seqs) / len(sequences) * 100
    
    # Sample for edit distance (expensive for large sets)
    sample = list(unique_seqs)[:500]
    if len(sample) > 1:
        # Convert to numeric for pdist
        from Levenshtein import distance as levenshtein_distance
        n = len(sample)
        distances = []
        for i in range(min(n, 100)):
            for j in range(i + 1, min(n, 100)):
                distances.append(levenshtein_distance(sample[i], sample[j]))
        mean_edit_dist = np.mean(distances) if distances else 0
    else:
        mean_edit_dist = 0
    
    return {
        'uniqueness': uniqueness,
        'num_unique': len(unique_seqs),
        'mean_edit_distance': mean_edit_dist
    }


def compute_aa_distribution(sequences):
    """Compute amino acid frequency distribution"""
    all_aas = ''.join(sequences)
    counts = Counter(all_aas)
    total = sum(counts.values())
    return {aa: counts.get(aa, 0) / total for aa in AMINO_ACIDS}


# ============================================
# VISUALIZATION
# ============================================

def plot_aa_composition_comparison(generated_seqs, training_seqs, save_path=None):
    """Compare AA composition: generated vs training"""
    gen_dist = compute_aa_distribution(generated_seqs)
    train_dist = compute_aa_distribution(training_seqs)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(AMINO_ACIDS))
    width = 0.35
    
    ax.bar(x - width/2, [train_dist[aa] for aa in AMINO_ACIDS], 
           width, label='Training', alpha=0.8)
    ax.bar(x + width/2, [gen_dist[aa] for aa in AMINO_ACIDS], 
           width, label='Generated', alpha=0.8)
    
    ax.set_xlabel('Amino Acid')
    ax.set_ylabel('Frequency')
    ax.set_title('Amino Acid Composition: Training vs Generated')
    ax.set_xticks(x)
    ax.set_xticklabels(list(AMINO_ACIDS))
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_bbbp_score_distribution(scores, threshold=0.8, save_path=None):
    """Plot BBBP score distribution of generated peptides"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.hist(scores, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=threshold, color='r', linestyle='--', linewidth=2, 
               label=f'Threshold = {threshold}')
    
    passing = sum(1 for s in scores if s >= threshold)
    ax.set_xlabel('BBBP Probability')
    ax.set_ylabel('Count')
    ax.set_title(f'BBBP Score Distribution\n{passing}/{len(scores)} ({passing/len(scores)*100:.1f}%) pass threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_length_distribution(generated_seqs, training_seqs, save_path=None):
    """Compare length distributions"""
    gen_lengths = [len(s) for s in generated_seqs]
    train_lengths = [len(s) for s in training_seqs]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    bins = range(4, 22)
    ax.hist(train_lengths, bins=bins, alpha=0.6, label='Training', density=True)
    ax.hist(gen_lengths, bins=bins, alpha=0.6, label='Generated', density=True)
    
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Density')
    ax.set_title('Length Distribution: Training vs Generated')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================
# COMPREHENSIVE EVALUATION REPORT
# ============================================

def evaluate_diffusion_model(generated_seqs, training_seqs, bbbp_scores, 
                             threshold=0.8, save_dir=None):
    """
    Run full evaluation and generate report
    
    Returns:
        Dictionary with all metrics
    """
    print("\n" + "=" * 60)
    print("DIFFUSION MODEL EVALUATION")
    print("=" * 60)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Compute metrics
    validity = compute_validity(generated_seqs)
    novelty = compute_novelty(generated_seqs, training_seqs)
    diversity = compute_diversity(generated_seqs)
    
    passing_rate = sum(1 for s in bbbp_scores if s >= threshold) / len(bbbp_scores) * 100
    mean_bbbp = np.mean(bbbp_scores)
    
    # Print report
    print(f"\n📊 Generation Statistics:")
    print(f"   Total generated: {len(generated_seqs)}")
    print(f"   Unique sequences: {diversity['num_unique']}")
    
    print(f"\n✅ Quality Metrics:")
    print(f"   Validity:    {validity:.1f}%")
    print(f"   Novelty:     {novelty:.1f}%")
    print(f"   Uniqueness:  {diversity['uniqueness']:.1f}%")
    print(f"   Mean edit distance: {diversity['mean_edit_distance']:.2f}")
    
    print(f"\n🧬 BBBP Performance:")
    print(f"   Mean BBBP score: {mean_bbbp:.3f}")
    print(f"   Passing rate (≥{threshold}): {passing_rate:.1f}%")
    
    # Generate plots
    if save_dir:
        plot_aa_composition_comparison(
            generated_seqs, training_seqs,
            save_path=os.path.join(save_dir, 'aa_composition.png')
        )
        plot_bbbp_score_distribution(
            bbbp_scores, threshold,
            save_path=os.path.join(save_dir, 'bbbp_distribution.png')
        )
        plot_length_distribution(
            generated_seqs, training_seqs,
            save_path=os.path.join(save_dir, 'length_distribution.png')
        )
        print(f"\n📈 Plots saved to {save_dir}")
    
    return {
        'validity': validity,
        'novelty': novelty,
        'uniqueness': diversity['uniqueness'],
        'mean_edit_distance': diversity['mean_edit_distance'],
        'mean_bbbp_score': mean_bbbp,
        'passing_rate': passing_rate,
        'num_generated': len(generated_seqs),
        'num_unique': diversity['num_unique']
    }