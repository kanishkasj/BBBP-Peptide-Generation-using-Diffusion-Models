"""
Utility functions for the Peptide BBBP Generation System
"""

import numpy as np
import torch
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from config import RANDOM_SEED, AA_TO_IDX, IDX_TO_AA, PAD_TOKEN, MAX_SEQ_LEN, AMINO_ACIDS, OUTPUT_DIR

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11


def set_seed(seed=RANDOM_SEED):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def tokenize_sequence(seq, max_len=MAX_SEQ_LEN):
    """
    Convert amino acid sequence to token indices with padding
    
    Args:
        seq: Amino acid sequence string
        max_len: Maximum sequence length for padding
        
    Returns:
        List of token indices
    """
    tokens = [AA_TO_IDX.get(aa, PAD_TOKEN) for aa in seq.upper()]
    # Pad to max_len
    if len(tokens) < max_len:
        tokens = tokens + [PAD_TOKEN] * (max_len - len(tokens))
    return tokens[:max_len]


def detokenize_sequence(tokens):
    """
    Convert token indices back to amino acid sequence
    
    Args:
        tokens: List or tensor of token indices
        
    Returns:
        Amino acid sequence string
    """
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy()
    
    seq = ""
    for t in tokens:
        if t == PAD_TOKEN:
            break
        if t in IDX_TO_AA:
            seq += IDX_TO_AA[t]
    return seq


def is_valid_sequence(seq):
    """
    Check if sequence contains only standard amino acids
    
    Args:
        seq: Amino acid sequence string
        
    Returns:
        Boolean indicating validity
    """
    return all(aa in AMINO_ACIDS for aa in seq.upper())


def create_mask(lengths, max_len=MAX_SEQ_LEN):
    """
    Create attention mask based on sequence lengths
    
    Args:
        lengths: Tensor of sequence lengths
        max_len: Maximum sequence length
        
    Returns:
        Boolean mask tensor (True for valid positions)
    """
    batch_size = lengths.size(0)
    mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
    return mask


def get_ngrams(seq, n=3):
    """
    Generate n-grams from a sequence
    
    Args:
        seq: Amino acid sequence string
        n: n-gram size
        
    Returns:
        List of n-grams
    """
    if len(seq) < n:
        return [seq]
    return [seq[i:i+n] for i in range(len(seq) - n + 1)]


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0, mode='min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for loss or metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']
    return 0, float('inf')


def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    try:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['auc'] = 0.0
        
    return metrics


# ============================================
# PLOTTING FUNCTIONS FOR RESEARCH
# ============================================

def plot_training_curves(history, save_path=None):
    """
    Plot training and validation loss/metrics curves
    
    Args:
        history: Dictionary containing training history with keys:
                 'train_loss', 'val_loss', 'train_acc', 'val_acc',
                 'train_f1', 'val_f1', 'train_auc', 'val_auc'
        save_path: Path to save the figure (optional)
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot Accuracy
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # Plot F1 Score
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['train_f1'], 'b-', label='Training F1', linewidth=2)
    ax3.plot(epochs, history['val_f1'], 'r-', label='Validation F1', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('Training and Validation F1 Score')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])
    
    # Plot AUC-ROC
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['train_auc'], 'b-', label='Training AUC', linewidth=2)
    ax4.plot(epochs, history['val_auc'], 'r-', label='Validation AUC', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('AUC-ROC')
    ax4.set_title('Training and Validation AUC-ROC')
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Training curves saved to {save_path}")
    
    plt.show()
    return fig


def plot_roc_curve(y_true, y_prob, save_path=None):
    """
    Plot ROC curve with AUC score
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save the figure (optional)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2.5, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    # Fill area under curve
    ax.fill_between(fpr, tpr, alpha=0.3, color='darkorange')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Add threshold annotations
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=100,
               label=f'Optimal threshold = {optimal_threshold:.3f}', zorder=5)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"ROC curve saved to {save_path}")
    
    plt.show()
    return fig, roc_auc


def plot_precision_recall_curve(y_true, y_prob, save_path=None):
    """
    Plot Precision-Recall curve
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save the figure (optional)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(recall, precision, color='green', lw=2.5,
            label=f'PR curve (AUC = {pr_auc:.4f})')
    
    # Fill area under curve
    ax.fill_between(recall, precision, alpha=0.3, color='green')
    
    # Baseline (proportion of positive class)
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color='navy', linestyle='--', lw=2, 
               label=f'Baseline = {baseline:.3f}')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Precision-Recall curve saved to {save_path}")
    
    plt.show()
    return fig, pr_auc


def plot_confusion_matrix(y_true, y_pred, save_path=None, normalize=False):
    """
    Plot confusion matrix heatmap
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the figure (optional)
        normalize: Whether to normalize the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=['Non-BBBP', 'BBBP'],
                yticklabels=['Non-BBBP', 'BBBP'],
                annot_kws={'size': 16},
                ax=ax)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    return fig, cm


def plot_metrics_comparison(metrics_dict, save_path=None):
    """
    Plot bar chart comparing different metrics
    
    Args:
        metrics_dict: Dictionary with metric names as keys and values
        save_path: Path to save the figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    
    bars = ax.bar(metrics, values, color=colors[:len(metrics)], edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylim([0, 1.15])
    ax.set_ylabel('Score')
    ax.set_title('BBBP Classifier Performance Metrics')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Metrics comparison saved to {save_path}")
    
    plt.show()
    return fig


def plot_all_evaluation_figures(y_true, y_pred, y_prob, history=None, save_dir=None):
    """
    Generate all evaluation plots for research publication
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        history: Training history dictionary (optional)
        save_dir: Directory to save all figures (optional)
        
    Returns:
        Dictionary containing all figures
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    figures = {}
    
    print("\n" + "=" * 50)
    print("GENERATING EVALUATION PLOTS FOR RESEARCH")
    print("=" * 50)
    
    # 1. Training curves (if history provided)
    if history is not None:
        save_path = os.path.join(save_dir, 'training_curves.png') if save_dir else None
        figures['training_curves'] = plot_training_curves(history, save_path)
    
    # 2. ROC Curve
    save_path = os.path.join(save_dir, 'roc_curve.png') if save_dir else None
    fig_roc, roc_auc = plot_roc_curve(y_true, y_prob, save_path)
    figures['roc_curve'] = fig_roc
    
    # 3. Precision-Recall Curve
    save_path = os.path.join(save_dir, 'precision_recall_curve.png') if save_dir else None
    fig_pr, pr_auc = plot_precision_recall_curve(y_true, y_prob, save_path)
    figures['pr_curve'] = fig_pr
    
    # 4. Confusion Matrix
    save_path = os.path.join(save_dir, 'confusion_matrix.png') if save_dir else None
    fig_cm, cm = plot_confusion_matrix(y_true, y_pred, save_path)
    figures['confusion_matrix'] = fig_cm
    
    # 5. Normalized Confusion Matrix
    save_path = os.path.join(save_dir, 'confusion_matrix_normalized.png') if save_dir else None
    fig_cm_norm, cm_norm = plot_confusion_matrix(y_true, y_pred, save_path, normalize=True)
    figures['confusion_matrix_normalized'] = fig_cm_norm
    
    # 6. Metrics Comparison Bar Chart
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    metrics_display = {
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1 Score': metrics['f1'],
        'AUC-ROC': metrics['auc']
    }
    save_path = os.path.join(save_dir, 'metrics_comparison.png') if save_dir else None
    figures['metrics_comparison'] = plot_metrics_comparison(metrics_display, save_path)
    
    print("\n" + "=" * 50)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    if save_dir:
        print(f"Figures saved to: {save_dir}")
    print("=" * 50)
    
    return figures

