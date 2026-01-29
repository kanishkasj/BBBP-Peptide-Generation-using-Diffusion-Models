import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
import torch
from config import DEVICE, GENERATED_DIR, OUTPUT_DIR, AMINO_ACIDS, MIN_SEQ_LEN, MAX_SEQ_LEN
from diffusion_model import load_diffusion_model
from generation import generate_peptides, is_valid_peptide

EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, 'experiments')
os.makedirs(EXPERIMENT_DIR, exist_ok=True)


# ============================================
# Generate 1000 peptides x 5 runs
# Plot length distribution
# ============================================

def experiment_length_distribution(num_peptides=1000, num_runs=5):
    """
    Generate peptides with RANDOM length conditioning (5-20)
    Analyze: Does model respect the random length? What's the actual distribution?
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Random Length Conditioning Analysis")
    print(f"Generating {num_peptides} peptides x {num_runs} runs")
    print(f"Length randomly sampled from [{MIN_SEQ_LEN}, {MAX_SEQ_LEN}]")
    print("=" * 60)
    
    model, diffusion = load_diffusion_model()
    
    all_runs_data = []
    length_counts_per_run = []
    requested_vs_actual = []  
    
    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")
        
        run_peptides = []
        run_requested_lengths = []
        run_actual_lengths = []
        
        batch_size = 64
        remaining = num_peptides
        
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            
            random_lengths = np.random.randint(MIN_SEQ_LEN, MAX_SEQ_LEN + 1, size=current_batch)
            
            for target_len in np.unique(random_lengths):
                count_this_len = np.sum(random_lengths == target_len)
                
                peptides = generate_peptides(
                    model, diffusion,
                    target_length=int(target_len),
                    num_samples=int(count_this_len),
                    bbbp_label=1,
                    temperature=0.8,
                    top_p=0.9
                )
                
                for pep in peptides:
                    if is_valid_peptide(pep):
                        run_peptides.append(pep)
                        run_requested_lengths.append(int(target_len))
                        run_actual_lengths.append(len(pep))
            
            remaining -= current_batch
        
        run_peptides = run_peptides[:num_peptides]
        run_requested_lengths = run_requested_lengths[:num_peptides]
        run_actual_lengths = run_actual_lengths[:num_peptides]

        # Count actual lengths for this run
        length_counter = Counter(run_actual_lengths)
        length_counts_per_run.append(length_counter)
        
        # Track requested vs actual
        matches = sum(1 for req, act in zip(run_requested_lengths, run_actual_lengths) if req == act)
        match_rate = matches / len(run_peptides) * 100 if run_peptides else 0
        
        # Store data
        for pep, req_len, act_len in zip(run_peptides, run_requested_lengths, run_actual_lengths):
            all_runs_data.append({
                'run': run + 1,
                'sequence': pep,
                'requested_length': req_len,
                'actual_length': act_len,
                'length_match': req_len == act_len
            })
            requested_vs_actual.append({
                'run': run + 1,
                'requested': req_len,
                'actual': act_len
            })
        
        print(f"  Generated {len(run_peptides)} valid peptides")
        print(f"  Requested length range: {min(run_requested_lengths)} - {max(run_requested_lengths)}")
        print(f"  Actual length range: {min(run_actual_lengths)} - {max(run_actual_lengths)}")
        print(f"  Length match rate: {match_rate:.1f}%")
    
    # Create DataFrame
    df = pd.DataFrame(all_runs_data)
    
    # Save data
    csv_path = os.path.join(EXPERIMENT_DIR, 'exp1_length_distribution.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nData saved to {csv_path}")
    
    # Plot results
    plot_length_distribution_experiment(df, length_counts_per_run, num_runs)
    
    return df, length_counts_per_run


def plot_length_distribution_experiment(df, length_counts_per_run, num_runs):
    """Plot length distribution across runs with requested vs actual analysis"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    lengths = list(range(MIN_SEQ_LEN, MAX_SEQ_LEN + 1))
    
    # Plot 1: Requested vs Actual Length (scatter with jitter)
    ax1 = axes[0, 0]
    jitter_req = df['requested_length'] + np.random.normal(0, 0.1, len(df))
    jitter_act = df['actual_length'] + np.random.normal(0, 0.1, len(df))
    ax1.scatter(jitter_req, jitter_act, alpha=0.3, s=10)
    ax1.plot([MIN_SEQ_LEN, MAX_SEQ_LEN], [MIN_SEQ_LEN, MAX_SEQ_LEN], 
             'r--', linewidth=2, label='Perfect match')
    ax1.set_xlabel('Requested Length')
    ax1.set_ylabel('Actual Length')
    ax1.set_title('Requested vs Actual Length')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Match rate per requested length
    ax2 = axes[0, 1]
    match_rates = []
    for length in lengths:
        subset = df[df['requested_length'] == length]
        if len(subset) > 0:
            rate = subset['length_match'].mean() * 100
        else:
            rate = 0
        match_rates.append(rate)
    
    colors = ['green' if r > 80 else 'orange' if r > 50 else 'red' for r in match_rates]
    ax2.bar(lengths, match_rates, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=100, color='green', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Requested Length')
    ax2.set_ylabel('Match Rate (%)')
    ax2.set_title('Length Conditioning Accuracy')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Actual length distribution (histogram)
    ax3 = axes[0, 2]
    for run in range(1, num_runs + 1):
        run_data = df[df['run'] == run]['actual_length']
        ax3.hist(run_data, bins=range(MIN_SEQ_LEN, MAX_SEQ_LEN + 2), 
                 alpha=0.5, label=f'Run {run}', edgecolor='black')
    ax3.set_xlabel('Actual Peptide Length')
    ax3.set_ylabel('Count')
    ax3.set_title('Actual Length Distribution Across Runs')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Requested vs Actual distribution comparison
    ax4 = axes[1, 0]
    req_counts = Counter(df['requested_length'])
    act_counts = Counter(df['actual_length'])
    
    x = np.arange(len(lengths))
    width = 0.35
    ax4.bar(x - width/2, [req_counts.get(l, 0) for l in lengths], width, 
            label='Requested', alpha=0.8)
    ax4.bar(x + width/2, [act_counts.get(l, 0) for l in lengths], width,
            label='Actual', alpha=0.8)
    ax4.set_xlabel('Length')
    ax4.set_ylabel('Count')
    ax4.set_title('Requested vs Actual Length Distribution')
    ax4.set_xticks(x)
    ax4.set_xticklabels(lengths)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Mean ± std of actual lengths across runs
    ax5 = axes[1, 1]
    means = [np.mean([lc.get(l, 0) for lc in length_counts_per_run]) for l in lengths]
    stds = [np.std([lc.get(l, 0) for lc in length_counts_per_run]) for l in lengths]
    
    ax5.bar(lengths, means, yerr=stds, capsize=3, alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Actual Peptide Length')
    ax5.set_ylabel('Mean Count ± Std')
    ax5.set_title(f'Actual Length Distribution ({num_runs} runs)')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Confusion matrix style - requested vs actual
    ax6 = axes[1, 2]
    confusion = np.zeros((len(lengths), len(lengths)))
    for _, row in df.iterrows():
        req_idx = row['requested_length'] - MIN_SEQ_LEN
        act_idx = row['actual_length'] - MIN_SEQ_LEN
        if 0 <= req_idx < len(lengths) and 0 <= act_idx < len(lengths):
            confusion[req_idx, act_idx] += 1
    
    # Normalize by row (requested)
    confusion_norm = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-8)
    
    im = ax6.imshow(confusion_norm, cmap='Blues', aspect='auto')
    ax6.set_xticks(range(len(lengths)))
    ax6.set_xticklabels(lengths)
    ax6.set_yticks(range(len(lengths)))
    ax6.set_yticklabels(lengths)
    ax6.set_xlabel('Actual Length')
    ax6.set_ylabel('Requested Length')
    ax6.set_title('Length Conditioning Confusion Matrix')
    plt.colorbar(im, ax=ax6, label='Proportion')
    
    plt.tight_layout()
    
    plot_path = os.path.join(EXPERIMENT_DIR, 'exp1_length_distribution.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {plot_path}")
    
    # Print summary statistics
    overall_match = df['length_match'].mean() * 100
    print(f"\nSummary:")
    print(f"   Overall length match rate: {overall_match:.1f}%")
    print(f"   Total peptides generated: {len(df)}")


# ============================================
# Generate with length = 0
# ============================================

def experiment_length_zero(num_samples=100):
    """
    Test edge case: What happens when target_length = 0?
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Edge Case - Length = 0")
    print("=" * 60)
    
    model, diffusion = load_diffusion_model()
    
    print(f"\nAttempting to generate {num_samples} peptides with length=0...")
    
    results = {
        'generated_sequences': [],
        'actual_lengths': [],
        'all_padding': [],
        'error': None
    }
    
    try:
        peptides = generate_peptides(
            model, diffusion,
            target_length=0,  # Edge case!
            num_samples=num_samples,
            bbbp_label=1,
            temperature=0.8,
            top_p=0.9
        )
        
        for pep in peptides:
            results['generated_sequences'].append(pep)
            results['actual_lengths'].append(len(pep))
            results['all_padding'].append(len(pep) == 0)
        
        print(f"\n✓ Generation completed (no crash)")
        print(f"  Sequences returned: {len(peptides)}")
        print(f"  Empty sequences: {sum(results['all_padding'])}")
        print(f"  Non-empty sequences: {len(peptides) - sum(results['all_padding'])}")
        
        if peptides and any(len(p) > 0 for p in peptides):
            non_empty = [p for p in peptides if len(p) > 0]
            print(f"  Example non-empty outputs: {non_empty[:5]}")
            print(f"  Actual lengths: {[len(p) for p in non_empty[:10]]}")
        
    except Exception as e:
        results['error'] = str(e)
        print(f"\n✗ Error occurred: {e}")
    
    # Save results
    report = {
        'experiment': 'length_zero',
        'num_samples': num_samples,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to text file
    report_path = os.path.join(EXPERIMENT_DIR, 'exp2_length_zero_report.txt')
    with open(report_path, 'w') as f:
        f.write("EXPERIMENT 2: Length = 0 Edge Case\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Samples attempted: {num_samples}\n")
        f.write(f"Error: {results['error']}\n")
        f.write(f"Sequences returned: {len(results['generated_sequences'])}\n")
        f.write(f"Empty sequences: {sum(results['all_padding'])}\n")
        if results['generated_sequences']:
            f.write(f"\nSample outputs:\n")
            for i, seq in enumerate(results['generated_sequences'][:20]):
                f.write(f"  {i+1}. '{seq}' (len={len(seq)})\n")
    
    print(f"\nReport saved to {report_path}")
    
    return results


# ============================================
# Generate with label = 0
# Non-BBBP conditioning
# ============================================

def experiment_label_zero(num_peptides=500, lengths_to_test=[8, 12, 16]):
    """
    Generate peptides with BBBP label = 0 (non-BBBP)
    Compare with label = 1 generation
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Label = 0 (Non-BBBP) Conditioning")
    print("=" * 60)
    
    model, diffusion = load_diffusion_model()
    
    results = {
        'label_0': {'sequences': [], 'lengths': []},
        'label_1': {'sequences': [], 'lengths': []}
    }
    
    for label in [0, 1]:
        label_name = "Non-BBBP (0)" if label == 0 else "BBBP+ (1)"
        print(f"\n--- Generating with label = {label} ({label_name}) ---")
        
        for target_len in lengths_to_test:
            print(f"  Length {target_len}...")
            
            peptides = generate_peptides(
                model, diffusion,
                target_length=target_len,
                num_samples=num_peptides // len(lengths_to_test),
                bbbp_label=label,  # Key difference!
                temperature=0.8,
                top_p=0.9
            )
            
            for pep in peptides:
                if is_valid_peptide(pep):
                    results[f'label_{label}']['sequences'].append(pep)
                    results[f'label_{label}']['lengths'].append(len(pep))
        
        print(f"  Total generated: {len(results[f'label_{label}']['sequences'])}")
    
    # Analyze and compare
    plot_label_comparison(results, lengths_to_test)
    
    # Save sequences
    for label in [0, 1]:
        df = pd.DataFrame({
            'sequence': results[f'label_{label}']['sequences'],
            'length': results[f'label_{label}']['lengths'],
            'label': label
        })
        csv_path = os.path.join(EXPERIMENT_DIR, f'exp3_label_{label}_peptides.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved to {csv_path}")
    
    return results


def plot_label_comparison(results, lengths_to_test):
    """Compare AA composition between label=0 and label=1"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: AA composition comparison
    ax1 = axes[0, 0]
    
    def get_aa_freq(sequences):
        all_aa = ''.join(sequences)
        counts = Counter(all_aa)
        total = sum(counts.values())
        return {aa: counts.get(aa, 0) / total if total > 0 else 0 for aa in AMINO_ACIDS}
    
    freq_0 = get_aa_freq(results['label_0']['sequences'])
    freq_1 = get_aa_freq(results['label_1']['sequences'])
    
    x = np.arange(len(AMINO_ACIDS))
    width = 0.35
    
    ax1.bar(x - width/2, [freq_0[aa] for aa in AMINO_ACIDS], width, 
            label='Label=0 (Non-BBBP)', alpha=0.8)
    ax1.bar(x + width/2, [freq_1[aa] for aa in AMINO_ACIDS], width,
            label='Label=1 (BBBP+)', alpha=0.8)
    
    ax1.set_xlabel('Amino Acid')
    ax1.set_ylabel('Frequency')
    ax1.set_title('AA Composition: Label=0 vs Label=1')
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(AMINO_ACIDS))
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Frequency difference
    ax2 = axes[0, 1]
    diff = [freq_1[aa] - freq_0[aa] for aa in AMINO_ACIDS]
    colors = ['green' if d > 0 else 'red' for d in diff]
    
    ax2.bar(x, diff, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Amino Acid')
    ax2.set_ylabel('Frequency Difference (Label1 - Label0)')
    ax2.set_title('AA Enrichment in BBBP+ vs Non-BBBP')
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(AMINO_ACIDS))
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Length distribution comparison
    ax3 = axes[1, 0]
    ax3.hist(results['label_0']['lengths'], bins=range(MIN_SEQ_LEN, MAX_SEQ_LEN + 2),
             alpha=0.6, label='Label=0', edgecolor='black')
    ax3.hist(results['label_1']['lengths'], bins=range(MIN_SEQ_LEN, MAX_SEQ_LEN + 2),
             alpha=0.6, label='Label=1', edgecolor='black')
    ax3.set_xlabel('Peptide Length')
    ax3.set_ylabel('Count')
    ax3.set_title('Length Distribution by Label')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Unique sequences comparison
    ax4 = axes[1, 1]
    
    unique_0 = len(set(results['label_0']['sequences']))
    unique_1 = len(set(results['label_1']['sequences']))
    total_0 = len(results['label_0']['sequences'])
    total_1 = len(results['label_1']['sequences'])
    
    # Check overlap
    overlap = len(set(results['label_0']['sequences']) & set(results['label_1']['sequences']))
    
    categories = ['Total', 'Unique', 'Overlap']
    label_0_vals = [total_0, unique_0, overlap]
    label_1_vals = [total_1, unique_1, overlap]
    
    x_cat = np.arange(len(categories))
    ax4.bar(x_cat - width/2, label_0_vals, width, label='Label=0', alpha=0.8)
    ax4.bar(x_cat + width/2, label_1_vals, width, label='Label=1', alpha=0.8)
    ax4.set_xlabel('Metric')
    ax4.set_ylabel('Count')
    ax4.set_title('Generation Statistics')
    ax4.set_xticks(x_cat)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add text annotations
    for i, v in enumerate(label_0_vals):
        ax4.text(i - width/2, v + 5, str(v), ha='center', fontsize=9)
    for i, v in enumerate(label_1_vals):
        ax4.text(i + width/2, v + 5, str(v), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    plot_path = os.path.join(EXPERIMENT_DIR, 'exp3_label_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {plot_path}")


# ============================================
# MAIN: Run all experiments
# ============================================

def run_all_experiments():
    """Run all three experiments"""
    print("\n" + "=" * 70)
    print("DIFFUSION MODEL EXPERIMENTS")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Experiment 1: Length distribution
    print("\n[1/3] Running length distribution experiment...")
    df_exp1, counts_exp1 = experiment_length_distribution(
        num_peptides=1000, 
        num_runs=5
    )
    
    # Experiment 2: Length = 0
    print("\n[2/3] Running length=0 edge case experiment...")
    results_exp2 = experiment_length_zero(num_samples=100)
    
    # Experiment 3: Label = 0
    print("\n[3/3] Running label=0 conditioning experiment...")
    results_exp3 = experiment_label_zero(
        num_peptides=500,
        lengths_to_test=[8, 12, 16]
    )
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {EXPERIMENT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diffusion Model Experiments")
    parser.add_argument('--exp', type=int, choices=[1, 2, 3], 
                        help='Run specific experiment (1, 2, or 3)')
    parser.add_argument('--all', action='store_true', 
                        help='Run all experiments')
    
    args = parser.parse_args()
    
    if args.all or args.exp is None:
        run_all_experiments()
    elif args.exp == 1:
        experiment_length_distribution(num_peptides=1000, num_runs=5)
    elif args.exp == 2:
        experiment_length_zero(num_samples=100)
    elif args.exp == 3:
        experiment_label_zero(num_peptides=500, lengths_to_test=[8, 12, 16])