import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import (
    DEVICE, NUM_TOKENS, MAX_SEQ_LEN, MIN_SEQ_LEN, DIFFUSION_TIMESTEPS,
    NUM_PEPTIDES_PER_LENGTH, GENERATED_DIR, IDX_TO_AA, AMINO_ACIDS
)
from utils import set_seed, detokenize_sequence
from diffusion_model import load_diffusion_model, DiscreteDiffusion


def generate_peptides(model, diffusion, target_length, num_samples, 
                      bbbp_label=1, temperature=1.0, top_k=0, top_p=0.9):
    """
    Generate peptides of a specific length using reverse diffusion
    
    Args:
        model: Trained diffusion model
        diffusion: DiscreteDiffusion instance
        target_length: Desired peptide length
        num_samples: Number of peptides to generate
        bbbp_label: BBBP label conditioning (1 for BBBP+)
        temperature: Sampling temperature
        top_k: Top-k sampling (0 to disable)
        top_p: Top-p (nucleus) sampling
        
    Returns:
        List of generated peptide sequences
    """
    model.eval()
    device = DEVICE
    
    generated = []
    batch_size = min(64, num_samples)  # Process in batches
    
    for batch_start in tqdm(range(0, num_samples, batch_size), desc=f"Length {target_length}"):
        current_batch_size = min(batch_size, num_samples - batch_start)
        
        # Initialize with random tokens
        tokens = torch.randint(1, 21, (current_batch_size, MAX_SEQ_LEN), device=device)
        
        # Set padding positions to 0
        padding_mask = torch.zeros(current_batch_size, MAX_SEQ_LEN, dtype=torch.bool, device=device)
        padding_mask[:, target_length:] = True
        tokens[padding_mask] = 0
        
        # Conditions
        lengths = torch.full((current_batch_size,), target_length, dtype=torch.long, device=device)
        labels = torch.full((current_batch_size,), bbbp_label, dtype=torch.long, device=device)
        
        # Reverse diffusion process
        timesteps = list(range(DIFFUSION_TIMESTEPS - 1, -1, -1))
        
        for t_idx, t in enumerate(timesteps):
            t_batch = torch.full((current_batch_size,), t, dtype=torch.long, device=device)
            
            with torch.no_grad():
                # Predict token logits
                logits = model(tokens, t_batch, lengths, labels, padding_mask)
                
                # Apply temperature
                logits = logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    threshold = top_k_vals[:, :, -1].unsqueeze(-1)
                    logits[logits < threshold] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, :, 1:] = sorted_indices_to_remove[:, :, :-1].clone()
                    sorted_indices_to_remove[:, :, 0] = 0
                    
                    # Scatter back to original indexing
                    for b in range(current_batch_size):
                        for s in range(MAX_SEQ_LEN):
                            remove_indices = sorted_indices[b, s][sorted_indices_to_remove[b, s]]
                            logits[b, s, remove_indices] = float('-inf')
                
                # Sample from distribution, candidate denoised token sequence for this timestep.
                probs = F.softmax(logits, dim=-1)
                sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
                sampled = sampled.view(current_batch_size, MAX_SEQ_LEN)
                
                # Convert from logit index (0-19) to token index (1-20)
                new_tokens = sampled + 1
                
                # Determine which positions to update based on timestep
                # More positions are "denoised" as t decreases, at the start only few will be denoised !
                alpha_t = diffusion.alpha_cumprod[t].to(device)
                update_prob = 1 - alpha_t
                
                # Create update mask
                update_mask = torch.rand(current_batch_size, MAX_SEQ_LEN, device=device) < update_prob
                
                # Don't update padding positions
                update_mask[padding_mask] = False
                
                # For final steps, force update all non-padding
                if t < 10:
                    update_mask[:, :target_length] = True
                
                # Update tokens
                tokens = torch.where(update_mask, new_tokens, tokens)
                tokens[padding_mask] = 0
        
        # Convert to sequences
        for i in range(current_batch_size):
            seq = detokenize_sequence(tokens[i])
            if len(seq) == target_length:
                generated.append(seq)
    
    return generated


def generate_all_lengths(model, diffusion, num_per_length=NUM_PEPTIDES_PER_LENGTH,
                         min_len=MIN_SEQ_LEN, max_len=MAX_SEQ_LEN):
    """
    Generate peptides for all lengths from min_len to max_len
    
    Args:
        model: Trained diffusion model
        diffusion: DiscreteDiffusion instance
        num_per_length: Number of peptides to generate per length
        min_len: Minimum length (default 5)
        max_len: Maximum length (default 20)
        
    Returns:
        Dictionary mapping length to list of sequences
    """
    print("\n" + "=" * 60)
    print("STEP 7: Generating Peptides")
    print("=" * 60)
    
    set_seed()
    os.makedirs(GENERATED_DIR, exist_ok=True)
    
    all_generated = {}
    
    for length in range(min_len, max_len + 1):
        print(f"\nGenerating {num_per_length} peptides of length {length}...")
        
        # Generate more than needed to account for filtering
        peptides = generate_peptides(
            model, diffusion,
            target_length=length,
            num_samples=int(num_per_length * 1.5),  # Generate extra
            bbbp_label=1,
            temperature=0.8,
            top_p=0.9
        )
        
        # Remove duplicates and filter by length
        unique_peptides = []
        seen = set()
        for pep in peptides:
            if pep not in seen and len(pep) == length and is_valid_peptide(pep):
                seen.add(pep)
                unique_peptides.append(pep)
        
        # Take required number
        final_peptides = unique_peptides[:num_per_length]
        all_generated[length] = final_peptides
        
        print(f"  Generated {len(final_peptides)} unique valid peptides")
        
        # Save to CSV
        df = pd.DataFrame({
            'seq': final_peptides,
            'len': [length] * len(final_peptides),
            'label': [1] * len(final_peptides)  # All generated as BBBP+
        })
        
        csv_path = os.path.join(GENERATED_DIR, f"len_{length}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Saved to {csv_path}")
    
    # Save combined file
    all_peptides = []
    for length, peptides in all_generated.items():
        for pep in peptides:
            all_peptides.append({'seq': pep, 'len': length, 'label': 1})
    
    combined_df = pd.DataFrame(all_peptides)
    combined_path = os.path.join(GENERATED_DIR, "all_generated.csv")
    combined_df.to_csv(combined_path, index=False)
    print(f"\nAll peptides saved to {combined_path}")
    print(f"Total generated: {len(all_peptides)} peptides")
    
    return all_generated


def is_valid_peptide(sequence):
    """Check if sequence contains only valid amino acids"""
    return all(aa in AMINO_ACIDS for aa in sequence.upper())


def generate_single_batch(model, diffusion, length, num_samples, temperature=0.8):
    """
    Generate a single batch of peptides (utility function)
    
    Args:
        model: Diffusion model
        diffusion: Diffusion process
        length: Target length
        num_samples: Number to generate
        temperature: Sampling temperature
        
    Returns:
        List of generated sequences
    """
    model.eval()
    
    peptides = generate_peptides(
        model, diffusion,
        target_length=length,
        num_samples=num_samples,
        bbbp_label=1,
        temperature=temperature,
        top_p=0.9
    )
    
    # Filter valid unique peptides
    unique = list(set([p for p in peptides if len(p) == length and is_valid_peptide(p)]))
    return unique


if __name__ == "__main__":
    # Load model
    print("Loading diffusion model...")
    model, diffusion = load_diffusion_model()
    
    # Generate peptides for all lengths
    generated = generate_all_lengths(model, diffusion)
    
    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    for length in sorted(generated.keys()):
        print(f"  Length {length:2d}: {len(generated[length]):4d} peptides")
        if generated[length]:
            print(f"    Examples: {generated[length][:3]}")