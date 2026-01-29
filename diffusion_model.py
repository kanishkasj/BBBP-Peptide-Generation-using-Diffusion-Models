import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import (
    DEVICE, NUM_TOKENS, MAX_SEQ_LEN, EMBEDDING_DIM, AMINO_ACIDS,
    DIFFUSION_TIMESTEPS, DIFFUSION_EMBED_DIM, DIFFUSION_NUM_HEADS,
    DIFFUSION_NUM_LAYERS, DIFFUSION_FF_DIM, DIFFUSION_DROPOUT,
    DIFFUSION_LR, DIFFUSION_EPOCHS, DIFFUSION_PATIENCE, 
    DIFFUSION_DIR, BATCH_SIZE, MIN_SEQ_LEN, AA_TO_IDX, PAD_TOKEN
)
from utils import set_seed, EarlyStopping, tokenize_sequence, detokenize_sequence
import os


# ============================================
# TIMESTEP & POSITION EMBEDDINGS
# ============================================

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional embeddings for timesteps"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, timesteps):
        """
        Args:
            timesteps: (batch,) tensor of timesteps
            
        Returns:
            (batch, dim) embedding
        """
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence positions"""
    
    def __init__(self, d_model, max_len=MAX_SEQ_LEN):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """Add positional encoding to input"""
        return x + self.pe[:, :x.size(1), :]


# ============================================
# TRANSFORMER ENCODER BLOCK
# ============================================

class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block"""
    
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) - True for positions to mask
        """
        # Self attention with residual
        attn_mask = None
        if mask is not None:
            attn_mask = mask.unsqueeze(1).expand(-1, x.size(1), -1)
            
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


# ============================================
# DIFFUSION MODEL ARCHITECTURE
# ============================================

class PeptideDiffusionModel(nn.Module):
    """
    Discrete Diffusion Model for Peptide Generation
    
    Architecture:
        Token Embedding + Position Encoding
              ↓
        Timestep Embedding → Add
              ↓
        Length Embedding → Add
              ↓
        BBBP Label Embedding → Add
              ↓
        Transformer Encoder (N layers)
              ↓
        Token Logits (20 amino acids)
    """
    
    def __init__(self, 
                 num_tokens=NUM_TOKENS,
                 max_seq_len=MAX_SEQ_LEN,
                 embed_dim=DIFFUSION_EMBED_DIM,
                 num_heads=DIFFUSION_NUM_HEADS,
                 num_layers=DIFFUSION_NUM_LAYERS,
                 ff_dim=DIFFUSION_FF_DIM,
                 dropout=DIFFUSION_DROPOUT):
        super().__init__()
        
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
        # Token embedding (21 tokens: 20 AA + 1 PAD)
        self.token_embedding = nn.Embedding(num_tokens, embed_dim, padding_idx=0)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Length embedding (lengths 5-20)
        self.length_embedding = nn.Embedding(max_seq_len + 1, embed_dim)
        
        # BBBP label embedding (0 or 1)
        self.label_embedding = nn.Embedding(2, embed_dim)
        
        # Condition projection (to combine conditions)
        self.condition_proj = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU()
        )
        
        # Transformer encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection to token logits
        self.output_proj = nn.Linear(embed_dim, num_tokens - 1)  # Predict 20 AAs (no PAD)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, noisy_tokens, timesteps, lengths, labels, mask=None):
        """
        Args:
            noisy_tokens: (batch, seq_len) - noisy token indices
            timesteps: (batch,) - diffusion timesteps
            lengths: (batch,) - sequence lengths
            labels: (batch,) - BBBP labels (always 1 for generation)
            mask: (batch, seq_len) - padding mask
            
        Returns:
            logits: (batch, seq_len, 20) - predicted token logits
        """
        batch_size = noisy_tokens.size(0)
        seq_len = noisy_tokens.size(1)
        
        # Token embedding
        x = self.token_embedding(noisy_tokens)  # (batch, seq_len, embed_dim)
        x = self.pos_encoding(x)
        
        # Get condition embeddings
        time_emb = self.time_embed(timesteps)  # (batch, embed_dim)
        length_emb = self.length_embedding(lengths)  # (batch, embed_dim)
        label_emb = self.label_embedding(labels)  # (batch, embed_dim)
        
        # Combine conditions
        cond = torch.cat([time_emb, length_emb, label_emb], dim=-1)
        cond = self.condition_proj(cond)  # (batch, embed_dim)
        
        # Add condition to all positions
        x = x + cond.unsqueeze(1)
        
        # Transformer encoding
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # Output logits
        logits = self.output_proj(x)  # (batch, seq_len, 20)
        
        return logits


# ============================================
# DISCRETE DIFFUSION PROCESS
# ============================================

class DiscreteDiffusion:
    """
    Discrete diffusion process for token sequences
    
    Forward process: Add noise by randomly replacing tokens
    Reverse process: Predict original tokens from noisy sequence
    """
    
    def __init__(self, num_tokens=20, timesteps=DIFFUSION_TIMESTEPS):
        self.num_tokens = num_tokens  # 20 amino acids (excluding PAD)
        self.timesteps = timesteps
        
        # Define noise schedule (linear)
        # At t=0: no noise, at t=T: fully noisy (uniform random tokens)
        self.betas = torch.linspace(0.0001, 0.02, timesteps)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def get_noise_schedule(self, t, device):
        """Get alpha_cumprod for timestep t"""
        return self.alpha_cumprod[t].to(device)
    
    def add_noise(self, tokens, t, device):
        batch_size, seq_len = tokens.shape
        
        corruption_probs = 1 - self.alpha_cumprod[t].to(device)
        
        # Create mask for which tokens to corrupt
        noise_mask = torch.rand(batch_size, seq_len, device=device) < corruption_probs.unsqueeze(1)
        
        noise_mask = noise_mask & (tokens != 0)
        
        random_tokens = torch.randint(1, self.num_tokens + 1, (batch_size, seq_len), device=device)
        noisy_tokens = torch.where(noise_mask, random_tokens, tokens)
        
        return noisy_tokens
    
    def sample_timesteps(self, batch_size, device):
        """Sample random timesteps for training"""
        return torch.randint(0, self.timesteps, (batch_size,), device=device)


# ============================================
# DIFFUSION DATASET
# ============================================

class DiffusionDataset(Dataset):
    """Dataset for diffusion model training (BBBP+ sequences only)"""
    
    def __init__(self, sequences, labels=None, max_len=MAX_SEQ_LEN):
        self.sequences = sequences
        self.labels = labels if labels is not None else [1] * len(sequences)
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        tokens = tokenize_sequence(seq, self.max_len)
        length = len(seq)
        label = self.labels[idx]
        
        # Create padding mask (True for padded positions)
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[length:] = True
        
        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'length': torch.tensor(length, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'mask': mask,
            'sequence': seq
        }


# ============================================
# TRAINING FUNCTIONS
# ============================================

def train_diffusion_model(bbbp_sequences, bbbp_labels=None, tracker=None):
    """
    Train the diffusion model on BBBP positive sequences
    
    Args:
        bbbp_sequences: List of BBBP+ peptide sequences
        bbbp_labels: Labels (optional, defaults to all 1s)
        
    Returns:
        Trained diffusion model
    """
    print("\n" + "=" * 60)
    print("STEP 6: Training Diffusion Model")
    print("=" * 60)
    
    set_seed()
    
    # Create dataset
    dataset = DiffusionDataset(bbbp_sequences, bbbp_labels)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    print(f"\nTraining on {len(dataset)} BBBP+ sequences")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Initialize model and diffusion process
    model = PeptideDiffusionModel(
        num_tokens=NUM_TOKENS,
        max_seq_len=MAX_SEQ_LEN,
        embed_dim=DIFFUSION_EMBED_DIM,
        num_heads=DIFFUSION_NUM_HEADS,
        num_layers=DIFFUSION_NUM_LAYERS,
        ff_dim=DIFFUSION_FF_DIM,
        dropout=DIFFUSION_DROPOUT
    ).to(DEVICE)
    
    diffusion = DiscreteDiffusion(num_tokens=20, timesteps=DIFFUSION_TIMESTEPS)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD token
    optimizer = torch.optim.AdamW(model.parameters(), lr=DIFFUSION_LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=DIFFUSION_EPOCHS, eta_min=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=DIFFUSION_PATIENCE, mode='min')
    
    # Training loop
    print(f"\nTraining for max {DIFFUSION_EPOCHS} epochs...")
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(DIFFUSION_EPOCHS):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{DIFFUSION_EPOCHS}")
        for batch in pbar:
            tokens = batch['tokens'].to(DEVICE)  # (batch, seq_len)
            lengths = batch['length'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            masks = batch['mask'].to(DEVICE)
            
            batch_size = tokens.size(0)
            
            # Sample timesteps
            t = diffusion.sample_timesteps(batch_size, DEVICE)
            
            # Add noise
            noisy_tokens = diffusion.add_noise(tokens, t, DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(noisy_tokens, t, lengths, labels, masks)
            
            # Compute loss (predict original tokens)
            # Shift tokens for target (tokens are 1-20, logits are 0-19)
            target = tokens.clone()
            target[tokens == 0] = 0  # Keep PAD as ignore
            target[tokens > 0] -= 1  # Shift 1-20 to 0-19
            
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target.view(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss /= num_batches
        scheduler.step()
        if tracker is not None:
            tracker.update(
                epoch=epoch + 1,
                loss=epoch_loss,
                lr=scheduler.get_last_lr()[0]
            )
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch {epoch + 1}: Loss = {epoch_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_state = model.state_dict().copy()
        
        # Early stopping
        if early_stopping(epoch_loss):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save model
    os.makedirs(DIFFUSION_DIR, exist_ok=True)
    model_path = os.path.join(DIFFUSION_DIR, "diffusion_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_tokens': NUM_TOKENS,
        'max_seq_len': MAX_SEQ_LEN,
        'embed_dim': DIFFUSION_EMBED_DIM,
        'num_heads': DIFFUSION_NUM_HEADS,
        'num_layers': DIFFUSION_NUM_LAYERS,
        'ff_dim': DIFFUSION_FF_DIM,
        'dropout': DIFFUSION_DROPOUT,
        'timesteps': DIFFUSION_TIMESTEPS
    }, model_path)
    print(f"\nDiffusion model saved to {model_path}")
    
    return model, diffusion


def load_diffusion_model():
    """Load trained diffusion model"""
    model_path = os.path.join(DIFFUSION_DIR, "diffusion_model.pth")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    model = PeptideDiffusionModel(
        num_tokens=checkpoint['num_tokens'],
        max_seq_len=checkpoint['max_seq_len'],
        embed_dim=checkpoint['embed_dim'],
        num_heads=checkpoint['num_heads'],
        num_layers=checkpoint['num_layers'],
        ff_dim=checkpoint['ff_dim'],
        dropout=checkpoint['dropout']
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    diffusion = DiscreteDiffusion(num_tokens=20, timesteps=checkpoint['timesteps'])
    
    print(f"Diffusion model loaded from {model_path}")
    
    return model, diffusion


if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data, get_bbbp_positive_data
    
    # Load data
    df = load_and_preprocess_data()
    bbbp_df = get_bbbp_positive_data(df)
    
    # Train diffusion model
    model, diffusion = train_diffusion_model(
        bbbp_df['seq'].tolist(),
        bbbp_df['label'].tolist()
    )
    
    print("\nDiffusion model training complete!")
