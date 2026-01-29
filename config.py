"""
Configuration file for Peptide BBBP Generation System
Contains all hyperparameters and settings
"""

import os
import torch

# ============================================
# PATH CONFIGURATIONS
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "new.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Output subdirectories
CLASSIFIER_DIR = os.path.join(OUTPUT_DIR, "classifier")
DIFFUSION_DIR = os.path.join(OUTPUT_DIR, "diffusion")
GENERATED_DIR = os.path.join(OUTPUT_DIR, "generated_peptides")

# Create directories
for dir_path in [CLASSIFIER_DIR, DIFFUSION_DIR, GENERATED_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================
# DEVICE CONFIGURATION
# ============================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================
# RANDOM SEED FOR REPRODUCIBILITY
# ============================================  
RANDOM_SEED = 42

# ============================================
# AMINO ACID TOKENIZATION
# ============================================
# Standard 20 amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAD_TOKEN = 0
AA_TO_IDX = {aa: idx + 1 for idx, aa in enumerate(AMINO_ACIDS)}  # 1-20
IDX_TO_AA = {idx + 1: aa for idx, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA[0] = "<PAD>"
NUM_TOKENS = 21  # 20 amino acids + 1 padding token
MAX_SEQ_LEN = 20
MIN_SEQ_LEN = 5

# ============================================
# DATASET PARAMETERS
# ============================================
TRAIN_VAL_SPLIT = 0.2  # 20% for validation
BATCH_SIZE = 64

# ============================================
# EMBEDDING PARAMETERS
# ============================================
EMBEDDING_DIM = 128  # Token embedding dimension

# ============================================
# BIOVEC PARAMETERS (for classifier)
# ============================================
BIOVEC_NGRAM = 3  
BIOVEC_DIM = 100 
BIOVEC_WINDOW = 5
BIOVEC_MIN_COUNT = 1
BIOVEC_EPOCHS = 100

# ============================================
# CLASSIFIER PARAMETERS
# ============================================
CLASSIFIER_HIDDEN_DIM = 128
CLASSIFIER_NUM_LAYERS = 2
CLASSIFIER_DROPOUT = 0.3
CLASSIFIER_LR = 0.001
CLASSIFIER_EPOCHS = 100
CLASSIFIER_PATIENCE = 15  
# ============================================
# DIFFUSION MODEL PARAMETERS
# ============================================
DIFFUSION_TIMESTEPS = 100  
DIFFUSION_EMBED_DIM = 256  
DIFFUSION_NUM_HEADS = 8
DIFFUSION_NUM_LAYERS = 6
DIFFUSION_FF_DIM = 512
DIFFUSION_DROPOUT = 0.1
DIFFUSION_LR = 0.0001
DIFFUSION_EPOCHS = 200
DIFFUSION_PATIENCE = 20

# ============================================
# GENERATION PARAMETERS
# ============================================
NUM_PEPTIDES_PER_LENGTH = 500
BBBP_THRESHOLD = 0.8  
GRADIO_PORT = 7860
