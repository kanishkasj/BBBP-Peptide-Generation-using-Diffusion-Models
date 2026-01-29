# Peptide BBBP Generation System using Diffusion Models

A complete end-to-end system for generating Blood-Brain Barrier Permeable (BBBP) peptides using a custom discrete diffusion model.

## Overview

This system implements:
1. **BBBP Peptide Classifier** - BiLSTM + Attention with BioVec embeddings and iFeature descriptors
2. **Discrete Diffusion Model** - Transformer-based model for generating BBBP-positive peptides
3. **Gradio UI** - Interactive interface for validating peptide sequences
4. **Validation Pipeline** - Automated validation of generated peptides

## Project Structure

```
diffusion_bbbp/
├── config.py                 # Configuration and hyperparameters
├── utils.py                  # Utility functions
├── data_preprocessing.py     # Step 1: Dataset loading and preprocessing
├── ifeature_descriptors.py   # Step 2: iFeature extraction (classifier only)
├── classifier.py             # Step 3 & 4: BioVec + BiLSTM classifier
├── diffusion_model.py        # Step 5: Diffusion model architecture
├── diffusion_evaluation.py   # evaluating the diffusion model!
├── generation.py             # Step 6: Peptide generation
├── validation.py             # Step 7: Validation of generated peptides
├── addn.py                   # implementation with random length of 5 - 20 and plot the dis. and with len =0 and label = 0 
├── train.py                  # Main training pipeline
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── outputs/                  # Generated outputs
    ├── classifier/
    │   ├── biovec.model
    │   ├── classifier.pth
    │   └── scaler.pkl
    ├── diffusion/
    │   └── diffusion_model.pth
    └── generated_peptides/
        ├── len_5.csv ... len_20.csv
        ├── all_generated.csv
        └── validated_bbbp_peptides.csv
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
python train.py
```

### Training Only
```bash
python train.py --train-only
```

### Generation Only (requires pre-trained models)
```bash
python train.py --generate-only
```

### Validation Only
```bash
python train.py --validate-only
```

## Model Architecture

### BBBP Classifier

```
BioVec 3-gram Embeddings (100D)
         ↓
    BiLSTM (128 hidden, 2 layers)
         ↓
    Self-Attention
         ↓
[Concatenate iFeature Descriptors (2060D)]
         ↓
   Fully Connected (128 → 64 → 1)
         ↓
      Sigmoid
```

### Diffusion Model

```
Token Embedding (21 tokens → 256D)
         ↓
  Positional Encoding
         ↓
+ Timestep Embedding (sinusoidal)
+ Length Embedding
+ BBBP Label Embedding (always 1)
         ↓
Transformer Encoder (6 layers, 8 heads)
         ↓
   Token Logits (20 amino acids)
```

## Features

### iFeature Descriptors (Classifier Only)
- **AAC**: Amino Acid Composition (20 features)
- **DPC**: Dipeptide Composition (400 features)
- **CKSAAP**: Composition of k-Spaced AA Pairs (1600 features)
- **PAAC**: Pseudo Amino Acid Composition (30 features)
- **Charge**: Net charge, positive/negative ratios (5 features)
- **Hydrophobicity**: Mean, std, max, min, range (5 features)

Total: 2060 features


## Outputs

### Generated Peptides
- 500 unique peptides per length (5-20)
- Total: 8000 peptides
- CSV format with sequence, length, and BBBP probability

