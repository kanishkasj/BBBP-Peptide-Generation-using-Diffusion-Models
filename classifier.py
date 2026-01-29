import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import (
    BIOVEC_NGRAM, BIOVEC_DIM, BIOVEC_WINDOW, BIOVEC_MIN_COUNT, BIOVEC_EPOCHS,
    CLASSIFIER_HIDDEN_DIM, CLASSIFIER_NUM_LAYERS, CLASSIFIER_DROPOUT,
    CLASSIFIER_LR, CLASSIFIER_EPOCHS, CLASSIFIER_PATIENCE,
    MAX_SEQ_LEN, DEVICE, CLASSIFIER_DIR, BATCH_SIZE
)
from utils import set_seed, EarlyStopping, calculate_metrics, plot_all_evaluation_figures
from ifeature_descriptors import iFeatureExtractor, extract_features_batch


# ============================================
# BIOVEC EMBEDDINGS
# ============================================

class BioVecModel:
    """
    BioVec (ProtVec) model for peptide embeddings
    Uses 3-gram representation of amino acid sequences
    """
    
    def __init__(self, ngram=BIOVEC_NGRAM, dim=BIOVEC_DIM, 
                 window=BIOVEC_WINDOW, min_count=BIOVEC_MIN_COUNT):
        self.ngram = ngram
        self.dim = dim
        self.window = window
        self.min_count = min_count
        self.model = None
        
    def get_ngrams(self, sequence):
        """Generate overlapping n-grams from sequence"""
        sequence = sequence.upper()
        if len(sequence) < self.ngram:
            return [sequence]
        return [sequence[i:i+self.ngram] for i in range(len(sequence) - self.ngram + 1)]
    
    def train(self, sequences, epochs=BIOVEC_EPOCHS):
        """
        Train BioVec model on peptide sequences
        
        Args:
            sequences: List of amino acid sequences
            epochs: Training epochs
        """
        print("\nTraining BioVec model...")
        print(f"  n-gram size: {self.ngram}")
        print(f"  Embedding dimension: {self.dim}")
        print(f"  Training sequences: {len(sequences)}")
        
        # Convert sequences to n-grams
        corpus = [self.get_ngrams(seq) for seq in sequences]
        
        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=corpus,
            vector_size=self.dim,
            window=self.window,
            min_count=self.min_count,
            epochs=epochs,
            workers=4,
            sg=1  
        )
        
        print(f"  Vocabulary size: {len(self.model.wv.key_to_index)}")
        print("  BioVec training complete!")
        
    def embed_sequence(self, sequence):
        """
        Get BioVec embedding for a sequence
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Sequence of n-gram embeddings (L-n+1, dim)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        ngrams = self.get_ngrams(sequence)
        embeddings = []
        
        for gram in ngrams:
            if gram in self.model.wv:
                embeddings.append(self.model.wv[gram])
            else:
                # Use zero vector for unknown n-grams
                embeddings.append(np.zeros(self.dim))
                
        return np.array(embeddings)
    
    def embed_sequence_padded(self, sequence, max_len=MAX_SEQ_LEN):
        """
        Get padded BioVec embedding
        
        Args:
            sequence: Amino acid sequence
            max_len: Maximum length for padding
            
        Returns:
            Padded embedding tensor (max_len, dim)
        """
        embedding = self.embed_sequence(sequence)
        max_ngrams = max_len - self.ngram + 1
        
        # Pad or truncate
        if len(embedding) < max_ngrams:
            padding = np.zeros((max_ngrams - len(embedding), self.dim))
            embedding = np.vstack([embedding, padding])
        else:
            embedding = embedding[:max_ngrams]
            
        return embedding
    
    def save(self, filepath):
        """Save BioVec model"""
        self.model.save(filepath)
        print(f"BioVec model saved to {filepath}")
        
    def load(self, filepath):
        """Load BioVec model"""
        self.model = Word2Vec.load(filepath)
        print(f"BioVec model loaded from {filepath}")


# ============================================
# ATTENTION MECHANISM
# ============================================

class Attention(nn.Module):
    """Self-attention mechanism for sequence classification"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, lstm_output, mask=None):
        """
        Args:
            lstm_output: (batch, seq_len, hidden*2)
            mask: (batch, seq_len) - True for valid positions
            
        Returns:
            context: (batch, hidden*2)
            attention_weights: (batch, seq_len)
        """
        # Calculate attention scores
        attn_scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len)
        
        # Weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        
        return context, attn_weights


# ============================================
# BBBP CLASSIFIER MODEL
# ============================================

class BBBPClassifier(nn.Module):
    """
    BBBP Peptide Classifier with BiLSTM + Attention + iFeature fusion
    
    Architecture:
        BioVec embeddings -> BiLSTM -> Attention -> [Concat iFeature] -> FC -> Sigmoid
    """
    
    def __init__(self, biovec_dim=BIOVEC_DIM, ifeature_dim=2060,
                 hidden_dim=CLASSIFIER_HIDDEN_DIM, num_layers=CLASSIFIER_NUM_LAYERS,
                 dropout=CLASSIFIER_DROPOUT):
        super().__init__()
        
        self.biovec_dim = biovec_dim
        self.hidden_dim = hidden_dim
        
        # BiLSTM for sequence encoding
        self.lstm = nn.LSTM(
            input_size=biovec_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = Attention(hidden_dim)
        
        # Fusion layer (BiLSTM output + iFeature)
        fusion_dim = hidden_dim * 2 + ifeature_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, biovec_emb, ifeature, lengths=None):
        """
        Args:
            biovec_emb: (batch, seq_len, biovec_dim) - BioVec embeddings
            ifeature: (batch, ifeature_dim) - iFeature descriptors
            lengths: (batch,) - Original sequence lengths
            
        Returns:
            prob: (batch,) - BBBP probability
        """
        batch_size = biovec_emb.size(0)
        
        # BiLSTM encoding
        lstm_out, _ = self.lstm(biovec_emb)  # (batch, seq_len, hidden*2)
        
        # Create mask based on lengths
        if lengths is not None:
            max_len = lstm_out.size(1)
            mask = torch.arange(max_len, device=lstm_out.device).expand(batch_size, max_len) < lengths.unsqueeze(1)
        else:
            mask = None
        
        # Attention pooling
        context, attn_weights = self.attention(lstm_out, mask)  # (batch, hidden*2)
        
        # Concatenate with iFeature
        fused = torch.cat([context, ifeature], dim=1)
        
        # Classification
        prob = self.classifier(fused).squeeze(-1)
        
        return prob


# ============================================
# CLASSIFIER DATASET
# ============================================

class ClassifierDataset(Dataset):
    """Dataset for BBBP classifier with BioVec and iFeature"""
    
    def __init__(self, sequences, labels, biovec_model, ifeature_extractor, 
                 scaler=None, fit_scaler=False, max_len=MAX_SEQ_LEN):
        self.sequences = sequences
        self.labels = labels
        self.biovec_model = biovec_model
        self.extractor = ifeature_extractor
        self.max_len = max_len
        self.ngram = biovec_model.ngram
        
        # Pre-extract all features
        print("  Extracting iFeature descriptors...")
        self.ifeatures = extract_features_batch(sequences, ifeature_extractor)
        
        # Scale iFeatures
        if fit_scaler:
            self.scaler = StandardScaler()
            self.ifeatures = self.scaler.fit_transform(self.ifeatures)
        elif scaler is not None:
            self.scaler = scaler
            self.ifeatures = self.scaler.transform(self.ifeatures)
        else:
            self.scaler = None
            
        # Pre-compute BioVec embeddings
        print("  Computing BioVec embeddings...")
        self.biovec_embeddings = []
        for seq in tqdm(sequences, desc="BioVec"):
            emb = biovec_model.embed_sequence_padded(seq, max_len)
            self.biovec_embeddings.append(emb)
        self.biovec_embeddings = np.array(self.biovec_embeddings)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        biovec = torch.tensor(self.biovec_embeddings[idx], dtype=torch.float32)
        ifeature = torch.tensor(self.ifeatures[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        length = len(self.sequences[idx]) - self.ngram + 1
        
        return {
            'biovec': biovec,
            'ifeature': ifeature,
            'label': label,
            'length': torch.tensor(length, dtype=torch.long),
            'sequence': self.sequences[idx]
        }


# ============================================
# TRAINING FUNCTIONS
# ============================================

def train_classifier(train_df, val_df, biovec_model=None):
    """
    Train the BBBP classifier
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        biovec_model: Pre-trained BioVec model (optional)
        
    Returns:
        Trained model, scaler, and BioVec model
    """
    print("\n" + "=" * 60)
    print("STEP 4: Training BBBP Classifier")
    print("=" * 60)
    
    set_seed()
    
    # Extract sequences and labels
    train_seqs = train_df['seq'].tolist()
    train_labels = train_df['label'].values
    val_seqs = val_df['seq'].tolist()
    val_labels = val_df['label'].values
    
    # Train BioVec if not provided
    if biovec_model is None:
        biovec_model = BioVecModel()
        biovec_model.train(train_seqs + val_seqs)
    
    # Initialize iFeature extractor
    ifeature_extractor = iFeatureExtractor()
    ifeature_dim = ifeature_extractor.get_feature_dimension()
    print(f"\niFeature dimension: {ifeature_dim}")
    
    # Create datasets
    print("\nPreparing training data...")
    train_dataset = ClassifierDataset(
        train_seqs, train_labels, biovec_model, ifeature_extractor,
        fit_scaler=True
    )
    scaler = train_dataset.scaler
    
    print("\nPreparing validation data...")
    val_dataset = ClassifierDataset(
        val_seqs, val_labels, biovec_model, ifeature_extractor,
        scaler=scaler
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = BBBPClassifier(
        biovec_dim=BIOVEC_DIM,
        ifeature_dim=ifeature_dim,
        hidden_dim=CLASSIFIER_HIDDEN_DIM,
        num_layers=CLASSIFIER_NUM_LAYERS,
        dropout=CLASSIFIER_DROPOUT
    ).to(DEVICE)
    
    print(f"\nModel architecture:")
    print(model)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CLASSIFIER_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=CLASSIFIER_PATIENCE, mode='min')
    
    # Training history for plotting
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'train_auc': [], 'val_auc': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': []
    }
    
    # Training loop
    print(f"\nTraining for max {CLASSIFIER_EPOCHS} epochs...")
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(CLASSIFIER_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_preds, train_probs, train_targets = [], [], []
        
        for batch in train_loader:
            biovec = batch['biovec'].to(DEVICE)
            ifeature = batch['ifeature'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            lengths = batch['length'].to(DEVICE)
            
            optimizer.zero_grad()
            probs = model(biovec, ifeature, lengths)
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_probs.extend(probs.detach().cpu().numpy())
            train_preds.extend((probs > 0.5).long().cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_metrics = calculate_metrics(train_targets, train_preds, train_probs)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_probs, val_targets = [], [], []
        
        with torch.no_grad():
            for batch in val_loader:
                biovec = batch['biovec'].to(DEVICE)
                ifeature = batch['ifeature'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                lengths = batch['length'].to(DEVICE)
                
                probs = model(biovec, ifeature, lengths)
                loss = criterion(probs, labels)
                
                val_loss += loss.item()
                val_probs.extend(probs.cpu().numpy())
                val_preds.extend((probs > 0.5).long().cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_metrics = calculate_metrics(val_targets, val_preds, val_probs)
        
        # Record history for plotting
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_auc'].append(val_metrics['auc'])
        history['train_precision'].append(train_metrics['precision'])
        history['val_precision'].append(val_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['val_recall'].append(val_metrics['recall'])
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch {epoch + 1}/{CLASSIFIER_EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        
        # Save best model and store best predictions for plotting
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_val_targets = val_targets
            best_val_preds = val_preds
            best_val_probs = val_probs
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save model and components
    os.makedirs(CLASSIFIER_DIR, exist_ok=True)
    
    # Save BioVec
    biovec_path = os.path.join(CLASSIFIER_DIR, "biovec.model")
    biovec_model.save(biovec_path)
    
    # Save classifier
    classifier_path = os.path.join(CLASSIFIER_DIR, "classifier.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'biovec_dim': BIOVEC_DIM,
        'ifeature_dim': ifeature_dim,
        'hidden_dim': CLASSIFIER_HIDDEN_DIM,
        'num_layers': CLASSIFIER_NUM_LAYERS,
        'dropout': CLASSIFIER_DROPOUT
    }, classifier_path)
    print(f"\nClassifier saved to {classifier_path}")
    
    # Save scaler
    scaler_path = os.path.join(CLASSIFIER_DIR, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    # Final evaluation
    print("\n" + "=" * 40)
    print("Final Validation Metrics:")
    print("=" * 40)
    print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall:    {val_metrics['recall']:.4f}")
    print(f"  F1 Score:  {val_metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {val_metrics['auc']:.4f}")
    
    # Generate and save all evaluation plots for research
    plots_dir = os.path.join(CLASSIFIER_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("\n" + "=" * 40)
    print("Generating Evaluation Plots...")
    print("=" * 40)
    
    figures = plot_all_evaluation_figures(
        y_true=best_val_targets,
        y_pred=best_val_preds,
        y_prob=best_val_probs,
        history=history,
        save_dir=plots_dir
    )
    
    # Also save the training history for later analysis
    import json
    history_path = os.path.join(CLASSIFIER_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    return model, scaler, biovec_model


def load_classifier():
    """Load trained classifier and components"""
    
    # Load BioVec
    biovec_path = os.path.join(CLASSIFIER_DIR, "biovec.model")
    biovec_model = BioVecModel()
    biovec_model.load(biovec_path)
    
    # Load scaler
    scaler_path = os.path.join(CLASSIFIER_DIR, "scaler.pkl")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load classifier
    classifier_path = os.path.join(CLASSIFIER_DIR, "classifier.pth")
    checkpoint = torch.load(classifier_path, map_location=DEVICE)
    
    model = BBBPClassifier(
        biovec_dim=checkpoint['biovec_dim'],
        ifeature_dim=checkpoint['ifeature_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout']
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, scaler, biovec_model


def predict_bbbp(sequence, model, scaler, biovec_model):
    """
    Predict BBBP probability for a single sequence
    
    Args:
        sequence: Amino acid sequence
        model: Trained classifier
        scaler: Feature scaler
        biovec_model: BioVec model
        
    Returns:
        probability, prediction (0/1)
    """
    model.eval()
    
    # Extract features
    extractor = iFeatureExtractor()
    ifeature = extractor.extract_all_features(sequence)
    ifeature = scaler.transform(ifeature.reshape(1, -1))
    
    # Get BioVec embedding
    biovec = biovec_model.embed_sequence_padded(sequence, MAX_SEQ_LEN)
    
    # Convert to tensors
    biovec_tensor = torch.tensor(biovec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    ifeature_tensor = torch.tensor(ifeature, dtype=torch.float32).to(DEVICE)
    length_tensor = torch.tensor([len(sequence) - biovec_model.ngram + 1], dtype=torch.long).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        prob = model(biovec_tensor, ifeature_tensor, length_tensor)
    
    prob = prob.item()
    pred = 1 if prob >= 0.5 else 0
    
    return prob, pred


if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data, split_dataset
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    train_df, val_df = split_dataset(df)
    
    # Train classifier
    model, scaler, biovec_model = train_classifier(train_df, val_df)
    
    # Test prediction
    test_sequences = ["YGGFL", "MEHFRW", "ACDEFGHIK"]
    print("\n\nTest Predictions:")
    for seq in test_sequences:
        prob, pred = predict_bbbp(seq, model, scaler, biovec_model)
        label = "BBBP" if pred == 1 else "Non-BBBP"
        print(f"  {seq}: {prob:.4f} ({label})")
