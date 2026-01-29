"""
FOR CLASSIFIER ONLY
Extracts various physicochemical descriptors from peptide sequences
"""

import numpy as np
from collections import Counter


# ============================================
# AMINO ACID PROPERTIES
# ============================================

# Amino acid hydrophobicity (Kyte-Doolittle scale)
HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

# Amino acid charge at pH 7
CHARGE = {
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
    'G': 0, 'H': 0.1, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
    'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
}

# Amino acid molecular weight
MOLECULAR_WEIGHT = {
    'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2,
    'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
    'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2,
    'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2
}

# Amino acid polarity
POLARITY = {
    'A': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 0,
    'G': 0, 'H': 1, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 1, 'P': 0, 'Q': 1, 'R': 1,
    'S': 1, 'T': 1, 'V': 0, 'W': 0, 'Y': 1
}

# Standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class iFeatureExtractor:
    """
    Extract iFeature descriptors from peptide sequences
    
    Descriptors implemented:
    - AAC: Amino Acid Composition
    - DPC: Dipeptide Composition  
    - CKSAAP: Composition of k-spaced Amino Acid Pairs
    - PAAC: Pseudo Amino Acid Composition
    - Charge features
    - Hydrophobicity features
    """
    
    def __init__(self):
        self.amino_acids = AMINO_ACIDS
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
        
    def extract_all_features(self, sequence):
        """
        Extract all iFeature descriptors for a sequence
        
        Args:
            sequence: Amino acid sequence string
            
        Returns:
            Concatenated feature vector
        """
        features = []
        
        # AAC - 20 features
        features.extend(self.aac(sequence))
        
        # DPC - 400 features
        features.extend(self.dpc(sequence))
        
        # CKSAAP - 400 features (k=0 to k=3)
        features.extend(self.cksaap(sequence, k_max=3))
        
        # PAAC - 30 features (lambda=10)
        features.extend(self.paac(sequence, lambda_val=10))
        
        # Charge features - 5 features
        features.extend(self.charge_features(sequence))
        
        # Hydrophobicity features - 5 features
        features.extend(self.hydrophobicity_features(sequence))
        
        return np.array(features, dtype=np.float32)
    
    def aac(self, sequence):
        """
        Amino Acid Composition (AAC)
        
        Returns:
            List of 20 composition values
        """
        seq = sequence.upper()
        length = len(seq)
        
        if length == 0:
            return [0.0] * 20
            
        counter = Counter(seq)
        composition = [counter.get(aa, 0) / length for aa in self.amino_acids]
        
        return composition
    
    def dpc(self, sequence):
        """
        Dipeptide Composition (DPC)
        
        Returns:
            List of 400 dipeptide composition values
        """
        seq = sequence.upper()
        length = len(seq)
        
        if length < 2:
            return [0.0] * 400
        
        # Generate all dipeptides
        dipeptides = [seq[i:i+2] for i in range(length - 1)]
        counter = Counter(dipeptides)
        total = len(dipeptides)
        
        # Create composition vector
        composition = []
        for aa1 in self.amino_acids:
            for aa2 in self.amino_acids:
                dipep = aa1 + aa2
                composition.append(counter.get(dipep, 0) / total)
                
        return composition
    
    def cksaap(self, sequence, k_max=3):
        """
        Composition of k-Spaced Amino Acid Pairs (CKSAAP)
        
        Args:
            sequence: Amino acid sequence
            k_max: Maximum k value (0 to k_max)
            
        Returns:
            Concatenated feature vector for k=0 to k_max
        """
        seq = sequence.upper()
        features = []
        
        for k in range(k_max + 1):
            pairs = []
            for i in range(len(seq) - k - 1):
                pairs.append(seq[i] + seq[i + k + 1])
            
            if len(pairs) == 0:
                features.extend([0.0] * 400)
                continue
                
            counter = Counter(pairs)
            total = len(pairs)
            
            for aa1 in self.amino_acids:
                for aa2 in self.amino_acids:
                    pair = aa1 + aa2
                    features.append(counter.get(pair, 0) / total)
                    
        return features
    
    def paac(self, sequence, lambda_val=10):
        """
        Pseudo Amino Acid Composition (PAAC)
        
        Args:
            sequence: Amino acid sequence
            lambda_val: Number of correlation factors
            
        Returns:
            PAAC feature vector (20 + lambda features)
        """
        seq = sequence.upper()
        length = len(seq)
        
        if length < 2:
            return [0.0] * (20 + lambda_val)
        
        # Get AAC
        aac = self.aac(sequence)
        
        # Calculate sequence-order correlation factors
        tau = []
        for lag in range(1, min(lambda_val + 1, length)):
            correlation = 0
            count = 0
            for i in range(length - lag):
                aa1, aa2 = seq[i], seq[i + lag]
                if aa1 in HYDROPHOBICITY and aa2 in HYDROPHOBICITY:
                    h1, h2 = HYDROPHOBICITY[aa1], HYDROPHOBICITY[aa2]
                    correlation += (h1 - h2) ** 2
                    count += 1
            if count > 0:
                tau.append(correlation / count)
            else:
                tau.append(0.0)
        
        # Pad if sequence is too short
        while len(tau) < lambda_val:
            tau.append(0.0)
        
        # Normalize
        weight = 0.05  # Weight factor for correlation
        sum_tau = sum(tau) * weight
        denom = 1 + sum_tau
        
        paac_features = [a / denom for a in aac]
        paac_features.extend([weight * t / denom for t in tau])
        
        return paac_features
    
    def charge_features(self, sequence):
        """
        Charge-based features
        
        Returns:
            5 charge features: total, positive, negative, net, ratio
        """
        seq = sequence.upper()
        length = len(seq)
        
        if length == 0:
            return [0.0] * 5
        
        charges = [CHARGE.get(aa, 0) for aa in seq]
        
        total_charge = sum(abs(c) for c in charges)
        positive = sum(1 for c in charges if c > 0)
        negative = sum(1 for c in charges if c < 0)
        net_charge = sum(charges)
        
        return [
            total_charge / length,
            positive / length,
            negative / length,
            net_charge / length,
            (positive - negative) / length if length > 0 else 0
        ]
    
    def hydrophobicity_features(self, sequence):
        """
        Hydrophobicity-based features
        
        Returns:
            5 hydrophobicity features: mean, std, max, min, range
        """
        seq = sequence.upper()
        length = len(seq)
        
        if length == 0:
            return [0.0] * 5
        
        hydro = [HYDROPHOBICITY.get(aa, 0) for aa in seq]
        
        mean_h = np.mean(hydro)
        std_h = np.std(hydro)
        max_h = np.max(hydro)
        min_h = np.min(hydro)
        range_h = max_h - min_h
        
        return [mean_h, std_h, max_h, min_h, range_h]
    
    def get_feature_dimension(self):
        """Get total number of features"""
        # AAC: 20
        # DPC: 400
        # CKSAAP (k=0-3): 400 * 4 = 1600
        # PAAC: 30
        # Charge: 5
        # Hydrophobicity: 5
        return 20 + 400 + 1600 + 30 + 5 + 5  # = 2060


def extract_features_batch(sequences, extractor=None):
    """
    Extract features for a batch of sequences
    
    Args:
        sequences: List of amino acid sequences
        extractor: iFeatureExtractor instance (optional)
        
    Returns:
        numpy array of features
    """
    if extractor is None:
        extractor = iFeatureExtractor()
    
    features = []
    for seq in sequences:
        features.append(extractor.extract_all_features(seq))
    
    return np.array(features)


if __name__ == "__main__":
    # Test feature extraction
    extractor = iFeatureExtractor()
    
    test_sequences = [
        "ACDEFGHIK",
        "YGGFL",
        "MEHFRW",
        "RRLSYSRRRF"
    ]
    
    print("Testing iFeature Descriptor Extraction")
    print("=" * 50)
    
    for seq in test_sequences:
        features = extractor.extract_all_features(seq)
        print(f"\nSequence: {seq}")
        print(f"  Length: {len(seq)}")
        print(f"  Feature dimension: {len(features)}")
        print(f"  Feature range: [{features.min():.4f}, {features.max():.4f}]")
        print(f"  Feature mean: {features.mean():.4f}")
    
    print(f"\n\nTotal feature dimension: {extractor.get_feature_dimension()}")
