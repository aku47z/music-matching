"""
N-Gram Similarity Module
Baseline similarity measure using symbolic string matching.
"""

from typing import List, Tuple, Set
from collections import Counter


def compute_ngrams(features: List[Tuple[float, float, int]], n: int = 5) -> List[Tuple]:
    """
    Generate n-grams from a feature sequence.
    
    Args:
        features: List of (pitch_interval, duration_ratio, is_downbeat) tuples
        n: Size of n-grams
        
    Returns:
        List of n-gram tuples
    """
    if len(features) < n:
        return []
    
    ngrams = []
    for i in range(len(features) - n + 1):
        # Create n-gram as tuple of feature tuples
        ngram = tuple(features[i:i + n])
        ngrams.append(ngram)
    
    return ngrams


def compute_ngrams_quantized(
    features: List[Tuple[float, float, int]], 
    n: int = 3,
    pitch_bins: int = 12,  # Quantize to semitone steps
    duration_bins: List[float] = None
) -> List[Tuple]:
    """
    Generate n-grams with quantized features for more robust matching.
    
    Args:
        features: List of (pitch_interval, duration_ratio, is_downbeat) tuples
        n: Size of n-grams
        pitch_bins: Number of pitch interval bins (default: 12 semitones)
        duration_bins: Bin edges for duration ratios
        
    Returns:
        List of quantized n-gram tuples
    """
    if duration_bins is None:
        duration_bins = [0.25, 0.5, 1.0, 2.0, 4.0]
    
    def quantize_pitch(p: float) -> int:
        # Clamp to reasonable range and round
        return max(-12, min(12, round(p)))
    
    def quantize_duration(d: float) -> int:
        # Find closest bin
        for i, edge in enumerate(duration_bins):
            if d <= edge:
                return i
        return len(duration_bins)
    
    quantized = []
    for pitch, duration, downbeat in features:
        q_pitch = quantize_pitch(pitch)
        q_duration = quantize_duration(duration)
        quantized.append((q_pitch, q_duration, downbeat))
    
    return compute_ngrams(quantized, n)


def ngram_similarity(ngrams_a: List[Tuple], ngrams_b: List[Tuple]) -> float:
    """
    Compute Jaccard similarity coefficient between two n-gram sets.
    
    Args:
        ngrams_a: N-grams from song A
        ngrams_b: N-grams from song B
        
    Returns:
        Similarity score in [0, 1]
    """
    set_a = set(ngrams_a)
    set_b = set(ngrams_b)
    
    if not set_a and not set_b:
        return 1.0  # Both empty
    if not set_a or not set_b:
        return 0.0  # One empty
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return intersection / union if union > 0 else 0.0


def ngram_similarity_weighted(ngrams_a: List[Tuple], ngrams_b: List[Tuple]) -> float:
    """
    Compute weighted n-gram similarity using frequency counts.
    Uses cosine similarity over n-gram frequency vectors.
    
    Args:
        ngrams_a: N-grams from song A
        ngrams_b: N-grams from song B
        
    Returns:
        Similarity score in [0, 1]
    """
    counter_a = Counter(ngrams_a)
    counter_b = Counter(ngrams_b)
    
    if not counter_a and not counter_b:
        return 1.0
    if not counter_a or not counter_b:
        return 0.0
    
    # Get all unique n-grams
    all_ngrams = set(counter_a.keys()) | set(counter_b.keys())
    
    # Compute dot product and magnitudes
    dot_product = 0
    mag_a = 0
    mag_b = 0
    
    for ngram in all_ngrams:
        a_count = counter_a.get(ngram, 0)
        b_count = counter_b.get(ngram, 0)
        dot_product += a_count * b_count
        mag_a += a_count ** 2
        mag_b += b_count ** 2
    
    if mag_a == 0 or mag_b == 0:
        return 0.0
    
    return dot_product / (mag_a ** 0.5 * mag_b ** 0.5)


def compute_baseline_similarity(
    features_a: List[Tuple[float, float, int]],
    features_b: List[Tuple[float, float, int]],
    n: int = 3,
    use_quantization: bool = True,
    use_weighted: bool = False
) -> float:
    """
    Compute baseline N-gram similarity between two feature sequences.
    
    Args:
        features_a: Features from song A
        features_b: Features from song B
        n: N-gram size
        use_quantization: Whether to quantize features
        use_weighted: Whether to use weighted (cosine) similarity
        
    Returns:
        Similarity score in [0, 1]
    """
    if use_quantization:
        ngrams_a = compute_ngrams_quantized(features_a, n)
        ngrams_b = compute_ngrams_quantized(features_b, n)
    else:
        ngrams_a = compute_ngrams(features_a, n)
        ngrams_b = compute_ngrams(features_b, n)
    
    if use_weighted:
        return ngram_similarity_weighted(ngrams_a, ngrams_b)
    else:
        return ngram_similarity(ngrams_a, ngrams_b)
