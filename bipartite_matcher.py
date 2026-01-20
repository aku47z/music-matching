"""
Bipartite Matcher Module
Core graph-theoretic matching using Maximum Weight Bipartite Matching.
Includes "Smoking Gun" detection for localized plagiarism analysis.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any, Optional
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
import math


# Default window configuration
DEFAULT_WINDOW_LEN = 10
DEFAULT_STEP = 5

# BMM-Det Parameters
SIGMA_SQ = 150.0  # Similarity curve parameter for Euclidean distance
TOP_K_MATCHES = 10  # Number of top matches for scoring

# Smoking Gun thresholds
HIGH_SIMILARITY_THRESHOLD = 0.7
HIGH_CONFIDENCE_THRESHOLD = 0.7
MEDIUM_SIMILARITY_THRESHOLD = 0.4


@dataclass
class MatchInfo:
    """Detailed information about a matched segment pair."""
    frag_a_idx: int
    frag_b_idx: int
    weight: float
    start_note_a: int
    end_note_a: int
    start_note_b: int
    end_note_b: int
    time_a: Optional[str] = None
    time_b: Optional[str] = None


@dataclass
class ClusterInfo:
    """Information about a cluster of consecutive matches (structural plagiarism)."""
    start_a: int
    end_a: int
    start_b: int
    end_b: int
    length: int
    avg_weight: float
    matches: List[Tuple[int, int]]


def align_features_by_motif(
    feat_a: List[Tuple[float, float, int]], 
    feat_b: List[Tuple[float, float, int]],
    motif_len: int = 3
) -> Tuple[List[Tuple[float, float, int]], List[Tuple[float, float, int]]]:
    """Align two feature sequences by finding the first matching motif of pitch intervals."""
    if len(feat_a) < motif_len or len(feat_b) < motif_len:
        return feat_a, feat_b

    # Search for first shared 3-note pitch sequence
    for i in range(len(feat_a) - motif_len + 1):
        motif_a = tuple(f[0] for f in feat_a[i:i+motif_len])
        for j in range(len(feat_b) - motif_len + 1):
            motif_b = tuple(f[0] for f in feat_b[j:j+motif_len])
            
            if motif_a == motif_b:
                # Found match! align from here
                return feat_a[i:], feat_b[j:]
                
    return feat_a, feat_b # No alignment found


def create_fragments(
    features: List[Tuple[float, float, int]], 
    window_len: int = DEFAULT_WINDOW_LEN, 
    step: int = DEFAULT_STEP
) -> Tuple[List[List[Tuple[float, float, int]]], List[Tuple[int, int]]]:
    """
    Create overlapping fragments using sliding window.
    Returns fragments and their note index ranges.
    
    Args:
        features: List of (pitch_interval, duration_ratio, is_downbeat) tuples
        window_len: Length of each window (default: 8 notes for musical phrase)
        step: Step size between windows (default: 4 notes)
        
    Returns:
        Tuple of (fragment lists, index ranges as (start, end) tuples)
    """
    if len(features) < window_len:
        # If sequence is too short, return entire sequence as one fragment
        if features:
            return [features], [(0, len(features))]
        return [], []
    
    fragments = []
    ranges = []
    
    for i in range(0, len(features) - window_len + 1, step):
        fragment = features[i:i + window_len]
        fragments.append(fragment)
        ranges.append((i, i + window_len))
    
    # Add final fragment if there's remaining content
    if len(features) > window_len and (len(features) - window_len) % step != 0:
        fragments.append(features[-window_len:])
        ranges.append((len(features) - window_len, len(features)))
    
    return fragments, ranges


def compute_edit_distance(
    frag_a: List[Tuple[float, float, int]], 
    frag_b: List[Tuple[float, float, int]],
    downbeat_weight: float = 0.3,
    pitch_weight: float = 1.0,
    duration_weight: float = 0.3
) -> float:
    """Compute Edit Distance (Levenshtein) with weighted Euclidean feature cost."""
    m, n = len(frag_a), len(frag_b)
    
    # Initialize DP table
    dp = np.zeros((m + 1, n + 1))
    
    # Base cases
    gap_cost = 2.0
    for i in range(m + 1):
        dp[i][0] = i * gap_cost
    for j in range(n + 1):
        dp[0][j] = j * gap_cost

    # Fill DP table with weighted Euclidean distances
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Feature vector: [Pitch_Interval, Duration_Ratio, Downbeat_Flag]
            p_a, r_a, d_a = float(frag_a[i - 1][0]), float(frag_a[i - 1][1]), float(frag_a[i - 1][2])
            p_b, r_b, d_b = float(frag_b[j - 1][0]), float(frag_b[j - 1][1]), float(frag_b[j - 1][2])
            
            # Weighted feature differences
            pitch_diff = 2.0 * abs(p_a - p_b)
            rhythm_diff = 0.3 * (r_a - r_b)
            downbeat_diff = 0.5 * (d_a - d_b)
            
            sub_cost = math.sqrt(pitch_diff**2 + rhythm_diff**2 + downbeat_diff**2)
            
            dp[i][j] = min(
                dp[i - 1][j] + gap_cost,
                dp[i][j - 1] + gap_cost,
                dp[i - 1][j - 1] + sub_cost
            )
    
    # Return raw Edit Distance (D) for BMM-Det weighting
    return dp[m][n]


def distance_to_similarity(distance: float) -> float:
    """Convert edit distance to similarity weight using exponential decay."""
    return math.exp(-(distance ** 2) / SIGMA_SQ)


def build_bipartite_graph(
    fragments_a: List[List[Tuple]], 
    fragments_b: List[List[Tuple]],
    downbeat_weight: float = 0.3
) -> Tuple[nx.Graph, np.ndarray]:
    """Build a complete weighted bipartite graph between fragments."""
    m, n = len(fragments_a), len(fragments_b)
    
    if m == 0 or n == 0:
        return nx.Graph(), np.array([])
    
    weight_matrix = np.zeros((m, n))
    
    for i, frag_a in enumerate(fragments_a):
        for j, frag_b in enumerate(fragments_b):
            distance = compute_edit_distance(frag_a, frag_b, downbeat_weight)
            weight = distance_to_similarity(distance)
            weight_matrix[i][j] = weight
    
    # Build NetworkX graph for visualization
    G = nx.Graph()
    
    for i in range(m):
        G.add_node(f"A{i}", bipartite=0, label=f"A{i}")
    
    for j in range(n):
        G.add_node(f"B{j}", bipartite=1, label=f"B{j}")
    
    for i in range(m):
        for j in range(n):
            G.add_edge(f"A{i}", f"B{j}", weight=weight_matrix[i][j])
    
    return G, weight_matrix


def hungarian_matching(weight_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], float]:
    """Apply Hungarian Algorithm to find optimal maximum weight matching."""
    if weight_matrix.size == 0:
        return [], 0.0
    
    cost_matrix = -weight_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    all_matches = list(zip(row_ind.tolist(), col_ind.tolist()))
    total_weight = weight_matrix[row_ind, col_ind].sum()
    
    return all_matches, total_weight


def get_top_k_matches(
    weight_matrix: np.ndarray, 
    k: int = TOP_K_MATCHES
) -> List[Tuple[int, int, float]]:
    """
    Get the Top-K highest similarity pairs from the weight matrix.
    These represent the most likely "copied" segments.
    """
    m, n = weight_matrix.shape
    all_pairs = []
    
    for i in range(m):
        for j in range(n):
            all_pairs.append((i, j, weight_matrix[i, j]))
    
    # Sort by weight descending
    all_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return all_pairs[:k]


def detect_sequence_clusters(
    matches: List[Tuple[int, int]],
    weight_matrix: np.ndarray,
    min_cluster_size: int = 3
) -> List[ClusterInfo]:
    """
    Detect clusters of consecutive matches indicating structural sequence plagiarism.
    If consecutive fragments in song A match consecutive fragments in song B,
    this suggests a copied section, not just coincidental similarity.
    """
    if not matches:
        return []
    
    # Sort matches by fragment index in song A
    sorted_matches = sorted(matches, key=lambda x: x[0])
    
    clusters = []
    current_cluster = [sorted_matches[0]]
    
    for i in range(1, len(sorted_matches)):
        prev_a, prev_b = sorted_matches[i - 1]
        curr_a, curr_b = sorted_matches[i]
        
        # Check if consecutive in both songs (allowing gap of 1)
        if curr_a - prev_a <= 2 and abs(curr_b - prev_b) <= 2:
            current_cluster.append(sorted_matches[i])
        else:
            # Process completed cluster
            if len(current_cluster) >= min_cluster_size:
                weights = [weight_matrix[a, b] for a, b in current_cluster]
                clusters.append(ClusterInfo(
                    start_a=current_cluster[0][0],
                    end_a=current_cluster[-1][0],
                    start_b=min(m[1] for m in current_cluster),
                    end_b=max(m[1] for m in current_cluster),
                    length=len(current_cluster),
                    avg_weight=np.mean(weights),
                    matches=current_cluster.copy()
                ))
            current_cluster = [sorted_matches[i]]
    
    # Process final cluster
    if len(current_cluster) >= min_cluster_size:
        weights = [weight_matrix[a, b] for a, b in current_cluster]
        clusters.append(ClusterInfo(
            start_a=current_cluster[0][0],
            end_a=current_cluster[-1][0],
            start_b=min(m[1] for m in current_cluster),
            end_b=max(m[1] for m in current_cluster),
            length=len(current_cluster),
            avg_weight=np.mean(weights),
            matches=current_cluster.copy()
        ))
    
    return clusters


def notes_to_time(note_idx: int, tempo_bpm: float = 120, notes_per_beat: float = 2) -> str:
    """
    Convert note index to approximate time string.
    Assumes the note index corresponds to position in feature sequence.
    """
    # Approximate: each note is 1/notes_per_beat beats
    beats = note_idx / notes_per_beat
    seconds = beats * (60 / tempo_bpm)
    
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    
    return f"{minutes}:{secs:02d}"


def compute_plagiarism_score(
    tuples_a: List[Tuple], 
    tuples_b: List[Tuple],
    window_len: int = DEFAULT_WINDOW_LEN,
    step: int = DEFAULT_STEP,
    downbeat_weight: float = 0.3,
    tempo_bpm: float = 120.0
) -> Dict[str, Any]:
    """
    Compute plagiarism score using BMM-Det framework.
    Returns dictionary with detailed results:
    - Combined score (structural integrity)
    - Top-K matches (hook similarity)
    - Structural sequence clusters
    - Time-localized match information
    """
    # Create fragments with index tracking
    # Align by first matching 3-note motif before fragmenting
    features_a, features_b = align_features_by_motif(tuples_a, tuples_b)

    fragments_a, ranges_a = create_fragments(features_a, window_len, step)
    fragments_b, ranges_b = create_fragments(features_b, window_len, step)
    
    if not fragments_a or not fragments_b:
        return {
            "score": 0.0,
            "hook_score": 0.0,
            "top_k_matches": [],
            "clusters": [],
            "localized_matches": [],
            "fragments_a": fragments_a,
            "fragments_b": fragments_b,
            "graph": nx.Graph(),
            "weight_matrix": np.array([])
        }
    
    # Build bipartite graph
    graph, weight_matrix = build_bipartite_graph(
        fragments_a, fragments_b, downbeat_weight
    )
    
    # Find optimal matching
    all_matches, total_weight = hungarian_matching(weight_matrix)
    
    # Top-K scoring (Hook Similarity Score)
    top_k = get_top_k_matches(weight_matrix, TOP_K_MATCHES)
    hook_score = np.mean([w for _, _, w in top_k]) if top_k else 0.0
    
    # Detect Significant Phrases (Clusters) - Threshold > 0.7, Min Size = 3
    threshold_matches = [(i, j) for i, j in all_matches 
                        if weight_matrix[i, j] >= HIGH_CONFIDENCE_THRESHOLD]
    clusters = detect_sequence_clusters(threshold_matches, weight_matrix, min_cluster_size=3)
    
    # 4. Localization Output - create detailed match info
    localized_matches = []
    for i, j, w in top_k:
        if w >= 0.05: # Capture all relevant matches for reporting
            start_a, end_a = ranges_a[i] if i < len(ranges_a) else (0, 0)
            start_b, end_b = ranges_b[j] if j < len(ranges_b) else (0, 0)
            
            match_info = MatchInfo(
                frag_a_idx=i,
                frag_b_idx=j,
                weight=w,
                start_note_a=start_a,
                end_note_a=end_a,
                start_note_b=start_b,
                end_note_b=end_b,
                time_a=f"{notes_to_time(start_a, tempo_bpm)}-{notes_to_time(end_a, tempo_bpm)}",
                time_b=f"{notes_to_time(start_b, tempo_bpm)}-{notes_to_time(end_b, tempo_bpm)}"
            )
            localized_matches.append(match_info)
    
    # Combined score = mean of Top-K matches (hook-focused scoring)
    if not top_k:
        combined_score = 0.0
    else:
        combined_score = np.mean([w for _, _, w in top_k])
    
    combined_score = max(0.0, min(1.0, combined_score))
    
    # High-Value Hook Match (Anchor Score)
    top_weights = [w for _, _, w in top_k]
    max_motif_match = max(top_weights) if top_weights else 0.0
    
    return {
        "score": combined_score,
        "combined_score": combined_score,
        "hook_score": hook_score,
        "max_motif_match": max_motif_match,
        "matches": all_matches,
        "top_k_matches": top_k,
        "clusters": clusters,
        "localized_matches": localized_matches,
        "total_weight": total_weight,
        "fragments_a": fragments_a,
        "fragments_b": fragments_b,
        "ranges_a": ranges_a,
        "ranges_b": ranges_b,
        "graph": graph,
        "weight_matrix": weight_matrix
    }
