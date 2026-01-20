"""
Visualizer Module
Bipartite graph visualization for plagiarism detection results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from typing import List, Tuple, Optional

HIGH_SIMILARITY_THRESHOLD = 0.8


def visualize_smoking_gun(
    graph: nx.Graph,
    weight_matrix: np.ndarray,
    top_k_matches: List[Tuple[int, int, float]],
    title: str = "Smoking Gun Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
    max_nodes_display: int = 15
) -> plt.Figure:
    """
    Visualize the bipartite graph with opacity-coded edges.
    Shows only top nodes with highest matching weights.
    """
    if graph.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data to visualize", ha='center', va='center', fontsize=14)
        ax.set_title(title)
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get all nodes
    all_top_nodes = sorted([n for n, d in graph.nodes(data=True) if d.get('bipartite') == 0],
                          key=lambda x: int(x[1:]))
    all_bottom_nodes = sorted([n for n, d in graph.nodes(data=True) if d.get('bipartite') == 1],
                             key=lambda x: int(x[1:]))
    
    # Find top nodes based on maximum matching weight
    node_a_max_weights = {}
    for node in all_top_nodes:
        node_idx = int(node[1:])
        if node_idx < weight_matrix.shape[0]:
            max_weight = np.max(weight_matrix[node_idx, :])
            node_a_max_weights[node] = max_weight
    
    node_b_max_weights = {}
    for node in all_bottom_nodes:
        node_idx = int(node[1:])
        if node_idx < weight_matrix.shape[1]:
            max_weight = np.max(weight_matrix[:, node_idx])
            node_b_max_weights[node] = max_weight
    
    # Select top N nodes with highest weights
    top_nodes_sorted = sorted(node_a_max_weights.items(), key=lambda x: x[1], reverse=True)
    top_nodes_selected = set([node for node, weight in top_nodes_sorted[:max_nodes_display]])
    top_nodes = [n for n in all_top_nodes if n in top_nodes_selected]
    
    bottom_nodes_sorted = sorted(node_b_max_weights.items(), key=lambda x: x[1], reverse=True)
    bottom_nodes_selected = set([node for node, weight in bottom_nodes_sorted[:max_nodes_display]])
    bottom_nodes = [n for n in all_bottom_nodes if n in bottom_nodes_selected]
    
    # Create positions
    pos = {}
    for i, node in enumerate(top_nodes):
        pos[node] = (i, 1)
    for i, node in enumerate(bottom_nodes):
        pos[node] = (i, 0)
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, nodelist=top_nodes,
                          node_color='#4A90D9', node_size=500, ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=bottom_nodes,
                          node_color='#D94A4A', node_size=500, ax=ax)
    
    # Draw node labels
    labels = {n: n for n in pos}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, font_color='white', ax=ax)
    
    # Add time labels for nodes
    for node in top_nodes:
        node_idx = int(node[1:])
        time_minutes = (node_idx * 2) // 60
        time_seconds = (node_idx * 2) % 60
        time_label = f"{time_minutes:02d}:{time_seconds:02d}"
        x, y = pos[node]
        ax.text(x, y + 0.15, time_label, fontsize=7, ha='center', va='bottom', 
                color='#2d5a7d', fontweight='bold')
    
    for node in bottom_nodes:
        node_idx = int(node[1:])
        time_minutes = (node_idx * 2) // 60
        time_seconds = (node_idx * 2) % 60
        time_label = f"{time_minutes:02d}:{time_seconds:02d}"
        x, y = pos[node]
        ax.text(x, y - 0.15, time_label, fontsize=7, ha='center', va='top',
                color='#7d2d2d', fontweight='bold')
    
    # Draw edges with opacity based on weight
    all_edges = []
    edge_alphas = []
    edge_widths = []
    edge_labels_high = {}
    
    for i in range(weight_matrix.shape[0]):
        for j in range(weight_matrix.shape[1]):
            edge = (f"A{i}", f"B{j}")
            if edge[0] in pos and edge[1] in pos:
                w = weight_matrix[i, j]
                all_edges.append(edge)
                
                # Calculate opacity
                alpha = 0.05 if w < 0.2 else w
                edge_alphas.append(alpha)
                
                # Calculate width
                width = 2.0 + (w * 3.0)
                edge_widths.append(width)
                
                # Labels for high-similarity edges
                if w >= HIGH_SIMILARITY_THRESHOLD:
                    edge_labels_high[edge] = f"{w:.2f}"
    
    # Draw edges
    for edge, alpha, width in zip(all_edges, edge_alphas, edge_widths):
        nx.draw_networkx_edges(graph, pos, edgelist=[edge],
                              edge_color='#000000', alpha=alpha, width=width, ax=ax)
    
    # Add weight labels for high-similarity edges
    if edge_labels_high:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels_high,
                                    font_size=9, font_color='darkred', font_weight='bold', ax=ax)
    
    # Customize plot
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(-0.4, 1.4)
    
    # Legend
    black_thick = mpatches.Patch(color='#000000', label='Edge opacity = similarity weight', alpha=0.8)
    high_line = mpatches.Patch(color='#000000', label=f'High weight (>{HIGH_SIMILARITY_THRESHOLD}): Thick & opaque', alpha=1.0)
    low_line = mpatches.Patch(color='#000000', label='Low weight: Thin & faint', alpha=0.2)
    blue_patch = mpatches.Patch(color='#4A90D9', label='Song A Fragments')
    red_patch = mpatches.Patch(color='#D94A4A', label='Song B Fragments')
    
    ax.legend(handles=[high_line, low_line, black_thick, blue_patch, red_patch], 
              loc='upper right', fontsize=9)
    
    # Axis labels
    ax.text(-0.5, 1, 'Song A', fontsize=12, fontweight='bold', va='center')
    ax.text(-0.5, 0, 'Song B', fontsize=12, fontweight='bold', va='center')
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
