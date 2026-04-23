"""
Smart City Decision-Support Framework
Author: Katherine Flores
Description: A Reproducible Computational Framework for Decision Support in Smart Cities
"""

# ===============================
# IMPORTS
# ===============================
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
# (Opcional - si usas deep learning real)
# import torch
# import torch.nn as nn


# ===============================
# PHASE 1: DATA INGESTION
# ===============================

def load_data(path, fmt="csv"):
    if fmt == "csv":
        return pd.read_csv(path)
    elif fmt == "json":
        return pd.read_json(path)
    else:
        raise ValueError("Unsupported format")

def ingest_urban_sources(source_registry):
    raw_layers = {}
    for source in source_registry:
        raw_layers[source["id"]] = load_data(source["path"], source["format"])
    return raw_layers


def normalize_pipeline(raw_layers):
    normalized = {}

    for layer_id, df in raw_layers.items():
        df = df.copy()

        # Example normalization steps
        df = df.fillna(df.mean(numeric_only=True))
        df = (df - df.mean()) / df.std()

        normalized[layer_id] = df

    return normalized


# ===============================
# PHASE 2: REPRESENTATION
# ===============================

def build_city_graph(edge_list):
    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    return G


def generate_embeddings(normalized_data):
    """
    Placeholder for representation learning.
    Could be replaced with GNN / embeddings.
    """
    embeddings = {}

    for key, df in normalized_data.items():
        embeddings[key] = df.values

    return embeddings


# ===============================
# PHASE 3: HYBRID MODELING
# ===============================

def deep_forecast(embeddings):
    """Dummy forecasting model"""
    return {k: np.mean(v, axis=0) for k, v in embeddings.items()}


def rl_simulation(embeddings):
    """Dummy RL policy simulation"""
    return {k: np.random.rand() for k in embeddings}


def fuzzy_evaluation(embeddings):
    """Dummy fuzzy scores"""
    return {k: np.random.uniform(0, 1) for k in embeddings}


def bayesian_inference(embeddings):
    """Dummy probabilistic output"""
    return {k: np.random.normal(0, 1) for k in embeddings}


def run_hybrid_ensemble(embeddings):
    forecasts = deep_forecast(embeddings)
    rl_policy = rl_simulation(embeddings)
    fuzzy_scores = fuzzy_evaluation(embeddings)
    posterior = bayesian_inference(embeddings)

    return {
        "forecasts": forecasts,
        "rl_policy": rl_policy,
        "fuzzy_scores": fuzzy_scores,
        "posterior": posterior
    }


def generate_scenarios(embeddings, n_scenarios=10):
    scenarios = []
    for _ in range(n_scenarios):
        noise = np.random.normal(0, 0.1)
        perturbed = {k: v + noise for k, v in embeddings.items()}
        scenarios.append(perturbed)
    return scenarios


# ===============================
# PHASE 4: TOPOLOGY VALIDATION
# ===============================

def compute_robustness(G):
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return 0

    # Remove a node and test connectivity
    test_node = nodes[0]
    G_copy = G.copy()
    G_copy.remove_node(test_node)

    return 1 if nx.is_weakly_connected(G_copy) else 0


def topology_validation(G):
    result = {}

    # Connectivity
    result["connected"] = nx.is_weakly_connected(G)

    # Cycles
    cycles = list(nx.simple_cycles(G))
    result["cycle_index"] = len(cycles) / max(len(G.nodes()), 1)

    # Robustness
    result["robustness"] = compute_robustness(G)

    result["valid"] = (
        result["connected"] and
        result["cycle_index"] > 0 and
        result["robustness"] > 0
    )

    return result


# ===============================
# PHASE 5: MULTIOBJECTIVE
# ===============================

def evaluate_solution(solution):
    return [
        np.random.rand(),  # efficiency
        np.random.rand(),  # equity
        np.random.rand(),  # sustainability
        np.random.rand()   # resilience
    ]


def compute_pareto_front(solutions):
    scores = [evaluate_solution(s) for s in solutions]
    return scores

# ===============================
# NETWORK TOPOLOGY        
# ===============================

def generate_network_figure(edge_list, output_path="results/figures"):
    """
    Generates the urban infrastructure network visualization
    and robustness analysis curves (Figure 4 in the chapter).
    Uses build_city_graph() and topology_validation() from Phase 2/4.
    """
    import matplotlib.patches as mpatches

    os.makedirs(output_path, exist_ok=True)
    np.random.seed(42)

    zones = {
        0:"Historic\nCenter", 1:"North\nDistrict",  2:"Industrial\nZone",
        3:"South\nResidential", 4:"University\nCampus", 5:"Airport\nArea",
        6:"East\nCommercial", 7:"West\nPark", 8:"Hospital\nHub", 9:"Tech\nPark"
    }

    G = build_city_graph(edge_list)
    betweenness = nx.betweenness_centrality(G)
    degree = dict(G.degree())

    critical   = [n for n,v in betweenness.items() if v > 0.12]
    vulnerable = [n for n in G.nodes() if degree[n] <= 4 and n not in critical]

    pos = {0:(0.5,0.5),1:(0.5,0.85),2:(0.85,0.85),3:(0.5,0.15),4:(0.2,0.75),
           5:(0.85,0.55),6:(0.8,0.25),7:(0.15,0.25),8:(0.2,0.45),9:(0.65,0.65)}

    fig, axes = plt.subplots(1, 2, figsize=(20,9))

    # Left: network
    ax = axes[0]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#9EAAB8',
        arrows=True, arrowsize=18, width=1.4, connectionstyle='arc3,rad=0.07')
    colors = ['#E63946' if n in critical else '#F4A261' if n in vulnerable
              else '#457B9D' for n in G.nodes()]
    sizes = [500 + betweenness[n]*4000 for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
        node_size=sizes, edgecolors='white', linewidths=2)
    nx.draw_networkx_labels(G, pos, labels=zones, ax=ax,
        font_size=7, font_weight='bold', font_color='white')
    patches = [mpatches.Patch(color='#E63946', label='Critical Hub'),
               mpatches.Patch(color='#F4A261', label='Vulnerable Zone'),
               mpatches.Patch(color='#457B9D', label='Standard Node')]
    ax.legend(handles=patches, loc='lower right')
    ax.set_title("Urban Infrastructure Network")
    ax.axis('off')

    # Right: robustness curves
    ax2 = axes[1]
    sorted_nodes = sorted(G.nodes(), key=lambda n: degree[n], reverse=True)

    def lcc(G, removed):
        G2 = G.copy()
        for n in removed:
            if n in G2: G2.remove_node(n)
        if not G2: return 0.0
        return max(len(c) for c in nx.weakly_connected_components(G2)) / len(G.nodes())

    targeted = [lcc(G, sorted_nodes[:i]) for i in range(len(sorted_nodes)+1)]
    random_avg = []
    for i in range(len(G.nodes())+1):
        trials = []
        for _ in range(30):
            s = list(G.nodes()); np.random.shuffle(s)
            trials.append(lcc(G, s[:i]))
        random_avg.append(np.mean(trials))

    x = np.linspace(0, 1, len(targeted))
    ax2.plot(x, targeted,    color='#E63946', linewidth=2.5, label='Targeted attack')
    ax2.plot(x, random_avg, color='#457B9D', linewidth=2.5,
             linestyle='--', label='Random failure')
    ax2.fill_between(x, random_avg, targeted, alpha=0.12,
                     color='#E63946', label='Vulnerability gap')
    ax2.axhline(0.5, color='gray', linestyle=':', linewidth=1.2)
    ax2.set_xlabel("Fraction of Nodes Removed")
    ax2.set_ylabel("Largest Connected Component (fraction)")
    ax2.legend(); ax2.set_xlim(0,1); ax2.set_ylim(0,1.05); ax2.grid(alpha=0.3)
    ax2.set_title("Network Robustness Under Node Removal")

    plt.tight_layout()
    plt.savefig(f"{output_path}/network_topology.tiff", dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Figure 4 saved to {output_path}/network_topology.tiff")
    
# ===============================
# MAIN PIPELINE
# ===============================

def run_pipeline(source_registry, edge_list):

    # Phase 1
    raw = ingest_urban_sources(source_registry)
    normalized = normalize_pipeline(raw)

    # Phase 2
    embeddings = generate_embeddings(normalized)
    G = build_city_graph(edge_list)

    # Phase 3
    results = run_hybrid_ensemble(embeddings)
    scenarios = generate_scenarios(embeddings)

    # Phase 4
    validation = topology_validation(G)

    # Phase 5
    pareto = compute_pareto_front(scenarios)

    # Generate figures
    output_path = "results/figures"
    generate_figures(results, output_path)
    generate_network_figure(edge_list, output_path)

    return {
        "results": results,
        "validation": validation,
        "pareto": pareto
    }
# ===============================
# Figures    
# ===============================

def generate_figures(results, output_path="results/figures"):
    os.makedirs(output_path, exist_ok=True)

    # Example: traffic (from forecasts)
    traffic = list(results["forecasts"].values())[0]

    plt.figure()
    plt.plot(traffic)
    plt.title("Traffic Flow Simulation")
    plt.xlabel("Time Step")
    plt.ylabel("Traffic")
    plt.savefig(f"{output_path}/traffic.tiff", dpi=600)
    plt.close()

    # Energy (dummy example)
    energy = list(results["forecasts"].values())[0]

    plt.figure()
    plt.plot(energy)
    plt.title("Energy Demand Forecast")
    plt.xlabel("Time Step")
    plt.ylabel("Energy")
    plt.savefig(f"{output_path}/energy.tiff", dpi=600)
    plt.close()

    # Air quality (dummy example)
    air = list(results["forecasts"].values())[0]

    plt.figure()
    plt.plot(air)
    plt.title("Air Quality Prediction")
    plt.xlabel("Time Step")
    plt.ylabel("AQI")
    plt.savefig(f"{output_path}/air_quality.tiff", dpi=600)
    plt.close()

