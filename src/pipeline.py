"""
Smart City Decision-Support Framework
Author: Katherine Flores
Description: Reproducible computational pipeline for urban modeling
"""

# ===============================
# IMPORTS
# ===============================
import numpy as np
import pandas as pd
import networkx as nx

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

    return {
        "results": results,
        "validation": validation,
        "pareto": pareto
    }


# ===============================
# EXAMPLE EXECUTION
# ===============================

if __name__ == "__main__":

    sources = [
        {"id": "mobility", "path": "mobility.csv", "format": "csv"},
        {"id": "energy", "path": "energy.csv", "format": "csv"}
    ]

    edges = [(1,2), (2,3), (3,1), (3,4)]

    output = run_pipeline(sources, edges)

    print("Pipeline executed successfully")
    print(output)
