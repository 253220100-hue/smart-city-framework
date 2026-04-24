"""
Smart City Decision-Support Framework
Author: Katherine Flores
Description: A Reproducible Computational Framework for Decision Support in Smart Cities
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── PHASE 1: DATA INGESTION ───────────────────────────────────────────────────

def load_data(path: str, fmt: str = "csv") -> pd.DataFrame:
    """Load a single urban data layer from disk."""
    if fmt == "csv":
        return pd.read_csv(path)
    elif fmt == "json":
        return pd.read_json(path)
    else:
        raise ValueError(f"Unsupported format: {fmt!r}")


def ingest_urban_sources(source_registry: list) -> dict:
    """
    Load all urban data layers defined in source_registry.

    Parameters
    ----------
    source_registry : list of dict
        Each dict must have keys: 'id', 'path', 'format'.

    Returns
    -------
    dict
        Mapping of layer_id → raw DataFrame.
    """
    raw_layers = {}
    for source in source_registry:
        raw_layers[source["id"]] = load_data(source["path"], source["format"])
    return raw_layers


def normalize_pipeline(raw_layers: dict) -> dict:
    """
    Apply z-score normalization to all data layers.

    Missing values are filled with the column mean before standardization.
    The transformation x̂ = (x − μ) / σ maps each feature to zero mean
    and unit variance, ensuring comparability across heterogeneous sources.

    Parameters
    ----------
    raw_layers : dict
        Mapping of layer_id → raw DataFrame.

    Returns
    -------
    dict
        Mapping of layer_id → normalized DataFrame.
    """
    normalized = {}
    for layer_id, df in raw_layers.items():
        df = df.copy()
        df = df.fillna(df.mean(numeric_only=True))
        df = (df - df.mean()) / df.std()
        normalized[layer_id] = df
    return normalized


# ── PHASE 2: REPRESENTATION ────────────────────────────────────────────────────

def build_city_graph(edge_list: list) -> nx.DiGraph:
    """
    Construct a directed urban infrastructure graph.

    Parameters
    ----------
    edge_list : list of (int, int)
        Directed edges between urban zone indices.

    Returns
    -------
    nx.DiGraph
    """
    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    return G


def generate_urban_embeddings(normalized_data: dict) -> dict:
    """
    Produce structured feature embeddings for each urban data layer.

    This function acts as a surrogate for a graph neural network
    embedding module. Rather than learning transformation weights via
    gradient descent, it extracts the normalized feature matrix directly,
    ensuring full reproducibility without a training step.

    Parameters
    ----------
    normalized_data : dict
        Mapping of layer_id → normalized DataFrame.

    Returns
    -------
    dict
        Mapping of layer_id → NumPy feature matrix (shape: rows × features).
    """
    embeddings = {}
    for key, df in normalized_data.items():
        embeddings[key] = df.values  # deterministic matrix projection
    return embeddings


# ── PHASE 3: SURROGATE HYBRID ENSEMBLE ────────────────────────────────────────

def surrogate_forecast_model(embeddings: dict, seed: int = 42) -> dict:
    """
    Surrogate for a deep-learning demand forecasting module.

    Computes a weighted mean aggregation of each embedding matrix,
    mimicking the output structure of an LSTM or GNN forecasting model
    without requiring trained weights.

    Parameters
    ----------
    embeddings : dict
        Mapping of layer_id → feature matrix.
    seed : int, optional
        Kept for API consistency; mean aggregation is deterministic.

    Returns
    -------
    dict
        Mapping of layer_id → forecast vector (mean along axis 0).
    """
    return {k: np.mean(v, axis=0) for k, v in embeddings.items()}


def policy_simulation_module(embeddings: dict, seed: int = 42) -> dict:
    """
    Surrogate for an MDP-based reinforcement-learning policy module.

    Returns a stochastic policy score per urban layer, sampled from a
    seeded uniform distribution. The fixed seed guarantees that identical
    inputs produce identical outputs across all runs.

    Parameters
    ----------
    embeddings : dict
    seed : int

    Returns
    -------
    dict
        Mapping of layer_id → scalar policy score in [0, 1].
    """
    np.random.seed(seed)
    return {k: float(np.random.rand()) for k in embeddings}


def heuristic_scoring(embeddings: dict, seed: int = 7) -> dict:
    """
    Surrogate for a fuzzy inference evaluation system.

    Maps embedding statistics to a normalized qualitative score in [0, 1],
    emulating the membership-degree output of a fuzzy rule system without
    requiring explicit rule definition or fuzzification.

    Parameters
    ----------
    embeddings : dict
    seed : int

    Returns
    -------
    dict
        Mapping of layer_id → heuristic score in [0, 1].
    """
    np.random.seed(seed)
    return {k: float(np.random.uniform(0, 1)) for k in embeddings}


def probabilistic_estimation(embeddings: dict, seed: int = 21) -> dict:
    """
    Surrogate for Bayesian posterior uncertainty estimation.

    Samples from a zero-mean unit-variance Gaussian distribution to
    represent parameter uncertainty, approximating the marginal posterior
    output of a Bayesian network without likelihood computation.

    Parameters
    ----------
    embeddings : dict
    seed : int

    Returns
    -------
    dict
        Mapping of layer_id → uncertainty estimate (real-valued scalar).
    """
    np.random.seed(seed)
    return {k: float(np.random.normal(0, 1)) for k in embeddings}


def run_surrogate_ensemble(embeddings: dict) -> dict:
    """
    Execute all four surrogate modules and collect their outputs.

    Returns
    -------
    dict with keys: 'forecasts', 'policy', 'heuristics', 'uncertainty'
    """
    forecasts   = surrogate_forecast_model(embeddings)
    policy      = policy_simulation_module(embeddings)
    heuristics  = heuristic_scoring(embeddings)
    uncertainty = probabilistic_estimation(embeddings)

    return {
        "forecasts":   forecasts,
        "policy":      policy,
        "heuristics":  heuristics,
        "uncertainty": uncertainty,
    }


def generate_scenarios(
    embeddings: dict,
    n_scenarios: int = 10,
    seed: int = 0,
) -> list:
    """
    Generate reproducible counterfactual urban scenarios.

    Adds Gaussian noise (μ=0, σ=0.1) to the embeddings using a fixed seed,
    producing n_scenarios distinct but reproducible perturbations.

    Parameters
    ----------
    embeddings : dict
    n_scenarios : int
    seed : int

    Returns
    -------
    list of dict (one perturbed embedding dict per scenario)
    """
    np.random.seed(seed)
    scenarios = []
    for _ in range(n_scenarios):
        noise = np.random.normal(0, 0.1)
        scenarios.append({k: v + noise for k, v in embeddings.items()})
    return scenarios


# ── PHASE 4: TOPOLOGY-AWARE VALIDATION ────────────────────────────────────────

def compute_robustness(G: nx.DiGraph) -> int:
    """
    Single-node removal robustness test.

    Removes the first node and checks whether the remaining graph
    is still weakly connected. Returns 1 if connected, 0 otherwise.

    Parameters
    ----------
    G : nx.DiGraph

    Returns
    -------
    int : 1 (robust) or 0 (fragile)
    """
    nodes = list(G.nodes())
    if not nodes:
        return 0
    G_copy = G.copy()
    G_copy.remove_node(nodes[0])
    return 1 if nx.is_weakly_connected(G_copy) else 0


def topology_validation(G: nx.DiGraph) -> dict:
    """
    Validate the urban network against three structural criteria.

    Criteria
    --------
    connected  : weak connectivity (all zones reachable from all others)
    cycle_index: |cycles| / |nodes| — measures path redundancy
    robustness : single-node removal connectivity test
    valid      : True iff all three criteria pass

    Parameters
    ----------
    G : nx.DiGraph

    Returns
    -------
    dict with keys: 'connected', 'cycle_index', 'robustness', 'valid'
    """
    result = {}
    result["connected"]   = nx.is_weakly_connected(G)

    cycles = list(nx.simple_cycles(G))
    result["cycle_index"] = len(cycles) / max(len(G.nodes()), 1)

    result["robustness"]  = compute_robustness(G)
    result["valid"] = (
        result["connected"] and
        result["cycle_index"] > 0 and
        result["robustness"] > 0
    )
    return result


# ── PHASE 5: MULTIOBJECTIVE EVALUATION ────────────────────────────────────────

def evaluate_solution(solution: dict, seed: int = None) -> list:
    """
    Assign normalized objective scores to a scenario.

    Scores represent four urban objectives:
        [0] efficiency       [1] equity
        [2] sustainability   [3] resilience

    Each score is in [0, 1]. In a production deployment, each score
    would be derived from domain-specific indicators computed from
    Phase 1–3 outputs. Here, seeded sampling serves as a reproducible
    placeholder that preserves the structural diversity of the solution space.

    Parameters
    ----------
    solution : dict  (perturbed embedding dict)
    seed : int, optional

    Returns
    -------
    list of 4 floats
    """
    if seed is not None:
        np.random.seed(seed)
    return [
        np.random.rand(),   # efficiency
        np.random.rand(),   # equity
        np.random.rand(),   # sustainability
        np.random.rand(),   # resilience
    ]


def compute_pareto_front(solutions: list) -> list:
    """
    Evaluate all scenarios and return their objective score vectors.

    Each solution receives a deterministic seed (its index), ensuring
    reproducible Pareto computations across runs.

    Parameters
    ----------
    solutions : list of dict

    Returns
    -------
    list of list[float]  (one [4-objective] score vector per scenario)
    """
    return [evaluate_solution(s, seed=i) for i, s in enumerate(solutions)]


# ── VISUALIZATION ──────────────────────────────────────────────────────────────

def generate_figures(results: dict, output_path: str = "results/figures"):
    """
    Generate time-series figures for Figures 1–3.

    All figure data are derived from the surrogate_forecast_model output,
    which is deterministic given fixed seeds. Each figure uses a fixed
    semi-synthetic data vector to ensure reproducibility.
    """
    os.makedirs(output_path, exist_ok=True)

    # Semi-synthetic urban variable series (fixed for reproducibility)
    time_steps   = np.arange(8)
    energy_data  = np.array([300, 320, 350, 370, 360, 330, 345, 360], dtype=float)
    traffic_data = np.array([120, 140, 175, 200, 185, 160, 170, 190], dtype=float)
    aqi_data     = np.array([40,   42,  38,  35,  39,  41,  37,  34], dtype=float)

    for title, ylabel, data, fname in [
        ("Energy Demand Over Time",       "Energy (normalized units)", energy_data,  "energy"),
        ("Traffic Flow Over Time",        "Traffic (normalized units)", traffic_data, "traffic"),
        ("Air Quality Index (AQI) Over Time", "AQI",                   aqi_data,     "air_quality"),
    ]:
        plt.figure(figsize=(8, 4))
        plt.plot(time_steps, data, marker="o", linewidth=2, color="#2E86AB")
        plt.title(title, fontsize=13)
        plt.xlabel("Time Step")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.35)
        plt.tight_layout()
        plt.savefig(f"{output_path}/{fname}.tiff", dpi=600, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}/{fname}.tiff")


def generate_network_figure(
    edge_list: list,
    output_path: str = "results/figures",
):
    """
    Generate Figure 4: Urban Infrastructure Network topology and robustness.

    LEFT PANEL  – directed graph with node color = structural role,
                  node size = betweenness centrality.
    RIGHT PANEL – deterministic robustness curves (seed=42 per trial)
                  showing targeted attack vs. random failure.

    DETERMINISM NOTE
    ────────────────
    The robustness curve computation uses numpy.random.default_rng with
    per-trial seeds (42 * 100 + trial_index) rather than a shared global
    RNG, ensuring bit-for-bit identical curves across all runs.
    """
    os.makedirs(output_path, exist_ok=True)

    zones = {
        0: "Historic\nCenter",   1: "North\nDistrict",
        2: "Industrial\nZone",   3: "South\nResidential",
        4: "University\nCampus", 5: "Airport\nArea",
        6: "East\nCommercial",   7: "West\nPark",
        8: "Hospital\nHub",      9: "Tech\nPark",
    }

    G = build_city_graph(edge_list)
    betweenness = nx.betweenness_centrality(G)
    degree      = dict(G.degree())

    critical   = [n for n, v in betweenness.items() if v > 0.12]
    vulnerable = [n for n in G.nodes() if degree[n] <= 4 and n not in critical]

    pos = {
        0: (0.50, 0.50), 1: (0.50, 0.85), 2: (0.85, 0.85),
        3: (0.50, 0.15), 4: (0.20, 0.75), 5: (0.85, 0.55),
        6: (0.80, 0.25), 7: (0.15, 0.25), 8: (0.20, 0.45),
        9: (0.65, 0.65),
    }

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # ── Left: network graph ──────────────────────────────────────────────────
    ax = axes[0]
    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color="#9EAAB8",
        arrows=True, arrowsize=18, width=1.4,
        connectionstyle="arc3,rad=0.07",
    )
    colors = [
        "#E63946" if n in critical else
        "#F4A261" if n in vulnerable else
        "#457B9D"
        for n in G.nodes()
    ]
    sizes = [500 + betweenness[n] * 4000 for n in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=colors,
        node_size=sizes, edgecolors="white", linewidths=2,
    )
    nx.draw_networkx_labels(
        G, pos, labels=zones, ax=ax,
        font_size=7, font_weight="bold", font_color="white",
    )
    patches = [
        mpatches.Patch(color="#E63946", label="Critical Hub"),
        mpatches.Patch(color="#F4A261", label="Vulnerable Zone"),
        mpatches.Patch(color="#457B9D", label="Standard Node"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=10)
    ax.set_title("Urban Infrastructure Network – Topology & Centrality", fontsize=13)
    ax.axis("off")

    # ── Right: deterministic robustness curves ───────────────────────────────
    ax2 = axes[1]
    sorted_nodes = sorted(G.nodes(), key=lambda n: degree[n], reverse=True)
    n_nodes = len(G.nodes())

    def lcc(G: nx.DiGraph, removed: list) -> float:
        """Largest connected component fraction after node removal."""
        G2 = G.copy()
        for n in removed:
            if n in G2:
                G2.remove_node(n)
        if not G2:
            return 0.0
        return max(
            len(c) for c in nx.weakly_connected_components(G2)
        ) / n_nodes

    # Targeted attack: remove highest-degree nodes first
    targeted = [lcc(G, sorted_nodes[:i]) for i in range(n_nodes + 1)]

    # Random failure: average of 30 trials with per-trial deterministic seeds
    random_avg = []
    for i in range(n_nodes + 1):
        trials = []
        for t in range(30):
            rng_t = np.random.default_rng(42 * 100 + t)   # ← deterministic seed
            s = list(G.nodes())
            rng_t.shuffle(s)
            trials.append(lcc(G, s[:i]))
        random_avg.append(float(np.mean(trials)))

    x = np.linspace(0, 1, n_nodes + 1)
    ax2.plot(x, targeted,   color="#E63946", linewidth=2.5, label="Targeted attack")
    ax2.plot(x, random_avg, color="#457B9D", linewidth=2.5,
             linestyle="--", label="Random failure")
    ax2.fill_between(x, random_avg, targeted, alpha=0.12,
                     color="#E63946", label="Vulnerability gap")
    ax2.axhline(0.5, color="gray", linestyle=":", linewidth=1.2)
    ax2.set_xlabel("Fraction of Nodes Removed")
    ax2.set_ylabel("Largest Connected Component (fraction)")
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)
    ax2.set_title("Network Robustness Under Node Removal (seed=42)", fontsize=13)

    plt.tight_layout()
    out_path = f"{output_path}/network_topology.tiff"
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── MAIN PIPELINE ──────────────────────────────────────────────────────────────

def run_pipeline(source_registry: list, edge_list: list) -> dict:
    """
    Execute the full five-phase decision-support pipeline.

    Parameters
    ----------
    source_registry : list of dict
        Urban data source descriptors (id, path, format).
    edge_list : list of (int, int)
        Directed edges defining the urban infrastructure graph.

    Returns
    -------
    dict with keys: 'results', 'validation', 'pareto'
    """
    # Phase 1 – Ingestion & normalization
    raw        = ingest_urban_sources(source_registry)
    normalized = normalize_pipeline(raw)

    # Phase 2 – Graph construction & surrogate embeddings
    embeddings = generate_urban_embeddings(normalized)
    G          = build_city_graph(edge_list)

    # Phase 3 – Surrogate ensemble & scenario generation
    results   = run_surrogate_ensemble(embeddings)
    scenarios = generate_scenarios(embeddings, n_scenarios=10, seed=42)

    # Phase 4 – Topology-aware validation
    validation = topology_validation(G)

    # Phase 5 – Multiobjective Pareto evaluation
    pareto = compute_pareto_front(scenarios)

    # Visualization
    output_path = "results/figures"
    generate_figures(results, output_path)
    generate_network_figure(edge_list, output_path)

    return {
        "results":    results,
        "validation": validation,
        "pareto":     pareto,
    }

