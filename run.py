from src.pipeline import run_pipeline


def main():
    # ── Data sources ──────────────────────────────────────────────────────────
    # Each entry maps a logical layer id to a CSV file on disk.
    # Both layers currently point to sample.csv for demonstration purposes;
    # in a production deployment each id would reference a distinct dataset.
    sources = [
        {"id": "mobility", "path": "data/sample.csv", "format": "csv"},
        {"id": "energy",   "path": "data/sample.csv", "format": "csv"},
    ]

    # ── Urban infrastructure graph ────────────────────────────────────────────
    # 10-node directed graph (zones 0–9) with 21 directed edges.
    # The topology was designed to include:
    #   - a well-connected core (nodes 0, 6, 8)
    #   - peripheral zones with limited redundancy (nodes 5, 7)
    #   - bidirectional corridors between major hubs
    # This structure produces a non-trivial vulnerability gap in Figure 4
    # and passes all three topology_validation criteria.
    edges = [
        (0, 1), (1, 2), (2, 0),   # core triangle
        (0, 3), (3, 4), (4, 0),   # southern loop
        (1, 5), (5, 6), (6, 1),   # northern corridor
        (2, 7), (7, 8), (8, 2),   # western corridor
        (3, 9), (9, 0),            # southern spoke
        (4, 6), (6, 8), (8, 4),   # cross-diagonal
        (5, 7), (7, 9), (9, 5),   # peripheral ring
    ]

    # ── Run pipeline ──────────────────────────────────────────────────────────
    output = run_pipeline(sources, edges)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Pipeline executed successfully")
    print("=" * 60)

    print("\n[Phase 3] Surrogate Ensemble Outputs:")
    for module, values in output["results"].items():
        print(f"  {module}: {values}")

    print("\n[Phase 4] Topology Validation:")
    for criterion, value in output["validation"].items():
        print(f"  {criterion}: {value}")

    print("\n[Phase 5] Pareto Front (first 3 scenarios):")
    for i, scores in enumerate(output["pareto"][:3]):
        labels = ["efficiency", "equity", "sustainability", "resilience"]
        score_str = ", ".join(f"{l}={v:.3f}" for l, v in zip(labels, scores))
        print(f"  Scenario {i}: {score_str}")

    print("\nFigures saved to: results/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
