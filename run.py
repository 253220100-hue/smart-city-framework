if __name__ == "__main__":

    sources = [
        {"id": "mobility", "path": "mobility.csv", "format": "csv"},
        {"id": "energy", "path": "energy.csv", "format": "csv"}
    ]

    edges = [(1,2), (2,3), (3,1), (3,4)]

    output = run_pipeline(sources, edges)

    print("Pipeline executed successfully")
    print(output)
