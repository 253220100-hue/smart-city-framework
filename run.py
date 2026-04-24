from src.pipeline import run_pipeline

def main():

    sources = [
        {"id": "mobility", "path": "data/sample.csv", "format": "csv"},
        {"id": "energy", "path": "data/sample.csv", "format": "csv"}
    ]

    edges = [
    (0,1),(1,2),(2,0),
    (0,3),(3,4),(4,0),
    (1,5),(5,6),(6,1),
    (2,7),(7,8),(8,2),
    (3,9),(9,0),
    (4,6),(6,8),(8,4),
    (5,7),(7,9),(9,5)
    ]

    output = run_pipeline(sources, edges)

    print("Pipeline executed successfully")
    print(output)


if __name__ == "__main__":
    main()
