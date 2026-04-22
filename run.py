from src.pipeline import run_pipeline

def main():

    sources = [
        {"id": "mobility", "path": "data/sample.csv", "format": "csv"},
        {"id": "energy", "path": "data/sample.csv", "format": "csv"}
    ]

    edges = [(1,2), (2,3), (3,1), (3,4)]

    output = run_pipeline(sources, edges)

    print("Pipeline executed successfully")
    print(output)


if __name__ == "__main__":
    main()
