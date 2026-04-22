from src.pipeline import run_pipeline

sources = [
    {"id": "mobility", "path": "data/sample.csv", "format": "csv"}
]

edges = [(1,2), (2,3), (3,1)]

output = run_pipeline(sources, edges)

print(output)
