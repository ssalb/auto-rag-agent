import os
import torch

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

class DuckDBConfig:
    # Database configuration
    DUCKDB_PATH: str = os.getenv("DB_DATA_PATH", "data/semantic_search.duckdb")
    EMBEDDING_DIM: int = 384
    VSS_METRIC: str = "cosine"  # Options: l2sq (default), cosine, dot, etc.
    VSS_M:int = 16  # HNSW M parameter (number of connections per layer)
    VSS_EF_CONSTRUCTION: int = 100  # Controls index build quality/time tradeoff