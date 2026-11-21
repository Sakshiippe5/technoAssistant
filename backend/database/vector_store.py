
from pathlib import Path
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import atexit


BASE_DIR = Path(__file__).resolve().parents[2]

QDRANT_PATH = BASE_DIR / "qdrant_storage"
COLLECTION_NAME = "troubleshooting_kb"
VECTOR_SIZE = 384


@lru_cache(maxsize=1)
def get_client():

    client = QdrantClient(path=str(QDRANT_PATH))

    atexit.register(client.close)

    return client


def create_collection():
    client = get_client()

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )

        print("Collection created successfully.")
    else:
        print("Collection already exists.")

    return client
