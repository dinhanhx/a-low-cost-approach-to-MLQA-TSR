import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Datatype,
    Distance,
    HnswConfigDiff,
    MultiVectorComparator,
    MultiVectorConfig,
    VectorParams,
)

load_dotenv()


DEFAULT_VECTORS_CONFIG = {
    # by jina-embeddings-v3 dim size
    "text_vector": VectorParams(
        size=1024,
        distance=Distance.COSINE,
        datatype=Datatype.FLOAT32,
    ),
    # by nvidia/C-RADIOv2-B dim size
    "image_general_feature_vector": VectorParams(
        size=2304,
        distance=Distance.COSINE,
        datatype=Datatype.FLOAT32,
    ),
    # by nvidia/C-RADIOv2-B dim size
    "image_object_feature_list_vector": VectorParams(
        size=2304,
        distance=Distance.COSINE,
        multivector_config=MultiVectorConfig(
            comparator=MultiVectorComparator.MAX_SIM,
        ),
        hnsw_config=HnswConfigDiff(m=0)
    )
}


class Qdrant:
    def __init__(self) -> None:
        self.client = QdrantClient(
            url=os.getenv("QDRANT_HOST", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY", None),
        )

    def check_health(self) -> bool:
        try:
            self.client.info()
            return True
        except Exception as e:
            return False

    def init_collection(
        self,
        collection_name: str,
        vectors_config: dict | None = None,
    ):
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=DEFAULT_VECTORS_CONFIG if vectors_config is None else vectors_config,
            )
            return True
        return False


if __name__ == "__main__":
    print("Connecting to Qdrant...")
    qdrant = Qdrant()
    if qdrant.check_health():
        print("Connection successful!")
    else:
        print("Connection failed!")
