import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Callable, Optional

from .config import ChatConfig
from .embedding import HuggingFaceEmbedding

class DocumentQueryModel:
    """
    A simplified Document Query Model that uses a single collection with a flexible pipeline strategy 
    for document indexing and query preprocessing.
    """

    def __init__(self, data: pd.DataFrame, db_path: str, embedding_function: Callable[[str], np.ndarray]):
        self.data = data
        self.db_path = db_path
        self.ef = embedding_function

    @classmethod
    def from_config(cls, config: ChatConfig) -> 'DocumentQueryModel':
        if not config.db_path.endswith(".pkl"):
            raise ValueError(f"Invalid file format in '{config.db_path}'. Expected .pkl file.")
        
        if os.path.exists(config.db_path):
            data = pd.read_pickle(config.db_path)
        else:
            data = cls.new()
        
        if not all(col in data.columns for col in ["embedding", "content"]):
            raise ValueError("Invalid data format. Expected columns: 'embedding', 'content'")
        if data.index.name != "doc_id":
            raise ValueError("Invalid data format. Expected index name: 'doc_id'")

        embedding_function = HuggingFaceEmbedding(model_name=config.embedding_model)

        return cls(data=data, db_path=config.db_path, embedding_function=embedding_function)

    @classmethod
    def new(cls) -> pd.DataFrame:
        data = pd.DataFrame(columns=["embedding", "content"])
        data.index.name = "doc_id"
        return data

    def load_jsonl(self, file_path: str, id_key: str, content_key: str) -> None:
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    # print(f"Loaded data: {data}")  # Debugging line
                    # # Check for keys
                    # if id_key not in data or content_key not in data:
                    #     print(f"Missing keys in data: {data}")
                    #     continue
                
                    self.insert(data[id_key], content=data[content_key])
                except (ValueError, KeyError) as ve:
                    print(f"Insert Error: {ve}")

    def save(self):
        self.data.to_pickle(self.db_path)

    @property
    def document_count(self) -> int:
        return self.data.shape[0]

    def insert(self, doc_id: str, content: str) -> np.ndarray:
        embedding = self.ef(content)
        
        # Ensure embedding is a 2D array
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        self.data.loc[doc_id] = [embedding, content]
        return embedding

    def query(self, query_text: str, top_n: int = 5) -> pd.DataFrame:
        query = self.ef(query_text)

        # Ensure query is a 2D array
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if self.data.empty:
            return pd.DataFrame()
        
        embeddings = np.vstack(self.data['embedding'].values)

        search = self.data.copy()
        search['distance'] = cosine_similarity(query, embeddings).flatten()
        return search.sort_values(by='distance', ascending=False).head(top_n)

    def get_document(self, doc_id: str) -> Optional[str]:
        try:
            result = self.data.loc[doc_id]
            return result['content']
        except KeyError:
            return None
    
    def clear(self):
        self.data = self.new()
