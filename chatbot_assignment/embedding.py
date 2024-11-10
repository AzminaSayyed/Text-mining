import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class HuggingFaceEmbedding:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        HuggingFaceEmbedding class for generating embeddings using Hugging Face models.

        Args:
            model_name (str): The name of the pre-trained model to use.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)
        self.model = AutoModel.from_pretrained(model_name)

    def __call__(self, text: str) -> np.ndarray:
        """
        Calculates the embedding for the given text using the pre-trained model.
        
        Args:
            text (str): The input text to calculate the embedding for.
        
        Returns:
            np.ndarray: The embedding vector for the input text as a 2D array.
        """
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Calculates the embedding for the given text using the pre-trained model.
        
        Args:
            text (str): The input text to calculate the embedding for.
        
        Returns:
            np.ndarray: The embedding vector for the input text as a 2D array.
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use only the first token's embedding (usually the [CLS] token)
        embeddings: np.ndarray = outputs.last_hidden_state[:, 0, :].numpy()

        # Ensure that we return a 2D array
        return embeddings
