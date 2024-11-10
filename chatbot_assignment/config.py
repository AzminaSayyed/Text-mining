# assignment/config.py

from dataclasses import dataclass, field
import os
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

ENV_CONFIG = {
    'embedding_model': os.getenv('EMBEDDING_MODEL', "Snowflake/snowflake-arctic-embed-l"),
    'api_key': os.getenv('API_KEY', ''),
    'user_id': os.getenv('USER_ID', "User"),
    'max_tokens': int(os.getenv('MAX_TOKENS', 512)),
    'temperature': float(os.getenv('TEMPERATURE', 0.7)),
}


@dataclass
class ChatConfig:
    embedding_model: str = "Snowflake/snowflake-arctic-embed-l"
    user_id: str = "user"
    max_tokens: int = 512
    temperature: float = 0.7
    top_n: int = 3
    presence: Optional[float] = None
    repetition: Optional[float] = None
    debug: bool = False
    stop_sequences: List[str] = field(default_factory=lambda: ["You:", "<|im_end|>", "</s>"])

    @classmethod
    def from_env(cls):
        return cls(**ENV_CONFIG)
