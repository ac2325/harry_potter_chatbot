import os
import requests
from langchain.embeddings import OpenAIEmbeddings

def get_embedding_function():
    """Fetch embedding using OpenRouter API"""
    api_key = os.getenv("OPENAI_API_KEY")  # Load from environment
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Set it in your .env file.")

    def fetch_embedding(text):
        """Make a request to OpenRouter API for embeddings"""
        url = "https://openrouter.ai/api/v1/embeddings"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"input": text, "model": "openai-text-embedding-ada-002"}
        
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        else:
            raise ValueError(f"Error fetching embedding: {response.text}")

    return OpenAIEmbeddings(openai_api_key=api_key, embedding_func=fetch_embedding)
