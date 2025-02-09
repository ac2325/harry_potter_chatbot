import requests
import streamlit as st
from langchain.embeddings.base import Embeddings  # ✅ Required for compatibility

# ✅ Load API key
api_key = st.secrets["OPENROUTER_API_KEY"]

class OpenRouterEmbeddings(Embeddings):
    """Wrapper for OpenRouter API embedding calls that implements LangChain's interface."""

    def embed_query(self, text):
        """Fetch embedding from OpenRouter API."""
        url = "https://openrouter.ai/api/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {"input": text, "model": "openai/text-embedding-ada-002"}
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        else:
            print(f"❌ Error fetching embedding: {response.text}")
            return None

def get_embedding_function():
    """Return an instance of OpenRouterEmbeddings that can be used by Chroma."""
    return OpenRouterEmbeddings()
