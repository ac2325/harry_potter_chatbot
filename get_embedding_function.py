import requests

import streamlit as st
api_key = st.secrets["OPENROUTER_API_KEY"]

# Example function to call OpenRouter
def get_embedding_function(text):
    """Fetch embedding from OpenRouter API"""
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",  # Use secret API key
        "Content-Type": "application/json"
    }
    payload = {"input": text, "model": "openai-text-embedding-ada-002"}
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()["data"][0]["embedding"]
    else:
        print(f"Error fetching embedding: {response.text}")
        return None
