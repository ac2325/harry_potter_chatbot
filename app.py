import streamlit as st
import os
import requests
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ✅ Load API key from Streamlit secrets
api_key = st.secrets.get("OPENROUTER_API_KEY")
if not api_key:
    st.error("🚨 OPENROUTER_API_KEY is missing! Please set it in Streamlit secrets.")
    st.stop()

# ✅ Define paths and prompt template
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🔍 Harry Potter Chatbot")
st.write("Ask any question about Harry Potter.")

# Text input for query
query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if not query:
        st.warning("Enter your question.")
    else:
        def query_rag(query_text):
            embedding_function = get_embedding_function()
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            results = db.similarity_search_with_score(query_text, k=5)

            # Build context from retrieved docs
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)

            # ✅ Call OpenRouter API for LLM response
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "openai/gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are an expert on Harry Potter."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
            }

            response = requests.post(url, headers=headers, json=payload)

            if response.status_code == 200:
                response_text = response.json()["choices"][0]["message"]["content"]
            else:
                response_text = f"❌ Error: {response.text}"

            return response_text, [doc.metadata.get("id", None) for doc, _ in results]

        response, sources = query_rag(query)
        st.subheader("Response")
        st.write(response)