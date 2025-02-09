import streamlit as st
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function

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

            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)

            model = Ollama(model="mistral")
            response_text = model.invoke(prompt)
            
            return response_text, [doc.metadata.get("id", None) for doc, _score in results]

        response, sources = query_rag(query)
        st.subheader("Response")
        st.write(response)
        st.subheader("Sources")
        st.write(sources)
