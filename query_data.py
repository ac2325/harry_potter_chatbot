import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.openai import OpenAI  # ✅ Use OpenRouter-compatible model

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    """Parse CLI arguments and execute the query."""
    parser = argparse.ArgumentParser(description="RAG-based chatbot using OpenRouter.")
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    
    response = query_rag(args.query_text)
    print("\n💬 Response:\n", response)

def query_rag(query_text: str):
    """Query the RAG system using OpenRouter embeddings & OpenAI models."""
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search for relevant documents
    results = db.similarity_search_with_score(query_text, k=5)
    
    if not results:
        print("⚠️ No relevant documents found.")
        return "I couldn't find an answer in the database."

    # Format the retrieved context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Use OpenRouter API to generate a response
    model = OpenAI(model="gpt-4") 
    try:
        response_text = model.invoke(prompt)
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return "I'm having trouble generating a response."

    # Extract and display sources
    sources = [doc.metadata.get("id", "Unknown") for doc, _ in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    
    return formatted_response

if __name__ == "__main__":
    main()
