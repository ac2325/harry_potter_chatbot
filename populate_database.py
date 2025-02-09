import argparse
import os
import shutil
import requests
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader 
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv 
load_dotenv()

CHROMA_PATH = "chroma_db"
DATA_PATH = "data"
OPENROUTER_API_KEY = os.getenv("OPENAI_API_KEY")


def get_embedding_function(text):
    """Fetch embedding from OpenRouter API"""
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"input": text, "model": "openai-text-embedding-ada-002"}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["data"][0]["embedding"]
    else:
        print(f"Error fetching embedding: {response.text}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            file_path = os.path.join(DATA_PATH, filename)
            print(f"📄 Loading: {filename}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function)
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
    
    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
